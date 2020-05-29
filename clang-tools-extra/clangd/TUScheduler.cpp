//===--- TUScheduler.cpp -----------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TUScheduler manages a worker per active file. This ASTWorker processes
// updates (modifications to file contents) and reads (actions performed on
// preamble/AST) to the file.
//
// Each ASTWorker owns a dedicated thread to process updates and reads to the
// relevant file. Any request gets queued in FIFO order to be processed by that
// thread.
//
// An update request replaces current praser inputs to ensure any subsequent
// read sees the version of the file they were requested. It will also issue a
// build for new inputs.
//
// ASTWorker processes the file in two parts, a preamble and a main-file
// section. A preamble can be reused between multiple versions of the file until
// invalidated by a modification to a header, compile commands or modification
// to relevant part of the current file. Such a preamble is called compatible.
// An update is considered dead if no read was issued for that version and
// diagnostics weren't requested by client or could be generated for a later
// version of the file. ASTWorker eliminates such requests as they are
// redundant.
//
// In the presence of stale (non-compatible) preambles, ASTWorker won't publish
// diagnostics for update requests. Read requests will be served with ASTs build
// with stale preambles, unless the read is picky and requires a compatible
// preamble. In such cases it will block until new preamble is built.
//
// ASTWorker owns a PreambleThread for building preambles. If the preamble gets
// invalidated by an update request, a new build will be requested on
// PreambleThread. Since PreambleThread only receives requests for newer
// versions of the file, in case of multiple requests it will only build the
// last one and skip requests in between. Unless client force requested
// diagnostics(WantDiagnostics::Yes).
//
// When a new preamble is built, a "golden" AST is immediately built from that
// version of the file. This ensures diagnostics get updated even if the queue
// is full.
//
// Some read requests might just need preamble. Since preambles can be read
// concurrently, ASTWorker runs these requests on their own thread. These
// requests will receive latest build preamble, which might possibly be stale.

#include "TUScheduler.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "GlobalCompilationDatabase.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "index/CanonicalIncludes.h"
#include "support/Cancellation.h"
#include "support/Context.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/Threading.h"
#include "support/Trace.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Threading.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
using std::chrono::steady_clock;

namespace {
class ASTWorker;
} // namespace

static clang::clangd::Key<std::string> kFileBeingProcessed;

llvm::Optional<llvm::StringRef> TUScheduler::getFileBeingProcessedInContext() {
  if (auto *File = Context::current().get(kFileBeingProcessed))
    return llvm::StringRef(*File);
  return None;
}

/// An LRU cache of idle ASTs.
/// Because we want to limit the overall number of these we retain, the cache
/// owns ASTs (and may evict them) while their workers are idle.
/// Workers borrow ASTs when active, and return them when done.
class TUScheduler::ASTCache {
public:
  using Key = const ASTWorker *;

  ASTCache(unsigned MaxRetainedASTs) : MaxRetainedASTs(MaxRetainedASTs) {}

  /// Returns result of getUsedBytes() for the AST cached by \p K.
  /// If no AST is cached, 0 is returned.
  std::size_t getUsedBytes(Key K) {
    std::lock_guard<std::mutex> Lock(Mut);
    auto It = findByKey(K);
    if (It == LRU.end() || !It->second)
      return 0;
    return It->second->getUsedBytes();
  }

  /// Store the value in the pool, possibly removing the last used AST.
  /// The value should not be in the pool when this function is called.
  void put(Key K, std::unique_ptr<ParsedAST> V) {
    std::unique_lock<std::mutex> Lock(Mut);
    assert(findByKey(K) == LRU.end());

    LRU.insert(LRU.begin(), {K, std::move(V)});
    if (LRU.size() <= MaxRetainedASTs)
      return;
    // We're past the limit, remove the last element.
    std::unique_ptr<ParsedAST> ForCleanup = std::move(LRU.back().second);
    LRU.pop_back();
    // Run the expensive destructor outside the lock.
    Lock.unlock();
    ForCleanup.reset();
  }

  /// Returns the cached value for \p K, or llvm::None if the value is not in
  /// the cache anymore. If nullptr was cached for \p K, this function will
  /// return a null unique_ptr wrapped into an optional.
  /// If \p AccessMetric is set records whether there was a hit or miss.
  llvm::Optional<std::unique_ptr<ParsedAST>>
  take(Key K, const trace::Metric *AccessMetric = nullptr) {
    // Record metric after unlocking the mutex.
    std::unique_lock<std::mutex> Lock(Mut);
    auto Existing = findByKey(K);
    if (Existing == LRU.end()) {
      if (AccessMetric)
        AccessMetric->record(1, "miss");
      return None;
    }
    if (AccessMetric)
      AccessMetric->record(1, "hit");
    std::unique_ptr<ParsedAST> V = std::move(Existing->second);
    LRU.erase(Existing);
    // GCC 4.8 fails to compile `return V;`, as it tries to call the copy
    // constructor of unique_ptr, so we call the move ctor explicitly to avoid
    // this miscompile.
    return llvm::Optional<std::unique_ptr<ParsedAST>>(std::move(V));
  }

private:
  using KVPair = std::pair<Key, std::unique_ptr<ParsedAST>>;

  std::vector<KVPair>::iterator findByKey(Key K) {
    return llvm::find_if(LRU, [K](const KVPair &P) { return P.first == K; });
  }

  std::mutex Mut;
  unsigned MaxRetainedASTs;
  /// Items sorted in LRU order, i.e. first item is the most recently accessed
  /// one.
  std::vector<KVPair> LRU; /* GUARDED_BY(Mut) */
};

namespace {
/// Threadsafe manager for updating a TUStatus and emitting it after each
/// update.
class SynchronizedTUStatus {
public:
  SynchronizedTUStatus(PathRef FileName, ParsingCallbacks &Callbacks)
      : FileName(FileName), Callbacks(Callbacks) {}

  void update(llvm::function_ref<void(TUStatus &)> Mutator) {
    std::lock_guard<std::mutex> Lock(StatusMu);
    Mutator(Status);
    emitStatusLocked();
  }

  /// Prevents emitting of further updates.
  void stop() {
    std::lock_guard<std::mutex> Lock(StatusMu);
    CanPublish = false;
  }

private:
  void emitStatusLocked() {
    if (CanPublish)
      Callbacks.onFileUpdated(FileName, Status);
  }

  const Path FileName;

  std::mutex StatusMu;
  TUStatus Status;
  bool CanPublish = true;
  ParsingCallbacks &Callbacks;
};

/// Responsible for building preambles. Whenever the thread is idle and the
/// preamble is outdated, it starts to build a fresh preamble from the latest
/// inputs. If RunSync is true, preambles are built synchronously in update()
/// instead.
class PreambleThread {
public:
  PreambleThread(llvm::StringRef FileName, ParsingCallbacks &Callbacks,
                 bool StorePreambleInMemory, bool RunSync,
                 SynchronizedTUStatus &Status, ASTWorker &AW)
      : FileName(FileName), Callbacks(Callbacks),
        StoreInMemory(StorePreambleInMemory), RunSync(RunSync), Status(Status),
        ASTPeer(AW) {}

  /// It isn't guaranteed that each requested version will be built. If there
  /// are multiple update requests while building a preamble, only the last one
  /// will be built.
  void update(std::unique_ptr<CompilerInvocation> CI, ParseInputs PI,
              std::vector<Diag> CIDiags, WantDiagnostics WantDiags) {
    Request Req = {std::move(CI), std::move(PI), std::move(CIDiags), WantDiags,
                   Context::current().clone()};
    if (RunSync) {
      build(std::move(Req));
      Status.update([](TUStatus &Status) {
        Status.PreambleActivity = PreambleAction::Idle;
      });
      return;
    }
    {
      std::unique_lock<std::mutex> Lock(Mutex);
      // If NextReq was requested with WantDiagnostics::Yes we cannot just drop
      // that on the floor. Block until we start building it. This won't
      // dead-lock as we are blocking the caller thread, while builds continue
      // on preamble thread.
      ReqCV.wait(Lock, [this] {
        return !NextReq || NextReq->WantDiags != WantDiagnostics::Yes;
      });
      NextReq = std::move(Req);
    }
    // Let the worker thread know there's a request, notify_one is safe as there
    // should be a single worker thread waiting on it.
    ReqCV.notify_all();
  }

  void run() {
    while (true) {
      {
        std::unique_lock<std::mutex> Lock(Mutex);
        assert(!CurrentReq && "Already processing a request?");
        // Wait until stop is called or there is a request.
        ReqCV.wait(Lock, [this] { return NextReq || Done; });
        if (Done)
          break;
        CurrentReq = std::move(*NextReq);
        NextReq.reset();
      }

      {
        WithContext Guard(std::move(CurrentReq->Ctx));
        // Build the preamble and let the waiters know about it.
        build(std::move(*CurrentReq));
      }
      bool IsEmpty = false;
      {
        std::lock_guard<std::mutex> Lock(Mutex);
        CurrentReq.reset();
        IsEmpty = !NextReq.hasValue();
      }
      if (IsEmpty) {
        // We don't perform this above, before waiting for a request to make
        // tests more deterministic. As there can be a race between this thread
        // and client thread(clangdserver).
        Status.update([](TUStatus &Status) {
          Status.PreambleActivity = PreambleAction::Idle;
        });
      }
      ReqCV.notify_all();
    }
    dlog("Preamble worker for {0} stopped", FileName);
  }

  /// Signals the run loop to exit.
  void stop() {
    dlog("Preamble worker for {0} received stop", FileName);
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      Done = true;
      NextReq.reset();
    }
    // Let the worker thread know that it should stop.
    ReqCV.notify_all();
  }

  bool blockUntilIdle(Deadline Timeout) const {
    std::unique_lock<std::mutex> Lock(Mutex);
    return wait(Lock, ReqCV, Timeout, [&] { return !NextReq && !CurrentReq; });
  }

private:
  /// Holds inputs required for building a preamble. CI is guaranteed to be
  /// non-null.
  struct Request {
    std::unique_ptr<CompilerInvocation> CI;
    ParseInputs Inputs;
    std::vector<Diag> CIDiags;
    WantDiagnostics WantDiags;
    Context Ctx;
  };

  bool isDone() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return Done;
  }

  /// Builds a preamble for \p Req, might reuse LatestBuild if possible.
  /// Notifies ASTWorker after build finishes.
  void build(Request Req);

  mutable std::mutex Mutex;
  bool Done = false;                  /* GUARDED_BY(Mutex) */
  llvm::Optional<Request> NextReq;    /* GUARDED_BY(Mutex) */
  llvm::Optional<Request> CurrentReq; /* GUARDED_BY(Mutex) */
  // Signaled whenever a thread populates NextReq or worker thread builds a
  // Preamble.
  mutable std::condition_variable ReqCV; /* GUARDED_BY(Mutex) */
  // Accessed only by preamble thread.
  std::shared_ptr<const PreambleData> LatestBuild;

  const Path FileName;
  ParsingCallbacks &Callbacks;
  const bool StoreInMemory;
  const bool RunSync;

  SynchronizedTUStatus &Status;
  ASTWorker &ASTPeer;
};

class ASTWorkerHandle;

/// Owns one instance of the AST, schedules updates and reads of it.
/// Also responsible for building and providing access to the preamble.
/// Each ASTWorker processes the async requests sent to it on a separate
/// dedicated thread.
/// The ASTWorker that manages the AST is shared by both the processing thread
/// and the TUScheduler. The TUScheduler should discard an ASTWorker when
/// remove() is called, but its thread may be busy and we don't want to block.
/// So the workers are accessed via an ASTWorkerHandle. Destroying the handle
/// signals the worker to exit its run loop and gives up shared ownership of the
/// worker.
class ASTWorker {
  friend class ASTWorkerHandle;
  ASTWorker(PathRef FileName, const GlobalCompilationDatabase &CDB,
            TUScheduler::ASTCache &LRUCache, Semaphore &Barrier, bool RunSync,
            const TUScheduler::Options &Opts, ParsingCallbacks &Callbacks);

public:
  /// Create a new ASTWorker and return a handle to it.
  /// The processing thread is spawned using \p Tasks. However, when \p Tasks
  /// is null, all requests will be processed on the calling thread
  /// synchronously instead. \p Barrier is acquired when processing each
  /// request, it is used to limit the number of actively running threads.
  static ASTWorkerHandle create(PathRef FileName,
                                const GlobalCompilationDatabase &CDB,
                                TUScheduler::ASTCache &IdleASTs,
                                AsyncTaskRunner *Tasks, Semaphore &Barrier,
                                const TUScheduler::Options &Opts,
                                ParsingCallbacks &Callbacks);
  ~ASTWorker();

  void update(ParseInputs Inputs, WantDiagnostics);
  void
  runWithAST(llvm::StringRef Name,
             llvm::unique_function<void(llvm::Expected<InputsAndAST>)> Action,
             TUScheduler::ASTActionInvalidation);
  bool blockUntilIdle(Deadline Timeout) const;

  std::shared_ptr<const PreambleData> getPossiblyStalePreamble() const;

  /// Used to inform ASTWorker about a new preamble build by PreambleThread.
  /// Diagnostics are only published through this callback. This ensures they
  /// are always for newer versions of the file, as the callback gets called in
  /// the same order as update requests.
  void updatePreamble(std::unique_ptr<CompilerInvocation> CI, ParseInputs PI,
                      std::shared_ptr<const PreambleData> Preamble,
                      std::vector<Diag> CIDiags, WantDiagnostics WantDiags);

  /// Obtain a preamble reflecting all updates so far. Threadsafe.
  /// It may be delivered immediately, or later on the worker thread.
  void getCurrentPreamble(
      llvm::unique_function<void(std::shared_ptr<const PreambleData>)>);
  /// Returns compile command from the current file inputs.
  tooling::CompileCommand getCurrentCompileCommand() const;

  /// Wait for the first build of preamble to finish. Preamble itself can be
  /// accessed via getPossiblyStalePreamble(). Note that this function will
  /// return after an unsuccessful build of the preamble too, i.e. result of
  /// getPossiblyStalePreamble() can be null even after this function returns.
  void waitForFirstPreamble() const;

  TUScheduler::FileStats stats() const;
  bool isASTCached() const;

private:
  /// Publishes diagnostics for \p Inputs. It will build an AST or reuse the
  /// cached one if applicable. Assumes LatestPreamble is compatible for \p
  /// Inputs.
  void generateDiagnostics(std::unique_ptr<CompilerInvocation> Invocation,
                           ParseInputs Inputs, std::vector<Diag> CIDiags);

  // Must be called exactly once on processing thread. Will return after
  // stop() is called on a separate thread and all pending requests are
  // processed.
  void run();
  /// Signal that run() should finish processing pending requests and exit.
  void stop();
  /// Adds a new task to the end of the request queue.
  void startTask(llvm::StringRef Name, llvm::unique_function<void()> Task,
                 llvm::Optional<WantDiagnostics> UpdateType,
                 TUScheduler::ASTActionInvalidation);

  /// Determines the next action to perform.
  /// All actions that should never run are discarded.
  /// Returns a deadline for the next action. If it's expired, run now.
  /// scheduleLocked() is called again at the deadline, or if requests arrive.
  Deadline scheduleLocked();
  /// Should the first task in the queue be skipped instead of run?
  bool shouldSkipHeadLocked() const;

  struct Request {
    llvm::unique_function<void()> Action;
    std::string Name;
    steady_clock::time_point AddTime;
    Context Ctx;
    llvm::Optional<WantDiagnostics> UpdateType;
    TUScheduler::ASTActionInvalidation InvalidationPolicy;
    Canceler Invalidate;
  };

  /// Handles retention of ASTs.
  TUScheduler::ASTCache &IdleASTs;
  const bool RunSync;
  /// Time to wait after an update to see whether another update obsoletes it.
  const DebouncePolicy UpdateDebounce;
  /// File that ASTWorker is responsible for.
  const Path FileName;
  const GlobalCompilationDatabase &CDB;
  /// Callback invoked when preamble or main file AST is built.
  ParsingCallbacks &Callbacks;
  /// Latest build preamble for current TU.
  std::shared_ptr<const PreambleData> LatestPreamble;
  Notification BuiltFirstPreamble;

  Semaphore &Barrier;
  /// Whether the 'onMainAST' callback ran for the current FileInputs.
  bool RanASTCallback = false;
  /// Guards members used by both TUScheduler and the worker thread.
  mutable std::mutex Mutex;
  /// File inputs, currently being used by the worker.
  /// Writes and reads from unknown threads are locked. Reads from the worker
  /// thread are not locked, as it's the only writer.
  ParseInputs FileInputs; /* GUARDED_BY(Mutex) */
  /// Times of recent AST rebuilds, used for UpdateDebounce computation.
  llvm::SmallVector<DebouncePolicy::clock::duration, 8>
      RebuildTimes; /* GUARDED_BY(Mutex) */
  /// Set to true to signal run() to finish processing.
  bool Done;                              /* GUARDED_BY(Mutex) */
  std::deque<Request> Requests;           /* GUARDED_BY(Mutex) */
  std::queue<Request> PreambleRequests;   /* GUARDED_BY(Mutex) */
  llvm::Optional<Request> CurrentRequest; /* GUARDED_BY(Mutex) */
  mutable std::condition_variable RequestsCV;
  Notification ReceivedPreamble;
  /// Guards the callback that publishes results of AST-related computations
  /// (diagnostics, highlightings) and file statuses.
  std::mutex PublishMu;
  // Used to prevent remove document + add document races that lead to
  // out-of-order callbacks for publishing results of onMainAST callback.
  //
  // The lifetime of the old/new ASTWorkers will overlap, but their handles
  // don't. When the old handle is destroyed, the old worker will stop reporting
  // any results to the user.
  bool CanPublishResults = true; /* GUARDED_BY(PublishMu) */
  std::atomic<unsigned> ASTBuildCount = {0};
  std::atomic<unsigned> PreambleBuildCount = {0};

  SynchronizedTUStatus Status;
  PreambleThread PreamblePeer;
};

/// A smart-pointer-like class that points to an active ASTWorker.
/// In destructor, signals to the underlying ASTWorker that no new requests will
/// be sent and the processing loop may exit (after running all pending
/// requests).
class ASTWorkerHandle {
  friend class ASTWorker;
  ASTWorkerHandle(std::shared_ptr<ASTWorker> Worker)
      : Worker(std::move(Worker)) {
    assert(this->Worker);
  }

public:
  ASTWorkerHandle(const ASTWorkerHandle &) = delete;
  ASTWorkerHandle &operator=(const ASTWorkerHandle &) = delete;
  ASTWorkerHandle(ASTWorkerHandle &&) = default;
  ASTWorkerHandle &operator=(ASTWorkerHandle &&) = default;

  ~ASTWorkerHandle() {
    if (Worker)
      Worker->stop();
  }

  ASTWorker &operator*() {
    assert(Worker && "Handle was moved from");
    return *Worker;
  }

  ASTWorker *operator->() {
    assert(Worker && "Handle was moved from");
    return Worker.get();
  }

  /// Returns an owning reference to the underlying ASTWorker that can outlive
  /// the ASTWorkerHandle. However, no new requests to an active ASTWorker can
  /// be schedule via the returned reference, i.e. only reads of the preamble
  /// are possible.
  std::shared_ptr<const ASTWorker> lock() { return Worker; }

private:
  std::shared_ptr<ASTWorker> Worker;
};

ASTWorkerHandle ASTWorker::create(PathRef FileName,
                                  const GlobalCompilationDatabase &CDB,
                                  TUScheduler::ASTCache &IdleASTs,
                                  AsyncTaskRunner *Tasks, Semaphore &Barrier,
                                  const TUScheduler::Options &Opts,
                                  ParsingCallbacks &Callbacks) {
  std::shared_ptr<ASTWorker> Worker(new ASTWorker(
      FileName, CDB, IdleASTs, Barrier, /*RunSync=*/!Tasks, Opts, Callbacks));
  if (Tasks) {
    Tasks->runAsync("ASTWorker:" + llvm::sys::path::filename(FileName),
                    [Worker]() { Worker->run(); });
    Tasks->runAsync("PreambleWorker:" + llvm::sys::path::filename(FileName),
                    [Worker]() { Worker->PreamblePeer.run(); });
  }

  return ASTWorkerHandle(std::move(Worker));
}

ASTWorker::ASTWorker(PathRef FileName, const GlobalCompilationDatabase &CDB,
                     TUScheduler::ASTCache &LRUCache, Semaphore &Barrier,
                     bool RunSync, const TUScheduler::Options &Opts,
                     ParsingCallbacks &Callbacks)
    : IdleASTs(LRUCache), RunSync(RunSync), UpdateDebounce(Opts.UpdateDebounce),
      FileName(FileName), CDB(CDB), Callbacks(Callbacks), Barrier(Barrier),
      Done(false), Status(FileName, Callbacks),
      PreamblePeer(FileName, Callbacks, Opts.StorePreamblesInMemory,
                   RunSync || !Opts.AsyncPreambleBuilds, Status, *this) {
  // Set a fallback command because compile command can be accessed before
  // `Inputs` is initialized. Other fields are only used after initialization
  // from client inputs.
  FileInputs.CompileCommand = CDB.getFallbackCommand(FileName);
}

ASTWorker::~ASTWorker() {
  // Make sure we remove the cached AST, if any.
  IdleASTs.take(this);
#ifndef NDEBUG
  std::lock_guard<std::mutex> Lock(Mutex);
  assert(Done && "handle was not destroyed");
  assert(Requests.empty() && !CurrentRequest &&
         "unprocessed requests when destroying ASTWorker");
#endif
}

void ASTWorker::update(ParseInputs Inputs, WantDiagnostics WantDiags) {
  std::string TaskName = llvm::formatv("Update ({0})", Inputs.Version);
  auto Task = [=]() mutable {
    // Get the actual command as `Inputs` does not have a command.
    // FIXME: some build systems like Bazel will take time to preparing
    // environment to build the file, it would be nice if we could emit a
    // "PreparingBuild" status to inform users, it is non-trivial given the
    // current implementation.
    if (auto Cmd = CDB.getCompileCommand(FileName))
      Inputs.CompileCommand = *Cmd;
    else
      // FIXME: consider using old command if it's not a fallback one.
      Inputs.CompileCommand = CDB.getFallbackCommand(FileName);

    bool InputsAreTheSame =
        std::tie(FileInputs.CompileCommand, FileInputs.Contents) ==
        std::tie(Inputs.CompileCommand, Inputs.Contents);
    // Cached AST is invalidated.
    if (!InputsAreTheSame) {
      IdleASTs.take(this);
      RanASTCallback = false;
    }

    // Update current inputs so that subsequent reads can see them.
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      FileInputs = Inputs;
    }

    log("ASTWorker building file {0} version {1} with command {2}\n[{3}]\n{4}",
        FileName, Inputs.Version, Inputs.CompileCommand.Heuristic,
        Inputs.CompileCommand.Directory,
        llvm::join(Inputs.CompileCommand.CommandLine, " "));

    StoreDiags CompilerInvocationDiagConsumer;
    std::vector<std::string> CC1Args;
    std::unique_ptr<CompilerInvocation> Invocation = buildCompilerInvocation(
        Inputs, CompilerInvocationDiagConsumer, &CC1Args);
    // Log cc1 args even (especially!) if creating invocation failed.
    if (!CC1Args.empty())
      vlog("Driver produced command: cc1 {0}", llvm::join(CC1Args, " "));
    std::vector<Diag> CompilerInvocationDiags =
        CompilerInvocationDiagConsumer.take();
    if (!Invocation) {
      elog("Could not build CompilerInvocation for file {0}", FileName);
      // Remove the old AST if it's still in cache.
      IdleASTs.take(this);
      RanASTCallback = false;
      // Report the diagnostics we collected when parsing the command line.
      Callbacks.onFailedAST(FileName, Inputs.Version,
                            std::move(CompilerInvocationDiags),
                            [&](llvm::function_ref<void()> Publish) {
                              // Ensure we only publish results from the worker
                              // if the file was not removed, making sure there
                              // are not race conditions.
                              std::lock_guard<std::mutex> Lock(PublishMu);
                              if (CanPublishResults)
                                Publish();
                            });
      // Make sure anyone waiting for the preamble gets notified it could not be
      // built.
      BuiltFirstPreamble.notify();
      return;
    }

    PreamblePeer.update(std::move(Invocation), std::move(Inputs),
                        std::move(CompilerInvocationDiags), WantDiags);
    // Block until first preamble is ready, as patching an empty preamble would
    // imply rebuilding it from scratch.
    // This isn't the natural place to block, rather where the preamble would be
    // consumed. But that's too late, we'd be running on the worker thread with
    // the PreambleTask scheduled and so we'd deadlock.
    ReceivedPreamble.wait();
    return;
  };
  startTask(TaskName, std::move(Task), WantDiags, TUScheduler::NoInvalidation);
}

void ASTWorker::runWithAST(
    llvm::StringRef Name,
    llvm::unique_function<void(llvm::Expected<InputsAndAST>)> Action,
    TUScheduler::ASTActionInvalidation Invalidation) {
  // Tracks ast cache accesses for read operations.
  static constexpr trace::Metric ASTAccessForRead(
      "ast_access_read", trace::Metric::Counter, "result");
  auto Task = [=, Action = std::move(Action)]() mutable {
    if (auto Reason = isCancelled())
      return Action(llvm::make_error<CancelledError>(Reason));
    llvm::Optional<std::unique_ptr<ParsedAST>> AST =
        IdleASTs.take(this, &ASTAccessForRead);
    if (!AST) {
      StoreDiags CompilerInvocationDiagConsumer;
      std::unique_ptr<CompilerInvocation> Invocation =
          buildCompilerInvocation(FileInputs, CompilerInvocationDiagConsumer);
      // Try rebuilding the AST.
      vlog("ASTWorker rebuilding evicted AST to run {0}: {1} version {2}", Name,
           FileName, FileInputs.Version);
      // FIXME: We might need to build a patched ast once preamble thread starts
      // running async. Currently getPossiblyStalePreamble below will always
      // return a compatible preamble as ASTWorker::update blocks.
      llvm::Optional<ParsedAST> NewAST;
      if (Invocation) {
        NewAST = ParsedAST::build(FileName, FileInputs, std::move(Invocation),
                                  CompilerInvocationDiagConsumer.take(),
                                  getPossiblyStalePreamble());
        ++ASTBuildCount;
      }
      AST = NewAST ? std::make_unique<ParsedAST>(std::move(*NewAST)) : nullptr;
    }
    // Make sure we put the AST back into the LRU cache.
    auto _ = llvm::make_scope_exit(
        [&AST, this]() { IdleASTs.put(this, std::move(*AST)); });
    // Run the user-provided action.
    if (!*AST)
      return Action(llvm::make_error<llvm::StringError>(
          "invalid AST", llvm::errc::invalid_argument));
    vlog("ASTWorker running {0} on version {2} of {1}", Name, FileName,
         FileInputs.Version);
    Action(InputsAndAST{FileInputs, **AST});
  };
  startTask(Name, std::move(Task), /*UpdateType=*/None, Invalidation);
}

void PreambleThread::build(Request Req) {
  assert(Req.CI && "Got preamble request with null compiler invocation");
  const ParseInputs &Inputs = Req.Inputs;

  Status.update([&](TUStatus &Status) {
    Status.PreambleActivity = PreambleAction::Building;
  });
  auto _ = llvm::make_scope_exit([this, &Req] {
    ASTPeer.updatePreamble(std::move(Req.CI), std::move(Req.Inputs),
                           LatestBuild, std::move(Req.CIDiags),
                           std::move(Req.WantDiags));
  });

  if (!LatestBuild || Inputs.ForceRebuild) {
    vlog("Building first preamble for {0} version {1}", FileName,
         Inputs.Version);
  } else if (isPreambleCompatible(*LatestBuild, Inputs, FileName, *Req.CI)) {
    vlog("Reusing preamble version {0} for version {1} of {2}",
         LatestBuild->Version, Inputs.Version, FileName);
    return;
  } else {
    vlog("Rebuilding invalidated preamble for {0} version {1} (previous was "
         "version {2})",
         FileName, Inputs.Version, LatestBuild->Version);
  }

  LatestBuild = clang::clangd::buildPreamble(
      FileName, *Req.CI, Inputs, StoreInMemory,
      [this, Version(Inputs.Version)](ASTContext &Ctx,
                                      std::shared_ptr<clang::Preprocessor> PP,
                                      const CanonicalIncludes &CanonIncludes) {
        Callbacks.onPreambleAST(FileName, Version, Ctx, std::move(PP),
                                CanonIncludes);
      });
}

void ASTWorker::updatePreamble(std::unique_ptr<CompilerInvocation> CI,
                               ParseInputs PI,
                               std::shared_ptr<const PreambleData> Preamble,
                               std::vector<Diag> CIDiags,
                               WantDiagnostics WantDiags) {
  std::string TaskName = llvm::formatv("Build AST for ({0})", PI.Version);
  // Store preamble and build diagnostics with new preamble if requested.
  auto Task = [this, Preamble = std::move(Preamble), CI = std::move(CI),
               PI = std::move(PI), CIDiags = std::move(CIDiags),
               WantDiags = std::move(WantDiags)]() mutable {
    // Update the preamble inside ASTWorker queue to ensure atomicity. As a task
    // running inside ASTWorker assumes internals won't change until it
    // finishes.
    if (Preamble != LatestPreamble) {
      ++PreambleBuildCount;
      // Cached AST is no longer valid.
      IdleASTs.take(this);
      RanASTCallback = false;
      std::lock_guard<std::mutex> Lock(Mutex);
      // LatestPreamble might be the last reference to old preamble, do not
      // trigger destructor while holding the lock.
      std::swap(LatestPreamble, Preamble);
    }
    // Give up our ownership to old preamble before starting expensive AST
    // build.
    Preamble.reset();
    BuiltFirstPreamble.notify();
    // We only need to build the AST if diagnostics were requested.
    if (WantDiags == WantDiagnostics::No)
      return;
    // Report diagnostics with the new preamble to ensure progress. Otherwise
    // diagnostics might get stale indefinitely if user keeps invalidating the
    // preamble.
    generateDiagnostics(std::move(CI), std::move(PI), std::move(CIDiags));
  };
  if (RunSync) {
    Task();
    ReceivedPreamble.notify();
    return;
  }
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    PreambleRequests.push({std::move(Task), std::move(TaskName),
                           steady_clock::now(), Context::current().clone(),
                           llvm::None, TUScheduler::NoInvalidation, nullptr});
  }
  ReceivedPreamble.notify();
  RequestsCV.notify_all();
}

void ASTWorker::generateDiagnostics(
    std::unique_ptr<CompilerInvocation> Invocation, ParseInputs Inputs,
    std::vector<Diag> CIDiags) {
  // Tracks ast cache accesses for publishing diags.
  static constexpr trace::Metric ASTAccessForDiag(
      "ast_access_diag", trace::Metric::Counter, "result");
  assert(Invocation);
  // No need to rebuild the AST if we won't send the diagnostics.
  {
    std::lock_guard<std::mutex> Lock(PublishMu);
    if (!CanPublishResults)
      return;
  }
  // Used to check whether we can update AST cache.
  bool InputsAreLatest =
      std::tie(FileInputs.CompileCommand, FileInputs.Contents) ==
      std::tie(Inputs.CompileCommand, Inputs.Contents);
  // Take a shortcut and don't report the diagnostics, since they should be the
  // same. All the clients should handle the lack of OnUpdated() call anyway to
  // handle empty result from buildAST.
  // FIXME: the AST could actually change if non-preamble includes changed,
  // but we choose to ignore it.
  if (InputsAreLatest && RanASTCallback)
    return;

  // Get the AST for diagnostics, either build it or use the cached one.
  std::string TaskName = llvm::formatv("Build AST ({0})", Inputs.Version);
  Status.update([&](TUStatus &Status) {
    Status.ASTActivity.K = ASTAction::Building;
    Status.ASTActivity.Name = std::move(TaskName);
  });
  // We might be able to reuse the last we've built for a read request.
  // FIXME: It might be better to not reuse this AST. That way queued AST builds
  // won't be required for diags.
  llvm::Optional<std::unique_ptr<ParsedAST>> AST =
      IdleASTs.take(this, &ASTAccessForDiag);
  if (!AST || !InputsAreLatest) {
    auto RebuildStartTime = DebouncePolicy::clock::now();
    llvm::Optional<ParsedAST> NewAST = ParsedAST::build(
        FileName, Inputs, std::move(Invocation), CIDiags, LatestPreamble);
    auto RebuildDuration = DebouncePolicy::clock::now() - RebuildStartTime;
    ++ASTBuildCount;
    // Try to record the AST-build time, to inform future update debouncing.
    // This is best-effort only: if the lock is held, don't bother.
    std::unique_lock<std::mutex> Lock(Mutex, std::try_to_lock);
    if (Lock.owns_lock()) {
      // Do not let RebuildTimes grow beyond its small-size (i.e.
      // capacity).
      if (RebuildTimes.size() == RebuildTimes.capacity())
        RebuildTimes.erase(RebuildTimes.begin());
      RebuildTimes.push_back(RebuildDuration);
      Lock.unlock();
    }
    Status.update([&](TUStatus &Status) {
      Status.Details.ReuseAST = false;
      Status.Details.BuildFailed = !NewAST.hasValue();
    });
    AST = NewAST ? std::make_unique<ParsedAST>(std::move(*NewAST)) : nullptr;
  } else {
    log("Skipping rebuild of the AST for {0}, inputs are the same.", FileName);
    Status.update([](TUStatus &Status) {
      Status.Details.ReuseAST = true;
      Status.Details.BuildFailed = false;
    });
  }

  // Publish diagnostics.
  auto RunPublish = [&](llvm::function_ref<void()> Publish) {
    // Ensure we only publish results from the worker if the file was not
    // removed, making sure there are not race conditions.
    std::lock_guard<std::mutex> Lock(PublishMu);
    if (CanPublishResults)
      Publish();
  };
  if (*AST) {
    trace::Span Span("Running main AST callback");
    Callbacks.onMainAST(FileName, **AST, RunPublish);
  } else {
    // Failed to build the AST, at least report diagnostics from the
    // command line if there were any.
    // FIXME: we might have got more errors while trying to build the
    // AST, surface them too.
    Callbacks.onFailedAST(FileName, Inputs.Version, CIDiags, RunPublish);
  }

  // AST might've been built for an older version of the source, as ASTWorker
  // queue raced ahead while we were waiting on the preamble. In that case the
  // queue can't reuse the AST.
  if (InputsAreLatest) {
    RanASTCallback = *AST != nullptr;
    IdleASTs.put(this, std::move(*AST));
  }
}

std::shared_ptr<const PreambleData>
ASTWorker::getPossiblyStalePreamble() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return LatestPreamble;
}

void ASTWorker::waitForFirstPreamble() const { BuiltFirstPreamble.wait(); }

tooling::CompileCommand ASTWorker::getCurrentCompileCommand() const {
  std::unique_lock<std::mutex> Lock(Mutex);
  return FileInputs.CompileCommand;
}

TUScheduler::FileStats ASTWorker::stats() const {
  TUScheduler::FileStats Result;
  Result.ASTBuilds = ASTBuildCount;
  Result.PreambleBuilds = PreambleBuildCount;
  // Note that we don't report the size of ASTs currently used for processing
  // the in-flight requests. We used this information for debugging purposes
  // only, so this should be fine.
  Result.UsedBytes = IdleASTs.getUsedBytes(this);
  if (auto Preamble = getPossiblyStalePreamble())
    Result.UsedBytes += Preamble->Preamble.getSize();
  return Result;
}

bool ASTWorker::isASTCached() const { return IdleASTs.getUsedBytes(this) != 0; }

void ASTWorker::stop() {
  {
    std::lock_guard<std::mutex> Lock(PublishMu);
    CanPublishResults = false;
  }
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "stop() called twice");
    Done = true;
  }
  // We are no longer going to build any preambles, let the waiters know that.
  ReceivedPreamble.notify();
  BuiltFirstPreamble.notify();
  PreamblePeer.stop();
  Status.stop();
  RequestsCV.notify_all();
}

void ASTWorker::startTask(llvm::StringRef Name,
                          llvm::unique_function<void()> Task,
                          llvm::Optional<WantDiagnostics> UpdateType,
                          TUScheduler::ASTActionInvalidation Invalidation) {
  if (RunSync) {
    assert(!Done && "running a task after stop()");
    trace::Span Tracer(Name + ":" + llvm::sys::path::filename(FileName));
    Task();
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "running a task after stop()");
    // Cancel any requests invalidated by this request.
    if (UpdateType) {
      for (auto &R : llvm::reverse(Requests)) {
        if (R.InvalidationPolicy == TUScheduler::InvalidateOnUpdate)
          R.Invalidate();
        if (R.UpdateType)
          break; // Older requests were already invalidated by the older update.
      }
    }

    // Allow this request to be cancelled if invalidated.
    Context Ctx = Context::current().derive(kFileBeingProcessed, FileName);
    Canceler Invalidate = nullptr;
    if (Invalidation) {
      WithContext WC(std::move(Ctx));
      std::tie(Ctx, Invalidate) = cancelableTask(
          /*Reason=*/static_cast<int>(ErrorCode::ContentModified));
    }
    Requests.push_back({std::move(Task), std::string(Name), steady_clock::now(),
                        std::move(Ctx), UpdateType, Invalidation,
                        std::move(Invalidate)});
  }
  RequestsCV.notify_all();
}

void ASTWorker::run() {
  while (true) {
    {
      std::unique_lock<std::mutex> Lock(Mutex);
      assert(!CurrentRequest && "A task is already running, multiple workers?");
      for (auto Wait = scheduleLocked(); !Wait.expired();
           Wait = scheduleLocked()) {
        assert(PreambleRequests.empty() &&
               "Preamble updates should be scheduled immediately");
        if (Done) {
          if (Requests.empty())
            return;
          else     // Even though Done is set, finish pending requests.
            break; // However, skip delays to shutdown fast.
        }

        // Tracing: we have a next request, attribute this sleep to it.
        llvm::Optional<WithContext> Ctx;
        llvm::Optional<trace::Span> Tracer;
        if (!Requests.empty()) {
          Ctx.emplace(Requests.front().Ctx.clone());
          Tracer.emplace("Debounce");
          SPAN_ATTACH(*Tracer, "next_request", Requests.front().Name);
          if (!(Wait == Deadline::infinity())) {
            Status.update([&](TUStatus &Status) {
              Status.ASTActivity.K = ASTAction::Queued;
              Status.ASTActivity.Name = Requests.front().Name;
            });
            SPAN_ATTACH(*Tracer, "sleep_ms",
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            Wait.time() - steady_clock::now())
                            .count());
          }
        }

        wait(Lock, RequestsCV, Wait);
      }
      // Any request in ReceivedPreambles is at least as old as the
      // Requests.front(), so prefer them first to preserve LSP order.
      if (!PreambleRequests.empty()) {
        CurrentRequest = std::move(PreambleRequests.front());
        PreambleRequests.pop();
      } else {
        CurrentRequest = std::move(Requests.front());
        Requests.pop_front();
      }
    } // unlock Mutex

    // It is safe to perform reads to CurrentRequest without holding the lock as
    // only writer is also this thread.
    {
      std::unique_lock<Semaphore> Lock(Barrier, std::try_to_lock);
      if (!Lock.owns_lock()) {
        Status.update([&](TUStatus &Status) {
          Status.ASTActivity.K = ASTAction::Queued;
          Status.ASTActivity.Name = CurrentRequest->Name;
        });
        Lock.lock();
      }
      WithContext Guard(std::move(CurrentRequest->Ctx));
      trace::Span Tracer(CurrentRequest->Name);
      Status.update([&](TUStatus &Status) {
        Status.ASTActivity.K = ASTAction::RunningAction;
        Status.ASTActivity.Name = CurrentRequest->Name;
      });
      CurrentRequest->Action();
    }

    bool IsEmpty = false;
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      CurrentRequest.reset();
      IsEmpty = Requests.empty() && PreambleRequests.empty();
    }
    if (IsEmpty) {
      Status.update([&](TUStatus &Status) {
        Status.ASTActivity.K = ASTAction::Idle;
        Status.ASTActivity.Name = "";
      });
    }
    RequestsCV.notify_all();
  }
}

Deadline ASTWorker::scheduleLocked() {
  // Process new preambles immediately.
  if (!PreambleRequests.empty())
    return Deadline::zero();
  if (Requests.empty())
    return Deadline::infinity(); // Wait for new requests.
  // Handle cancelled requests first so the rest of the scheduler doesn't.
  for (auto I = Requests.begin(), E = Requests.end(); I != E; ++I) {
    if (!isCancelled(I->Ctx)) {
      // Cancellations after the first read don't affect current scheduling.
      if (I->UpdateType == None)
        break;
      continue;
    }
    // Cancelled reads are moved to the front of the queue and run immediately.
    if (I->UpdateType == None) {
      Request R = std::move(*I);
      Requests.erase(I);
      Requests.push_front(std::move(R));
      return Deadline::zero();
    }
    // Cancelled updates are downgraded to auto-diagnostics, and may be elided.
    if (I->UpdateType == WantDiagnostics::Yes)
      I->UpdateType = WantDiagnostics::Auto;
  }

  while (shouldSkipHeadLocked()) {
    vlog("ASTWorker skipping {0} for {1}", Requests.front().Name, FileName);
    Requests.pop_front();
  }
  assert(!Requests.empty() && "skipped the whole queue");
  // Some updates aren't dead yet, but never end up being used.
  // e.g. the first keystroke is live until obsoleted by the second.
  // We debounce "maybe-unused" writes, sleeping in case they become dead.
  // But don't delay reads (including updates where diagnostics are needed).
  for (const auto &R : Requests)
    if (R.UpdateType == None || R.UpdateType == WantDiagnostics::Yes)
      return Deadline::zero();
  // Front request needs to be debounced, so determine when we're ready.
  Deadline D(Requests.front().AddTime + UpdateDebounce.compute(RebuildTimes));
  return D;
}

// Returns true if Requests.front() is a dead update that can be skipped.
bool ASTWorker::shouldSkipHeadLocked() const {
  assert(!Requests.empty());
  auto Next = Requests.begin();
  auto UpdateType = Next->UpdateType;
  if (!UpdateType) // Only skip updates.
    return false;
  ++Next;
  // An update is live if its AST might still be read.
  // That is, if it's not immediately followed by another update.
  if (Next == Requests.end() || !Next->UpdateType)
    return false;
  // The other way an update can be live is if its diagnostics might be used.
  switch (*UpdateType) {
  case WantDiagnostics::Yes:
    return false; // Always used.
  case WantDiagnostics::No:
    return true; // Always dead.
  case WantDiagnostics::Auto:
    // Used unless followed by an update that generates diagnostics.
    for (; Next != Requests.end(); ++Next)
      if (Next->UpdateType == WantDiagnostics::Yes ||
          Next->UpdateType == WantDiagnostics::Auto)
        return true; // Prefer later diagnostics.
    return false;
  }
  llvm_unreachable("Unknown WantDiagnostics");
}

bool ASTWorker::blockUntilIdle(Deadline Timeout) const {
  auto WaitUntilASTWorkerIsIdle = [&] {
    std::unique_lock<std::mutex> Lock(Mutex);
    return wait(Lock, RequestsCV, Timeout, [&] {
      return PreambleRequests.empty() && Requests.empty() && !CurrentRequest;
    });
  };
  // Make sure ASTWorker has processed all requests, which might issue new
  // updates to PreamblePeer.
  WaitUntilASTWorkerIsIdle();
  // Now that ASTWorker processed all requests, ensure PreamblePeer has served
  // all update requests. This might create new PreambleRequests for the
  // ASTWorker.
  PreamblePeer.blockUntilIdle(Timeout);
  assert(Requests.empty() &&
         "No new normal tasks can be scheduled concurrently with "
         "blockUntilIdle(): ASTWorker isn't threadsafe");
  // Finally make sure ASTWorker has processed all of the preamble updates.
  return WaitUntilASTWorkerIsIdle();
}

// Render a TUAction to a user-facing string representation.
// TUAction represents clangd-internal states, we don't intend to expose them
// to users (say C++ programmers) directly to avoid confusion, we use terms that
// are familiar by C++ programmers.
std::string renderTUAction(const PreambleAction PA, const ASTAction &AA) {
  llvm::SmallVector<std::string, 2> Result;
  switch (PA) {
  case PreambleAction::Building:
    Result.push_back("parsing includes");
    break;
  case PreambleAction::Idle:
    // We handle idle specially below.
    break;
  }
  switch (AA.K) {
  case ASTAction::Queued:
    Result.push_back("file is queued");
    break;
  case ASTAction::RunningAction:
    Result.push_back("running " + AA.Name);
    break;
  case ASTAction::Building:
    Result.push_back("parsing main file");
    break;
  case ASTAction::Idle:
    // We handle idle specially below.
    break;
  }
  if (Result.empty())
    return "idle";
  return llvm::join(Result, ",");
}

} // namespace

unsigned getDefaultAsyncThreadsCount() {
  return llvm::heavyweight_hardware_concurrency().compute_thread_count();
}

FileStatus TUStatus::render(PathRef File) const {
  FileStatus FStatus;
  FStatus.uri = URIForFile::canonicalize(File, /*TUPath=*/File);
  FStatus.state = renderTUAction(PreambleActivity, ASTActivity);
  return FStatus;
}

struct TUScheduler::FileData {
  /// Latest inputs, passed to TUScheduler::update().
  std::string Contents;
  ASTWorkerHandle Worker;
};

TUScheduler::TUScheduler(const GlobalCompilationDatabase &CDB,
                         const Options &Opts,
                         std::unique_ptr<ParsingCallbacks> Callbacks)
    : CDB(CDB), Opts(Opts),
      Callbacks(Callbacks ? move(Callbacks)
                          : std::make_unique<ParsingCallbacks>()),
      Barrier(Opts.AsyncThreadsCount),
      IdleASTs(
          std::make_unique<ASTCache>(Opts.RetentionPolicy.MaxRetainedASTs)) {
  if (0 < Opts.AsyncThreadsCount) {
    PreambleTasks.emplace();
    WorkerThreads.emplace();
  }
}

TUScheduler::~TUScheduler() {
  // Notify all workers that they need to stop.
  Files.clear();

  // Wait for all in-flight tasks to finish.
  if (PreambleTasks)
    PreambleTasks->wait();
  if (WorkerThreads)
    WorkerThreads->wait();
}

bool TUScheduler::blockUntilIdle(Deadline D) const {
  for (auto &File : Files)
    if (!File.getValue()->Worker->blockUntilIdle(D))
      return false;
  if (PreambleTasks)
    if (!PreambleTasks->wait(D))
      return false;
  return true;
}

bool TUScheduler::update(PathRef File, ParseInputs Inputs,
                         WantDiagnostics WantDiags) {
  std::unique_ptr<FileData> &FD = Files[File];
  bool NewFile = FD == nullptr;
  if (!FD) {
    // Create a new worker to process the AST-related tasks.
    ASTWorkerHandle Worker =
        ASTWorker::create(File, CDB, *IdleASTs,
                          WorkerThreads ? WorkerThreads.getPointer() : nullptr,
                          Barrier, Opts, *Callbacks);
    FD = std::unique_ptr<FileData>(
        new FileData{Inputs.Contents, std::move(Worker)});
  } else {
    FD->Contents = Inputs.Contents;
  }
  FD->Worker->update(std::move(Inputs), WantDiags);
  return NewFile;
}

void TUScheduler::remove(PathRef File) {
  bool Removed = Files.erase(File);
  if (!Removed)
    elog("Trying to remove file from TUScheduler that is not tracked: {0}",
         File);
}

llvm::StringMap<std::string> TUScheduler::getAllFileContents() const {
  llvm::StringMap<std::string> Results;
  for (auto &It : Files)
    Results.try_emplace(It.getKey(), It.getValue()->Contents);
  return Results;
}

void TUScheduler::run(llvm::StringRef Name,
                      llvm::unique_function<void()> Action) {
  if (!PreambleTasks)
    return Action();
  PreambleTasks->runAsync(Name, [this, Ctx = Context::current().clone(),
                                 Action = std::move(Action)]() mutable {
    std::lock_guard<Semaphore> BarrierLock(Barrier);
    WithContext WC(std::move(Ctx));
    Action();
  });
}

void TUScheduler::runWithAST(
    llvm::StringRef Name, PathRef File,
    llvm::unique_function<void(llvm::Expected<InputsAndAST>)> Action,
    TUScheduler::ASTActionInvalidation Invalidation) {
  auto It = Files.find(File);
  if (It == Files.end()) {
    Action(llvm::make_error<LSPError>(
        "trying to get AST for non-added document", ErrorCode::InvalidParams));
    return;
  }

  It->second->Worker->runWithAST(Name, std::move(Action), Invalidation);
}

void TUScheduler::runWithPreamble(llvm::StringRef Name, PathRef File,
                                  PreambleConsistency Consistency,
                                  Callback<InputsAndPreamble> Action) {
  auto It = Files.find(File);
  if (It == Files.end()) {
    Action(llvm::make_error<LSPError>(
        "trying to get preamble for non-added document",
        ErrorCode::InvalidParams));
    return;
  }

  if (!PreambleTasks) {
    trace::Span Tracer(Name);
    SPAN_ATTACH(Tracer, "file", File);
    std::shared_ptr<const PreambleData> Preamble =
        It->second->Worker->getPossiblyStalePreamble();
    Action(InputsAndPreamble{It->second->Contents,
                             It->second->Worker->getCurrentCompileCommand(),
                             Preamble.get()});
    return;
  }

  std::shared_ptr<const ASTWorker> Worker = It->second->Worker.lock();
  auto Task =
      [Worker, Consistency, Name = Name.str(), File = File.str(),
       Contents = It->second->Contents,
       Command = Worker->getCurrentCompileCommand(),
       Ctx = Context::current().derive(kFileBeingProcessed, std::string(File)),
       Action = std::move(Action), this]() mutable {
        std::shared_ptr<const PreambleData> Preamble;
        if (Consistency == PreambleConsistency::Stale) {
          // Wait until the preamble is built for the first time, if preamble
          // is required. This avoids extra work of processing the preamble
          // headers in parallel multiple times.
          Worker->waitForFirstPreamble();
        }
        Preamble = Worker->getPossiblyStalePreamble();

        std::lock_guard<Semaphore> BarrierLock(Barrier);
        WithContext Guard(std::move(Ctx));
        trace::Span Tracer(Name);
        SPAN_ATTACH(Tracer, "file", File);
        Action(InputsAndPreamble{Contents, Command, Preamble.get()});
      };

  PreambleTasks->runAsync("task:" + llvm::sys::path::filename(File),
                          std::move(Task));
}

llvm::StringMap<TUScheduler::FileStats> TUScheduler::fileStats() const {
  llvm::StringMap<TUScheduler::FileStats> Result;
  for (const auto &PathAndFile : Files)
    Result.try_emplace(PathAndFile.first(),
                       PathAndFile.second->Worker->stats());
  return Result;
}

std::vector<Path> TUScheduler::getFilesWithCachedAST() const {
  std::vector<Path> Result;
  for (auto &&PathAndFile : Files) {
    if (!PathAndFile.second->Worker->isASTCached())
      continue;
    Result.push_back(std::string(PathAndFile.first()));
  }
  return Result;
}

DebouncePolicy::clock::duration
DebouncePolicy::compute(llvm::ArrayRef<clock::duration> History) const {
  assert(Min <= Max && "Invalid policy");
  if (History.empty())
    return Max; // Arbitrary.

  // Base the result on the median rebuild.
  // nth_element needs a mutable array, take the chance to bound the data size.
  History = History.take_back(15);
  llvm::SmallVector<clock::duration, 15> Recent(History.begin(), History.end());
  auto Median = Recent.begin() + Recent.size() / 2;
  std::nth_element(Recent.begin(), Median, Recent.end());

  clock::duration Target =
      std::chrono::duration_cast<clock::duration>(RebuildRatio * *Median);
  if (Target > Max)
    return Max;
  if (Target < Min)
    return Min;
  return Target;
}

DebouncePolicy DebouncePolicy::fixed(clock::duration T) {
  DebouncePolicy P;
  P.Min = P.Max = T;
  return P;
}

} // namespace clangd
} // namespace clang
