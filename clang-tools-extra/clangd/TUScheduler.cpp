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
#include "CompileCommands.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "GlobalCompilationDatabase.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "index/CanonicalIncludes.h"
#include "support/Cancellation.h"
#include "support/Context.h"
#include "support/Logger.h"
#include "support/MemoryTree.h"
#include "support/Path.h"
#include "support/ThreadCrashReporter.h"
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
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
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
// Tracks latency (in seconds) of FS operations done during a preamble build.
// build_type allows to split by expected VFS cache state (cold on first
// preamble, somewhat warm after that when building first preamble for new file,
// likely ~everything cached on preamble rebuild.
constexpr trace::Metric
    PreambleBuildFilesystemLatency("preamble_fs_latency",
                                   trace::Metric::Distribution, "build_type");
// Tracks latency of FS operations done during a preamble build as a ratio of
// preamble build time. build_type is same as above.
constexpr trace::Metric PreambleBuildFilesystemLatencyRatio(
    "preamble_fs_latency_ratio", trace::Metric::Distribution, "build_type");

constexpr trace::Metric PreambleBuildSize("preamble_build_size",
                                          trace::Metric::Distribution);
constexpr trace::Metric PreambleSerializedSize("preamble_serialized_size",
                                               trace::Metric::Distribution);

void reportPreambleBuild(const PreambleBuildStats &Stats,
                         bool IsFirstPreamble) {
  auto RecordWithLabel = [&Stats](llvm::StringRef Label) {
    PreambleBuildFilesystemLatency.record(Stats.FileSystemTime, Label);
    if (Stats.TotalBuildTime > 0) // Avoid division by zero.
      PreambleBuildFilesystemLatencyRatio.record(
          Stats.FileSystemTime / Stats.TotalBuildTime, Label);
  };

  static llvm::once_flag OnceFlag;
  llvm::call_once(OnceFlag, [&] { RecordWithLabel("first_build"); });
  RecordWithLabel(IsFirstPreamble ? "first_build_for_file" : "rebuild");

  PreambleBuildSize.record(Stats.BuildSize);
  PreambleSerializedSize.record(Stats.SerializedSize);
}

class ASTWorker;
} // namespace

static clang::clangd::Key<std::string> FileBeingProcessed;

llvm::Optional<llvm::StringRef> TUScheduler::getFileBeingProcessedInContext() {
  if (auto *File = Context::current().get(FileBeingProcessed))
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

/// A map from header files to an opened "proxy" file that includes them.
/// If you open the header, the compile command from the proxy file is used.
///
/// This inclusion information could also naturally live in the index, but there
/// are advantages to using open files instead:
///  - it's easier to achieve a *stable* choice of proxy, which is important
///    to avoid invalidating the preamble
///  - context-sensitive flags for libraries with multiple configurations
///    (e.g. C++ stdlib sensitivity to -std version)
///  - predictable behavior, e.g. guarantees that go-to-def landing on a header
///    will have a suitable command available
///  - fewer scaling problems to solve (project include graphs are big!)
///
/// Implementation details:
/// - We only record this for mainfiles where the command was trustworthy
///   (i.e. not inferred). This avoids a bad inference "infecting" other files.
/// - Once we've picked a proxy file for a header, we stick with it until the
///   proxy file is invalidated *and* a new candidate proxy file is built.
///   Switching proxies is expensive, as the compile flags will (probably)
///   change and therefore we'll end up rebuilding the header's preamble.
/// - We don't capture the actual compile command, but just the filename we
///   should query to get it. This avoids getting out of sync with the CDB.
///
/// All methods are threadsafe. In practice, update() comes from preamble
/// threads, remove()s mostly from the main thread, and get() from ASTWorker.
/// Writes are rare and reads are cheap, so we don't expect much contention.
class TUScheduler::HeaderIncluderCache {
  // We should be a little careful how we store the include graph of open
  // files, as each can have a large number of transitive headers.
  // This representation is O(unique transitive source files).
  llvm::BumpPtrAllocator Arena;
  struct Association {
    llvm::StringRef MainFile;
    // Circular-linked-list of associations with the same mainFile.
    // Null indicates that the mainfile was removed.
    Association *Next;
  };
  llvm::StringMap<Association, llvm::BumpPtrAllocator &> HeaderToMain;
  llvm::StringMap<Association *, llvm::BumpPtrAllocator &> MainToFirst;
  std::atomic<size_t> UsedBytes; // Updated after writes.
  mutable std::mutex Mu;

  void invalidate(Association *First) {
    Association *Current = First;
    do {
      Association *Next = Current->Next;
      Current->Next = nullptr;
      Current = Next;
    } while (Current != First);
  }

  // Create the circular list and return the head of it.
  Association *associate(llvm::StringRef MainFile,
                         llvm::ArrayRef<std::string> Headers) {
    Association *First = nullptr, *Prev = nullptr;
    for (const std::string &Header : Headers) {
      auto &Assoc = HeaderToMain[Header];
      if (Assoc.Next)
        continue; // Already has a valid association.

      Assoc.MainFile = MainFile;
      Assoc.Next = Prev;
      Prev = &Assoc;
      if (!First)
        First = &Assoc;
    }
    if (First)
      First->Next = Prev;
    return First;
  }

  void updateMemoryUsage() {
    auto StringMapHeap = [](const auto &Map) {
      // StringMap stores the hashtable on the heap.
      // It contains pointers to the entries, and a hashcode for each.
      return Map.getNumBuckets() * (sizeof(void *) + sizeof(unsigned));
    };
    size_t Usage = Arena.getTotalMemory() + StringMapHeap(MainToFirst) +
                   StringMapHeap(HeaderToMain) + sizeof(*this);
    UsedBytes.store(Usage, std::memory_order_release);
  }

public:
  HeaderIncluderCache() : HeaderToMain(Arena), MainToFirst(Arena) {
    updateMemoryUsage();
  }

  // Associate each header with MainFile (unless already associated).
  // Headers not in the list will have their associations removed.
  void update(PathRef MainFile, llvm::ArrayRef<std::string> Headers) {
    std::lock_guard<std::mutex> Lock(Mu);
    auto It = MainToFirst.try_emplace(MainFile, nullptr);
    Association *&First = It.first->second;
    if (First)
      invalidate(First);
    First = associate(It.first->first(), Headers);
    updateMemoryUsage();
  }

  // Mark MainFile as gone.
  // This will *not* disassociate headers with MainFile immediately, but they
  // will be eligible for association with other files that get update()d.
  void remove(PathRef MainFile) {
    std::lock_guard<std::mutex> Lock(Mu);
    Association *&First = MainToFirst[MainFile];
    if (First) {
      invalidate(First);
      First = nullptr;
    }
    // MainToFirst entry should stay alive, as Associations might be pointing at
    // its key.
  }

  /// Get the mainfile associated with Header, or the empty string if none.
  std::string get(PathRef Header) const {
    std::lock_guard<std::mutex> Lock(Mu);
    return HeaderToMain.lookup(Header).MainFile.str();
  }

  size_t getUsedBytes() const {
    return UsedBytes.load(std::memory_order_acquire);
  }
};

namespace {

bool isReliable(const tooling::CompileCommand &Cmd) {
  return Cmd.Heuristic.empty();
}

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

// An attempt to acquire resources for a task using PreambleThrottler.
// Initially it is unsatisfied, it (hopefully) becomes satisfied later but may
// be destroyed before then. Destruction releases all resources.
class PreambleThrottlerRequest {
public:
  // The condition variable is signalled when the request is satisfied.
  PreambleThrottlerRequest(llvm::StringRef Filename,
                           PreambleThrottler *Throttler,
                           std::condition_variable &CV)
      : Throttler(Throttler),
        Satisfied(Throttler == nullptr) {
    // If there is no throttler, this dummy request is always satisfied.
    if (!Throttler)
      return;
    ID = Throttler->acquire(Filename, [&] {
      Satisfied.store(true, std::memory_order_release);
      CV.notify_all();
    });
  }

  bool satisfied() const { return Satisfied.load(std::memory_order_acquire); }

  // When the request is destroyed:
  //  - if resources are not yet obtained, stop trying to get them.
  //  - if resources were obtained, release them.
  ~PreambleThrottlerRequest() {
    if (Throttler)
      Throttler->release(ID);
  }

private:
  PreambleThrottler::RequestID ID;
  PreambleThrottler *Throttler;
  std::atomic<bool> Satisfied = {false};
};

/// Responsible for building preambles. Whenever the thread is idle and the
/// preamble is outdated, it starts to build a fresh preamble from the latest
/// inputs. If RunSync is true, preambles are built synchronously in update()
/// instead.
class PreambleThread {
public:
  PreambleThread(llvm::StringRef FileName, ParsingCallbacks &Callbacks,
                 bool StorePreambleInMemory, bool RunSync,
                 PreambleThrottler *Throttler, SynchronizedTUStatus &Status,
                 TUScheduler::HeaderIncluderCache &HeaderIncluders,
                 ASTWorker &AW)
      : FileName(FileName), Callbacks(Callbacks),
        StoreInMemory(StorePreambleInMemory), RunSync(RunSync),
        Throttler(Throttler), Status(Status), ASTPeer(AW),
        HeaderIncluders(HeaderIncluders) {}

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
      llvm::Optional<PreambleThrottlerRequest> Throttle;
      {
        std::unique_lock<std::mutex> Lock(Mutex);
        assert(!CurrentReq && "Already processing a request?");
        // Wait until stop is called or there is a request.
        ReqCV.wait(Lock, [&] { return NextReq || Done; });
        if (Done)
          break;

        {
          Throttle.emplace(FileName, Throttler, ReqCV);
          llvm::Optional<trace::Span> Tracer;
          // If acquire succeeded synchronously, avoid status jitter.
          if (!Throttle->satisfied()) {
            Tracer.emplace("PreambleThrottle");
            Status.update([&](TUStatus &Status) {
              Status.PreambleActivity = PreambleAction::Queued;
            });
          }
          ReqCV.wait(Lock, [&] { return Throttle->satisfied() || Done; });
        }
        if (Done)
          break;
        // While waiting for the throttler, the request may have been updated!
        // That's fine though, there's still guaranteed to be some request.

        CurrentReq = std::move(*NextReq);
        NextReq.reset();
      }

      {
        WithContext Guard(std::move(CurrentReq->Ctx));
        // Note that we don't make use of the ContextProvider here.
        // Preamble tasks are always scheduled by ASTWorker tasks, and we
        // reuse the context/config that was created at that level.

        // Build the preamble and let the waiters know about it.
        build(std::move(*CurrentReq));
      }
      // Releasing the throttle before destroying the request assists testing.
      Throttle.reset();
      bool IsEmpty = false;
      {
        std::lock_guard<std::mutex> Lock(Mutex);
        CurrentReq.reset();
        IsEmpty = !NextReq;
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
  PreambleThrottler *Throttler;

  SynchronizedTUStatus &Status;
  ASTWorker &ASTPeer;
  TUScheduler::HeaderIncluderCache &HeaderIncluders;
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
            TUScheduler::ASTCache &LRUCache,
            TUScheduler::HeaderIncluderCache &HeaderIncluders,
            Semaphore &Barrier, bool RunSync, const TUScheduler::Options &Opts,
            ParsingCallbacks &Callbacks);

public:
  /// Create a new ASTWorker and return a handle to it.
  /// The processing thread is spawned using \p Tasks. However, when \p Tasks
  /// is null, all requests will be processed on the calling thread
  /// synchronously instead. \p Barrier is acquired when processing each
  /// request, it is used to limit the number of actively running threads.
  static ASTWorkerHandle
  create(PathRef FileName, const GlobalCompilationDatabase &CDB,
         TUScheduler::ASTCache &IdleASTs,
         TUScheduler::HeaderIncluderCache &HeaderIncluders,
         AsyncTaskRunner *Tasks, Semaphore &Barrier,
         const TUScheduler::Options &Opts, ParsingCallbacks &Callbacks);
  ~ASTWorker();

  void update(ParseInputs Inputs, WantDiagnostics, bool ContentChanged);
  void
  runWithAST(llvm::StringRef Name,
             llvm::unique_function<void(llvm::Expected<InputsAndAST>)> Action,
             TUScheduler::ASTActionInvalidation);
  bool blockUntilIdle(Deadline Timeout) const;

  std::shared_ptr<const PreambleData> getPossiblyStalePreamble(
      std::shared_ptr<const ASTSignals> *ASTSignals = nullptr) const;

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
  // Details of an update request that are relevant to scheduling.
  struct UpdateType {
    // Do we want diagnostics from this version?
    // If Yes, we must always build this version.
    // If No, we only need to build this version if it's read.
    // If Auto, we build if it's read or if the debounce expires.
    WantDiagnostics Diagnostics;
    // Did the main-file content of the document change?
    // If so, we're allowed to cancel certain invalidated preceding reads.
    bool ContentChanged;
  };

  /// Publishes diagnostics for \p Inputs. It will build an AST or reuse the
  /// cached one if applicable. Assumes LatestPreamble is compatible for \p
  /// Inputs.
  void generateDiagnostics(std::unique_ptr<CompilerInvocation> Invocation,
                           ParseInputs Inputs, std::vector<Diag> CIDiags);

  void updateASTSignals(ParsedAST &AST);

  // Must be called exactly once on processing thread. Will return after
  // stop() is called on a separate thread and all pending requests are
  // processed.
  void run();
  /// Signal that run() should finish processing pending requests and exit.
  void stop();

  /// Adds a new task to the end of the request queue.
  void startTask(llvm::StringRef Name, llvm::unique_function<void()> Task,
                 llvm::Optional<UpdateType> Update,
                 TUScheduler::ASTActionInvalidation);
  /// Runs a task synchronously.
  void runTask(llvm::StringRef Name, llvm::function_ref<void()> Task);

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
    llvm::Optional<Context> QueueCtx;
    llvm::Optional<UpdateType> Update;
    TUScheduler::ASTActionInvalidation InvalidationPolicy;
    Canceler Invalidate;
  };

  /// Handles retention of ASTs.
  TUScheduler::ASTCache &IdleASTs;
  TUScheduler::HeaderIncluderCache &HeaderIncluders;
  const bool RunSync;
  /// Time to wait after an update to see whether another update obsoletes it.
  const DebouncePolicy UpdateDebounce;
  /// File that ASTWorker is responsible for.
  const Path FileName;
  /// Callback to create processing contexts for tasks.
  const std::function<Context(llvm::StringRef)> ContextProvider;
  const GlobalCompilationDatabase &CDB;
  /// Callback invoked when preamble or main file AST is built.
  ParsingCallbacks &Callbacks;

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
  llvm::SmallVector<DebouncePolicy::clock::duration>
      RebuildTimes; /* GUARDED_BY(Mutex) */
  /// Set to true to signal run() to finish processing.
  bool Done;                              /* GUARDED_BY(Mutex) */
  std::deque<Request> Requests;           /* GUARDED_BY(Mutex) */
  llvm::Optional<Request> CurrentRequest; /* GUARDED_BY(Mutex) */
  /// Signalled whenever a new request has been scheduled or processing of a
  /// request has completed.
  mutable std::condition_variable RequestsCV;
  std::shared_ptr<const ASTSignals> LatestASTSignals; /* GUARDED_BY(Mutex) */
  /// Latest build preamble for current TU.
  /// None means no builds yet, null means there was an error while building.
  /// Only written by ASTWorker's thread.
  llvm::Optional<std::shared_ptr<const PreambleData>> LatestPreamble;
  std::deque<Request> PreambleRequests; /* GUARDED_BY(Mutex) */
  /// Signaled whenever LatestPreamble changes state or there's a new
  /// PreambleRequest.
  mutable std::condition_variable PreambleCV;
  /// Guards the callback that publishes results of AST-related computations
  /// (diagnostics) and file statuses.
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

ASTWorkerHandle
ASTWorker::create(PathRef FileName, const GlobalCompilationDatabase &CDB,
                  TUScheduler::ASTCache &IdleASTs,
                  TUScheduler::HeaderIncluderCache &HeaderIncluders,
                  AsyncTaskRunner *Tasks, Semaphore &Barrier,
                  const TUScheduler::Options &Opts,
                  ParsingCallbacks &Callbacks) {
  std::shared_ptr<ASTWorker> Worker(
      new ASTWorker(FileName, CDB, IdleASTs, HeaderIncluders, Barrier,
                    /*RunSync=*/!Tasks, Opts, Callbacks));
  if (Tasks) {
    Tasks->runAsync("ASTWorker:" + llvm::sys::path::filename(FileName),
                    [Worker]() { Worker->run(); });
    Tasks->runAsync("PreambleWorker:" + llvm::sys::path::filename(FileName),
                    [Worker]() { Worker->PreamblePeer.run(); });
  }

  return ASTWorkerHandle(std::move(Worker));
}

ASTWorker::ASTWorker(PathRef FileName, const GlobalCompilationDatabase &CDB,
                     TUScheduler::ASTCache &LRUCache,
                     TUScheduler::HeaderIncluderCache &HeaderIncluders,
                     Semaphore &Barrier, bool RunSync,
                     const TUScheduler::Options &Opts,
                     ParsingCallbacks &Callbacks)
    : IdleASTs(LRUCache), HeaderIncluders(HeaderIncluders), RunSync(RunSync),
      UpdateDebounce(Opts.UpdateDebounce), FileName(FileName),
      ContextProvider(Opts.ContextProvider), CDB(CDB), Callbacks(Callbacks),
      Barrier(Barrier), Done(false), Status(FileName, Callbacks),
      PreamblePeer(FileName, Callbacks, Opts.StorePreamblesInMemory, RunSync,
                   Opts.PreambleThrottler, Status, HeaderIncluders, *this) {
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

void ASTWorker::update(ParseInputs Inputs, WantDiagnostics WantDiags,
                       bool ContentChanged) {
  llvm::StringLiteral TaskName = "Update";
  auto Task = [=]() mutable {
    // Get the actual command as `Inputs` does not have a command.
    // FIXME: some build systems like Bazel will take time to preparing
    // environment to build the file, it would be nice if we could emit a
    // "PreparingBuild" status to inform users, it is non-trivial given the
    // current implementation.
    auto Cmd = CDB.getCompileCommand(FileName);
    // If we don't have a reliable command for this file, it may be a header.
    // Try to find a file that includes it, to borrow its command.
    if (!Cmd || !isReliable(*Cmd)) {
      std::string ProxyFile = HeaderIncluders.get(FileName);
      if (!ProxyFile.empty()) {
        auto ProxyCmd = CDB.getCompileCommand(ProxyFile);
        if (!ProxyCmd || !isReliable(*ProxyCmd)) {
          // This command is supposed to be reliable! It's probably gone.
          HeaderIncluders.remove(ProxyFile);
        } else {
          // We have a reliable command for an including file, use it.
          Cmd = tooling::transferCompileCommand(std::move(*ProxyCmd), FileName);
        }
      }
    }
    if (Cmd)
      Inputs.CompileCommand = std::move(*Cmd);
    else
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
        printArgv(Inputs.CompileCommand.CommandLine));

    StoreDiags CompilerInvocationDiagConsumer;
    std::vector<std::string> CC1Args;
    std::unique_ptr<CompilerInvocation> Invocation = buildCompilerInvocation(
        Inputs, CompilerInvocationDiagConsumer, &CC1Args);
    // Log cc1 args even (especially!) if creating invocation failed.
    if (!CC1Args.empty())
      vlog("Driver produced command: cc1 {0}", printArgv(CC1Args));
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
      // Note that this might throw away a stale preamble that might still be
      // useful, but this is how we communicate a build error.
      LatestPreamble.emplace();
      // Make sure anyone waiting for the preamble gets notified it could not be
      // built.
      PreambleCV.notify_all();
      return;
    }

    PreamblePeer.update(std::move(Invocation), std::move(Inputs),
                        std::move(CompilerInvocationDiags), WantDiags);
    std::unique_lock<std::mutex> Lock(Mutex);
    PreambleCV.wait(Lock, [this] {
      // Block until we reiceve a preamble request, unless a preamble already
      // exists, as patching an empty preamble would imply rebuilding it from
      // scratch.
      // We block here instead of the consumer to prevent any deadlocks. Since
      // LatestPreamble is only populated by ASTWorker thread.
      return LatestPreamble || !PreambleRequests.empty() || Done;
    });
  };
  startTask(TaskName, std::move(Task), UpdateType{WantDiags, ContentChanged},
            TUScheduler::NoInvalidation);
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
      return Action(error(llvm::errc::invalid_argument, "invalid AST"));
    vlog("ASTWorker running {0} on version {2} of {1}", Name, FileName,
         FileInputs.Version);
    Action(InputsAndAST{FileInputs, **AST});
  };
  startTask(Name, std::move(Task), /*Update=*/None, Invalidation);
}

/// To be called from ThreadCrashReporter's signal handler.
static void crashDumpCompileCommand(llvm::raw_ostream &OS,
                                    const tooling::CompileCommand &Command) {
  OS << "  Filename: " << Command.Filename << "\n";
  OS << "  Directory: " << Command.Directory << "\n";
  OS << "  Command Line:";
  for (auto &Arg : Command.CommandLine) {
    OS << " " << Arg;
  }
  OS << "\n";
}

/// To be called from ThreadCrashReporter's signal handler.
static void crashDumpFileContents(llvm::raw_ostream &OS,
                                  const std::string &Contents) {
  // Avoid flooding the terminal with source code by default, but allow clients
  // to opt in. Use an env var to preserve backwards compatibility of the
  // command line interface, while allowing it to be set outside the clangd
  // launch site for more flexibility.
  if (getenv("CLANGD_CRASH_DUMP_SOURCE")) {
    OS << "  Contents:\n";
    OS << Contents << "\n";
  }
}

/// To be called from ThreadCrashReporter's signal handler.
static void crashDumpParseInputs(llvm::raw_ostream &OS,
                                 const ParseInputs &FileInputs) {
  auto &Command = FileInputs.CompileCommand;
  crashDumpCompileCommand(OS, Command);
  OS << "  Version: " << FileInputs.Version << "\n";
  crashDumpFileContents(OS, FileInputs.Contents);
}

void PreambleThread::build(Request Req) {
  assert(Req.CI && "Got preamble request with null compiler invocation");
  const ParseInputs &Inputs = Req.Inputs;
  bool ReusedPreamble = false;

  Status.update([&](TUStatus &Status) {
    Status.PreambleActivity = PreambleAction::Building;
  });
  auto _ = llvm::make_scope_exit([this, &Req, &ReusedPreamble] {
    ASTPeer.updatePreamble(std::move(Req.CI), std::move(Req.Inputs),
                           LatestBuild, std::move(Req.CIDiags),
                           std::move(Req.WantDiags));
    if (!ReusedPreamble)
      Callbacks.onPreamblePublished(FileName);
  });

  if (!LatestBuild || Inputs.ForceRebuild) {
    vlog("Building first preamble for {0} version {1}", FileName,
         Inputs.Version);
  } else if (isPreambleCompatible(*LatestBuild, Inputs, FileName, *Req.CI)) {
    vlog("Reusing preamble version {0} for version {1} of {2}",
         LatestBuild->Version, Inputs.Version, FileName);
    ReusedPreamble = true;
    return;
  } else {
    vlog("Rebuilding invalidated preamble for {0} version {1} (previous was "
         "version {2})",
         FileName, Inputs.Version, LatestBuild->Version);
  }

  ThreadCrashReporter ScopedReporter([&Inputs]() {
    llvm::errs() << "Signalled while building preamble\n";
    crashDumpParseInputs(llvm::errs(), Inputs);
  });

  PreambleBuildStats Stats;
  bool IsFirstPreamble = !LatestBuild;
  LatestBuild = clang::clangd::buildPreamble(
      FileName, *Req.CI, Inputs, StoreInMemory,
      [&](ASTContext &Ctx, Preprocessor &PP,
          const CanonicalIncludes &CanonIncludes) {
        Callbacks.onPreambleAST(FileName, Inputs.Version, *Req.CI, Ctx, PP,
                                CanonIncludes);
      },
      &Stats);
  if (!LatestBuild)
    return;
  reportPreambleBuild(Stats, IsFirstPreamble);
  if (isReliable(LatestBuild->CompileCommand))
    HeaderIncluders.update(FileName, LatestBuild->Includes.allHeaders());
}

void ASTWorker::updatePreamble(std::unique_ptr<CompilerInvocation> CI,
                               ParseInputs PI,
                               std::shared_ptr<const PreambleData> Preamble,
                               std::vector<Diag> CIDiags,
                               WantDiagnostics WantDiags) {
  llvm::StringLiteral TaskName = "Build AST";
  // Store preamble and build diagnostics with new preamble if requested.
  auto Task = [this, Preamble = std::move(Preamble), CI = std::move(CI),
               PI = std::move(PI), CIDiags = std::move(CIDiags),
               WantDiags = std::move(WantDiags)]() mutable {
    // Update the preamble inside ASTWorker queue to ensure atomicity. As a task
    // running inside ASTWorker assumes internals won't change until it
    // finishes.
    if (!LatestPreamble || Preamble != *LatestPreamble) {
      ++PreambleBuildCount;
      // Cached AST is no longer valid.
      IdleASTs.take(this);
      RanASTCallback = false;
      std::lock_guard<std::mutex> Lock(Mutex);
      // LatestPreamble might be the last reference to old preamble, do not
      // trigger destructor while holding the lock.
      if (LatestPreamble)
        std::swap(*LatestPreamble, Preamble);
      else
        LatestPreamble = std::move(Preamble);
    }
    // Notify anyone waiting for a preamble.
    PreambleCV.notify_all();
    // Give up our ownership to old preamble before starting expensive AST
    // build.
    Preamble.reset();
    // We only need to build the AST if diagnostics were requested.
    if (WantDiags == WantDiagnostics::No)
      return;
    // Report diagnostics with the new preamble to ensure progress. Otherwise
    // diagnostics might get stale indefinitely if user keeps invalidating the
    // preamble.
    generateDiagnostics(std::move(CI), std::move(PI), std::move(CIDiags));
  };
  if (RunSync) {
    runTask(TaskName, Task);
    return;
  }
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    PreambleRequests.push_back({std::move(Task), std::string(TaskName),
                                steady_clock::now(), Context::current().clone(),
                                llvm::None, llvm::None,
                                TUScheduler::NoInvalidation, nullptr});
  }
  PreambleCV.notify_all();
  RequestsCV.notify_all();
}

void ASTWorker::updateASTSignals(ParsedAST &AST) {
  auto Signals = std::make_shared<const ASTSignals>(ASTSignals::derive(AST));
  // Existing readers of ASTSignals will have their copy preserved until the
  // read is completed. The last reader deletes the old ASTSignals.
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    std::swap(LatestASTSignals, Signals);
  }
}

void ASTWorker::generateDiagnostics(
    std::unique_ptr<CompilerInvocation> Invocation, ParseInputs Inputs,
    std::vector<Diag> CIDiags) {
  // Tracks ast cache accesses for publishing diags.
  static constexpr trace::Metric ASTAccessForDiag(
      "ast_access_diag", trace::Metric::Counter, "result");
  assert(Invocation);
  assert(LatestPreamble);
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
        FileName, Inputs, std::move(Invocation), CIDiags, *LatestPreamble);
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
      Status.Details.BuildFailed = !NewAST;
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
    updateASTSignals(**AST);
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

std::shared_ptr<const PreambleData> ASTWorker::getPossiblyStalePreamble(
    std::shared_ptr<const ASTSignals> *ASTSignals) const {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (ASTSignals)
    *ASTSignals = LatestASTSignals;
  return LatestPreamble ? *LatestPreamble : nullptr;
}

void ASTWorker::waitForFirstPreamble() const {
  std::unique_lock<std::mutex> Lock(Mutex);
  PreambleCV.wait(Lock, [this] { return LatestPreamble || Done; });
}

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
  Result.UsedBytesAST = IdleASTs.getUsedBytes(this);
  if (auto Preamble = getPossiblyStalePreamble())
    Result.UsedBytesPreamble = Preamble->Preamble.getSize();
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
  PreamblePeer.stop();
  // We are no longer going to build any preambles, let the waiters know that.
  PreambleCV.notify_all();
  Status.stop();
  RequestsCV.notify_all();
}

void ASTWorker::runTask(llvm::StringRef Name, llvm::function_ref<void()> Task) {
  ThreadCrashReporter ScopedReporter([this, Name]() {
    llvm::errs() << "Signalled during AST worker action: " << Name << "\n";
    crashDumpParseInputs(llvm::errs(), FileInputs);
  });
  trace::Span Tracer(Name);
  WithContext WithProvidedContext(ContextProvider(FileName));
  Task();
}

void ASTWorker::startTask(llvm::StringRef Name,
                          llvm::unique_function<void()> Task,
                          llvm::Optional<UpdateType> Update,
                          TUScheduler::ASTActionInvalidation Invalidation) {
  if (RunSync) {
    assert(!Done && "running a task after stop()");
    runTask(Name, Task);
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "running a task after stop()");
    // Cancel any requests invalidated by this request.
    if (Update && Update->ContentChanged) {
      for (auto &R : llvm::reverse(Requests)) {
        if (R.InvalidationPolicy == TUScheduler::InvalidateOnUpdate)
          R.Invalidate();
        if (R.Update && R.Update->ContentChanged)
          break; // Older requests were already invalidated by the older update.
      }
    }

    // Allow this request to be cancelled if invalidated.
    Context Ctx = Context::current().derive(FileBeingProcessed, FileName);
    Canceler Invalidate = nullptr;
    if (Invalidation) {
      WithContext WC(std::move(Ctx));
      std::tie(Ctx, Invalidate) = cancelableTask(
          /*Reason=*/static_cast<int>(ErrorCode::ContentModified));
    }
    // Trace the time the request spends in the queue, and the requests that
    // it's going to wait for.
    llvm::Optional<Context> QueueCtx;
    if (trace::enabled()) {
      // Tracers that follow threads and need strict nesting will see a tiny
      // instantaneous event "we're enqueueing", and sometime later it runs.
      WithContext WC(Ctx.clone());
      trace::Span Tracer("Queued:" + Name);
      if (Tracer.Args) {
        if (CurrentRequest)
          SPAN_ATTACH(Tracer, "CurrentRequest", CurrentRequest->Name);
        llvm::json::Array PreambleRequestsNames;
        for (const auto &Req : PreambleRequests)
          PreambleRequestsNames.push_back(Req.Name);
        SPAN_ATTACH(Tracer, "PreambleRequestsNames",
                    std::move(PreambleRequestsNames));
        llvm::json::Array RequestsNames;
        for (const auto &Req : Requests)
          RequestsNames.push_back(Req.Name);
        SPAN_ATTACH(Tracer, "RequestsNames", std::move(RequestsNames));
      }
      // For tracers that follow contexts, keep the trace span's context alive
      // until we dequeue the request, so they see the full duration.
      QueueCtx = Context::current().clone();
    }
    Requests.push_back({std::move(Task), std::string(Name), steady_clock::now(),
                        std::move(Ctx), std::move(QueueCtx), Update,
                        Invalidation, std::move(Invalidate)});
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
          // Even though Done is set, finish pending requests.
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
        PreambleRequests.pop_front();
      } else {
        CurrentRequest = std::move(Requests.front());
        Requests.pop_front();
      }
    } // unlock Mutex

    // Inform tracing that the request was dequeued.
    CurrentRequest->QueueCtx.reset();

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
      Status.update([&](TUStatus &Status) {
        Status.ASTActivity.K = ASTAction::RunningAction;
        Status.ASTActivity.Name = CurrentRequest->Name;
      });
      runTask(CurrentRequest->Name, CurrentRequest->Action);
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
      if (I->Update == None)
        break;
      continue;
    }
    // Cancelled reads are moved to the front of the queue and run immediately.
    if (I->Update == None) {
      Request R = std::move(*I);
      Requests.erase(I);
      Requests.push_front(std::move(R));
      return Deadline::zero();
    }
    // Cancelled updates are downgraded to auto-diagnostics, and may be elided.
    if (I->Update->Diagnostics == WantDiagnostics::Yes)
      I->Update->Diagnostics = WantDiagnostics::Auto;
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
    if (R.Update == None || R.Update->Diagnostics == WantDiagnostics::Yes)
      return Deadline::zero();
  // Front request needs to be debounced, so determine when we're ready.
  Deadline D(Requests.front().AddTime + UpdateDebounce.compute(RebuildTimes));
  return D;
}

// Returns true if Requests.front() is a dead update that can be skipped.
bool ASTWorker::shouldSkipHeadLocked() const {
  assert(!Requests.empty());
  auto Next = Requests.begin();
  auto Update = Next->Update;
  if (!Update) // Only skip updates.
    return false;
  ++Next;
  // An update is live if its AST might still be read.
  // That is, if it's not immediately followed by another update.
  if (Next == Requests.end() || !Next->Update)
    return false;
  // The other way an update can be live is if its diagnostics might be used.
  switch (Update->Diagnostics) {
  case WantDiagnostics::Yes:
    return false; // Always used.
  case WantDiagnostics::No:
    return true; // Always dead.
  case WantDiagnostics::Auto:
    // Used unless followed by an update that generates diagnostics.
    for (; Next != Requests.end(); ++Next)
      if (Next->Update && Next->Update->Diagnostics != WantDiagnostics::No)
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
  if (!WaitUntilASTWorkerIsIdle())
    return false;
  // Now that ASTWorker processed all requests, ensure PreamblePeer has served
  // all update requests. This might create new PreambleRequests for the
  // ASTWorker.
  if (!PreamblePeer.blockUntilIdle(Timeout))
    return false;
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
  case PreambleAction::Queued:
    Result.push_back("includes are queued");
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
  return llvm::join(Result, ", ");
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
      Callbacks(Callbacks ? std::move(Callbacks)
                          : std::make_unique<ParsingCallbacks>()),
      Barrier(Opts.AsyncThreadsCount), QuickRunBarrier(Opts.AsyncThreadsCount),
      IdleASTs(
          std::make_unique<ASTCache>(Opts.RetentionPolicy.MaxRetainedASTs)),
      HeaderIncluders(std::make_unique<HeaderIncluderCache>()) {
  // Avoid null checks everywhere.
  if (!Opts.ContextProvider) {
    this->Opts.ContextProvider = [](llvm::StringRef) {
      return Context::current().clone();
    };
  }
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
  bool ContentChanged = false;
  if (!FD) {
    // Create a new worker to process the AST-related tasks.
    ASTWorkerHandle Worker =
        ASTWorker::create(File, CDB, *IdleASTs, *HeaderIncluders,
                          WorkerThreads ? WorkerThreads.getPointer() : nullptr,
                          Barrier, Opts, *Callbacks);
    FD = std::unique_ptr<FileData>(
        new FileData{Inputs.Contents, std::move(Worker)});
    ContentChanged = true;
  } else if (FD->Contents != Inputs.Contents) {
    ContentChanged = true;
    FD->Contents = Inputs.Contents;
  }
  FD->Worker->update(std::move(Inputs), WantDiags, ContentChanged);
  // There might be synthetic update requests, don't change the LastActiveFile
  // in such cases.
  if (ContentChanged)
    LastActiveFile = File.str();
  return NewFile;
}

void TUScheduler::remove(PathRef File) {
  bool Removed = Files.erase(File);
  if (!Removed)
    elog("Trying to remove file from TUScheduler that is not tracked: {0}",
         File);
  // We don't call HeaderIncluders.remove(File) here.
  // If we did, we'd avoid potentially stale header/mainfile associations.
  // However, it would mean that closing a mainfile could invalidate the
  // preamble of several open headers.
}

void TUScheduler::run(llvm::StringRef Name, llvm::StringRef Path,
                      llvm::unique_function<void()> Action) {
  runWithSemaphore(Name, Path, std::move(Action), Barrier);
}

void TUScheduler::runQuick(llvm::StringRef Name, llvm::StringRef Path,
                           llvm::unique_function<void()> Action) {
  // Use QuickRunBarrier to serialize quick tasks: we are ignoring
  // the parallelism level set by the user, don't abuse it
  runWithSemaphore(Name, Path, std::move(Action), QuickRunBarrier);
}

void TUScheduler::runWithSemaphore(llvm::StringRef Name, llvm::StringRef Path,
                                   llvm::unique_function<void()> Action,
                                   Semaphore &Sem) {
  if (Path.empty())
    Path = LastActiveFile;
  else
    LastActiveFile = Path.str();
  if (!PreambleTasks) {
    WithContext WithProvidedContext(Opts.ContextProvider(Path));
    return Action();
  }
  PreambleTasks->runAsync(Name, [this, &Sem, Ctx = Context::current().clone(),
                                 Path(Path.str()),
                                 Action = std::move(Action)]() mutable {
    std::lock_guard<Semaphore> BarrierLock(Sem);
    WithContext WC(std::move(Ctx));
    WithContext WithProvidedContext(Opts.ContextProvider(Path));
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
  LastActiveFile = File.str();

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
  LastActiveFile = File.str();

  if (!PreambleTasks) {
    trace::Span Tracer(Name);
    SPAN_ATTACH(Tracer, "file", File);
    std::shared_ptr<const ASTSignals> Signals;
    std::shared_ptr<const PreambleData> Preamble =
        It->second->Worker->getPossiblyStalePreamble(&Signals);
    WithContext WithProvidedContext(Opts.ContextProvider(File));
    Action(InputsAndPreamble{It->second->Contents,
                             It->second->Worker->getCurrentCompileCommand(),
                             Preamble.get(), Signals.get()});
    return;
  }

  std::shared_ptr<const ASTWorker> Worker = It->second->Worker.lock();
  auto Task = [Worker, Consistency, Name = Name.str(), File = File.str(),
               Contents = It->second->Contents,
               Command = Worker->getCurrentCompileCommand(),
               Ctx = Context::current().derive(FileBeingProcessed,
                                               std::string(File)),
               Action = std::move(Action), this]() mutable {
    ThreadCrashReporter ScopedReporter([&Name, &Contents, &Command]() {
      llvm::errs() << "Signalled during preamble action: " << Name << "\n";
      crashDumpCompileCommand(llvm::errs(), Command);
      crashDumpFileContents(llvm::errs(), Contents);
    });
    std::shared_ptr<const PreambleData> Preamble;
    if (Consistency == PreambleConsistency::Stale) {
      // Wait until the preamble is built for the first time, if preamble
      // is required. This avoids extra work of processing the preamble
      // headers in parallel multiple times.
      Worker->waitForFirstPreamble();
    }
    std::shared_ptr<const ASTSignals> Signals;
    Preamble = Worker->getPossiblyStalePreamble(&Signals);

    std::lock_guard<Semaphore> BarrierLock(Barrier);
    WithContext Guard(std::move(Ctx));
    trace::Span Tracer(Name);
    SPAN_ATTACH(Tracer, "file", File);
    WithContext WithProvidedContext(Opts.ContextProvider(File));
    Action(InputsAndPreamble{Contents, Command, Preamble.get(), Signals.get()});
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
  auto *Median = Recent.begin() + Recent.size() / 2;
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

void TUScheduler::profile(MemoryTree &MT) const {
  for (const auto &Elem : fileStats()) {
    MT.detail(Elem.first())
        .child("preamble")
        .addUsage(Opts.StorePreamblesInMemory ? Elem.second.UsedBytesPreamble
                                              : 0);
    MT.detail(Elem.first()).child("ast").addUsage(Elem.second.UsedBytesAST);
    MT.child("header_includer_cache").addUsage(HeaderIncluders->getUsedBytes());
  }
}
} // namespace clangd
} // namespace clang
