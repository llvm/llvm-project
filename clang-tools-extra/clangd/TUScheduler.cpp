//===--- TUScheduler.cpp -----------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// For each file, managed by TUScheduler, we create a single ASTWorker that
// manages an AST for that file. All operations that modify or read the AST are
// run on a separate dedicated thread asynchronously in FIFO order.
//
// We start processing each update immediately after we receive it. If two or
// more updates come subsequently without reads in-between, we attempt to drop
// an older one to not waste time building the ASTs we don't need.
//
// The processing thread of the ASTWorker is also responsible for building the
// preamble. However, unlike AST, the same preamble can be read concurrently, so
// we run each of async preamble reads on its own thread.
//
// To limit the concurrent load that clangd produces we maintain a semaphore
// that keeps more than a fixed number of threads from running concurrently.
//
// Rationale for cancelling updates.
// LSP clients can send updates to clangd on each keystroke. Some files take
// significant time to parse (e.g. a few seconds) and clangd can get starved by
// the updates to those files. Therefore we try to process only the last update,
// if possible.
// Our current strategy to do that is the following:
// - For each update we immediately schedule rebuild of the AST.
// - Rebuild of the AST checks if it was cancelled before doing any actual work.
//   If it was, it does not do an actual rebuild, only reports llvm::None to the
//   callback
// - When adding an update, we cancel the last update in the queue if it didn't
//   have any reads.
// There is probably a optimal ways to do that. One approach we might take is
// the following:
// - For each update we remember the pending inputs, but delay rebuild of the
//   AST for some timeout.
// - If subsequent updates come before rebuild was started, we replace the
//   pending inputs and reset the timer.
// - If any reads of the AST are scheduled, we start building the AST
//   immediately.

#include "TUScheduler.h"
#include "Cancellation.h"
#include "Logger.h"
#include "Trace.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <memory>
#include <queue>
#include <thread>

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
  llvm::Optional<std::unique_ptr<ParsedAST>> take(Key K) {
    std::unique_lock<std::mutex> Lock(Mut);
    auto Existing = findByKey(K);
    if (Existing == LRU.end())
      return None;
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
  ASTWorker(PathRef FileName, TUScheduler::ASTCache &LRUCache,
            Semaphore &Barrier, bool RunSync,
            steady_clock::duration UpdateDebounce,
            std::shared_ptr<PCHContainerOperations> PCHs,
            bool StorePreamblesInMemory, ParsingCallbacks &Callbacks);

public:
  /// Create a new ASTWorker and return a handle to it.
  /// The processing thread is spawned using \p Tasks. However, when \p Tasks
  /// is null, all requests will be processed on the calling thread
  /// synchronously instead. \p Barrier is acquired when processing each
  /// request, it is used to limit the number of actively running threads.
  static ASTWorkerHandle create(PathRef FileName,
                                TUScheduler::ASTCache &IdleASTs,
                                AsyncTaskRunner *Tasks, Semaphore &Barrier,
                                steady_clock::duration UpdateDebounce,
                                std::shared_ptr<PCHContainerOperations> PCHs,
                                bool StorePreamblesInMemory,
                                ParsingCallbacks &Callbacks);
  ~ASTWorker();

  void update(ParseInputs Inputs, WantDiagnostics);
  void
  runWithAST(llvm::StringRef Name,
             llvm::unique_function<void(llvm::Expected<InputsAndAST>)> Action);
  bool blockUntilIdle(Deadline Timeout) const;

  std::shared_ptr<const PreambleData> getPossiblyStalePreamble() const;
  /// Obtain a preamble reflecting all updates so far. Threadsafe.
  /// It may be delivered immediately, or later on the worker thread.
  void getCurrentPreamble(
      llvm::unique_function<void(std::shared_ptr<const PreambleData>)>);
  /// Wait for the first build of preamble to finish. Preamble itself can be
  /// accessed via getPossiblyStalePreamble(). Note that this function will
  /// return after an unsuccessful build of the preamble too, i.e. result of
  /// getPossiblyStalePreamble() can be null even after this function returns.
  void waitForFirstPreamble() const;

  std::size_t getUsedBytes() const;
  bool isASTCached() const;

private:
  // Must be called exactly once on processing thread. Will return after
  // stop() is called on a separate thread and all pending requests are
  // processed.
  void run();
  /// Signal that run() should finish processing pending requests and exit.
  void stop();
  /// Adds a new task to the end of the request queue.
  void startTask(llvm::StringRef Name, llvm::unique_function<void()> Task,
                 llvm::Optional<WantDiagnostics> UpdateType);
  /// Updates the TUStatus and emits it. Only called in the worker thread.
  void emitTUStatus(TUAction FAction,
                    const TUStatus::BuildDetails *Detail = nullptr);

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
  };

  /// Handles retention of ASTs.
  TUScheduler::ASTCache &IdleASTs;
  const bool RunSync;
  /// Time to wait after an update to see whether another update obsoletes it.
  const steady_clock::duration UpdateDebounce;
  /// File that ASTWorker is responsible for.
  const Path FileName;
  /// Whether to keep the built preambles in memory or on disk.
  const bool StorePreambleInMemory;
  /// Callback invoked when preamble or main file AST is built.
  ParsingCallbacks &Callbacks;
  /// Helper class required to build the ASTs.
  const std::shared_ptr<PCHContainerOperations> PCHs;
  /// Only accessed by the worker thread.
  TUStatus Status;

  Semaphore &Barrier;
  /// Inputs, corresponding to the current state of AST.
  ParseInputs FileInputs;
  /// Whether the diagnostics for the current FileInputs were reported to the
  /// users before.
  bool DiagsWereReported = false;
  /// Size of the last AST
  /// Guards members used by both TUScheduler and the worker thread.
  mutable std::mutex Mutex;
  std::shared_ptr<const PreambleData> LastBuiltPreamble; /* GUARDED_BY(Mutex) */
  /// Becomes ready when the first preamble build finishes.
  Notification PreambleWasBuilt;
  /// Set to true to signal run() to finish processing.
  bool Done;                    /* GUARDED_BY(Mutex) */
  std::deque<Request> Requests; /* GUARDED_BY(Mutex) */
  mutable std::condition_variable RequestsCV;
  // FIXME: rename it to better fix the current usage, we also use it to guard
  // emitting TUStatus.
  /// Guards a critical section for running the diagnostics callbacks.
  std::mutex DiagsMu;
  // Used to prevent remove document + leading to out-of-order diagnostics:
  // The lifetime of the old/new ASTWorkers will overlap, but their handles
  // don't. When the old handle is destroyed, the old worker will stop reporting
  // diagnostics.
  bool ReportDiagnostics = true; /* GUARDED_BY(DiagMu) */
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
                                  TUScheduler::ASTCache &IdleASTs,
                                  AsyncTaskRunner *Tasks, Semaphore &Barrier,
                                  steady_clock::duration UpdateDebounce,
                                  std::shared_ptr<PCHContainerOperations> PCHs,
                                  bool StorePreamblesInMemory,
                                  ParsingCallbacks &Callbacks) {
  std::shared_ptr<ASTWorker> Worker(new ASTWorker(
      FileName, IdleASTs, Barrier, /*RunSync=*/!Tasks, UpdateDebounce,
      std::move(PCHs), StorePreamblesInMemory, Callbacks));
  if (Tasks)
    Tasks->runAsync("worker:" + llvm::sys::path::filename(FileName),
                    [Worker]() { Worker->run(); });

  return ASTWorkerHandle(std::move(Worker));
}

ASTWorker::ASTWorker(PathRef FileName, TUScheduler::ASTCache &LRUCache,
                     Semaphore &Barrier, bool RunSync,
                     steady_clock::duration UpdateDebounce,
                     std::shared_ptr<PCHContainerOperations> PCHs,
                     bool StorePreamblesInMemory, ParsingCallbacks &Callbacks)
    : IdleASTs(LRUCache), RunSync(RunSync), UpdateDebounce(UpdateDebounce),
      FileName(FileName), StorePreambleInMemory(StorePreamblesInMemory),
      Callbacks(Callbacks),
      PCHs(std::move(PCHs)), Status{TUAction(TUAction::Idle, ""),
                                    TUStatus::BuildDetails()},
      Barrier(Barrier), Done(false) {}

ASTWorker::~ASTWorker() {
  // Make sure we remove the cached AST, if any.
  IdleASTs.take(this);
#ifndef NDEBUG
  std::lock_guard<std::mutex> Lock(Mutex);
  assert(Done && "handle was not destroyed");
  assert(Requests.empty() && "unprocessed requests when destroying ASTWorker");
#endif
}

void ASTWorker::update(ParseInputs Inputs, WantDiagnostics WantDiags) {
  llvm::StringRef TaskName = "Update";
  auto Task = [=]() mutable {
    // Will be used to check if we can avoid rebuilding the AST.
    bool InputsAreTheSame =
        std::tie(FileInputs.CompileCommand, FileInputs.Contents) ==
        std::tie(Inputs.CompileCommand, Inputs.Contents);

    tooling::CompileCommand OldCommand = std::move(FileInputs.CompileCommand);
    bool PrevDiagsWereReported = DiagsWereReported;
    FileInputs = Inputs;
    DiagsWereReported = false;
    emitTUStatus({TUAction::BuildingPreamble, TaskName});
    log("Updating file {0} with command [{1}] {2}", FileName,
        Inputs.CompileCommand.Directory,
        llvm::join(Inputs.CompileCommand.CommandLine, " "));
    // Rebuild the preamble and the AST.
    std::unique_ptr<CompilerInvocation> Invocation =
        buildCompilerInvocation(Inputs);
    if (!Invocation) {
      elog("Could not build CompilerInvocation for file {0}", FileName);
      // Remove the old AST if it's still in cache.
      IdleASTs.take(this);
      TUStatus::BuildDetails Details;
      Details.BuildFailed = true;
      emitTUStatus({TUAction::BuildingPreamble, TaskName}, &Details);
      // Make sure anyone waiting for the preamble gets notified it could not
      // be built.
      PreambleWasBuilt.notify();
      return;
    }

    std::shared_ptr<const PreambleData> OldPreamble =
        getPossiblyStalePreamble();
    std::shared_ptr<const PreambleData> NewPreamble = buildPreamble(
        FileName, *Invocation, OldPreamble, OldCommand, Inputs, PCHs,
        StorePreambleInMemory,
        [this](ASTContext &Ctx, std::shared_ptr<clang::Preprocessor> PP) {
          Callbacks.onPreambleAST(FileName, Ctx, std::move(PP));
        });

    bool CanReuseAST = InputsAreTheSame && (OldPreamble == NewPreamble);
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      LastBuiltPreamble = NewPreamble;
    }
    // Before doing the expensive AST reparse, we want to release our reference
    // to the old preamble, so it can be freed if there are no other references
    // to it.
    OldPreamble.reset();
    PreambleWasBuilt.notify();
    emitTUStatus({TUAction::BuildingFile, TaskName});
    if (!CanReuseAST) {
      IdleASTs.take(this); // Remove the old AST if it's still in cache.
    } else {
      // Since we don't need to rebuild the AST, we might've already reported
      // the diagnostics for it.
      if (PrevDiagsWereReported) {
        DiagsWereReported = true;
        // Take a shortcut and don't report the diagnostics, since they should
        // not changed. All the clients should handle the lack of OnUpdated()
        // call anyway to handle empty result from buildAST.
        // FIXME(ibiryukov): the AST could actually change if non-preamble
        // includes changed, but we choose to ignore it.
        // FIXME(ibiryukov): should we refresh the cache in IdleASTs for the
        // current file at this point?
        log("Skipping rebuild of the AST for {0}, inputs are the same.",
            FileName);
        TUStatus::BuildDetails Details;
        Details.ReuseAST = true;
        emitTUStatus({TUAction::BuildingFile, TaskName}, &Details);
        return;
      }
    }

    // We only need to build the AST if diagnostics were requested.
    if (WantDiags == WantDiagnostics::No)
      return;

    {
      std::lock_guard<std::mutex> Lock(DiagsMu);
      // No need to rebuild the AST if we won't send the diagnotics. However,
      // note that we don't prevent preamble rebuilds.
      if (!ReportDiagnostics)
        return;
    }

    // Get the AST for diagnostics.
    llvm::Optional<std::unique_ptr<ParsedAST>> AST = IdleASTs.take(this);
    if (!AST) {
      llvm::Optional<ParsedAST> NewAST =
          buildAST(FileName, std::move(Invocation), Inputs, NewPreamble, PCHs);
      AST = NewAST ? llvm::make_unique<ParsedAST>(std::move(*NewAST)) : nullptr;
      if (!(*AST)) { // buildAST fails.
        TUStatus::BuildDetails Details;
        Details.BuildFailed = true;
        emitTUStatus({TUAction::BuildingFile, TaskName}, &Details);
      }
    } else {
      // We are reusing the AST.
      TUStatus::BuildDetails Details;
      Details.ReuseAST = true;
      emitTUStatus({TUAction::BuildingFile, TaskName}, &Details);
    }
    // We want to report the diagnostics even if this update was cancelled.
    // It seems more useful than making the clients wait indefinitely if they
    // spam us with updates.
    // Note *AST can still be null if buildAST fails.
    if (*AST) {
      {
        std::lock_guard<std::mutex> Lock(DiagsMu);
        if (ReportDiagnostics)
          Callbacks.onDiagnostics(FileName, (*AST)->getDiagnostics());
      }
      trace::Span Span("Running main AST callback");
      Callbacks.onMainAST(FileName, **AST);
      DiagsWereReported = true;
    }
    // Stash the AST in the cache for further use.
    IdleASTs.put(this, std::move(*AST));
  };
  startTask(TaskName, std::move(Task), WantDiags);
}

void ASTWorker::runWithAST(
    llvm::StringRef Name,
    llvm::unique_function<void(llvm::Expected<InputsAndAST>)> Action) {
  auto Task = [=](decltype(Action) Action) {
    if (isCancelled())
      return Action(llvm::make_error<CancelledError>());
    llvm::Optional<std::unique_ptr<ParsedAST>> AST = IdleASTs.take(this);
    if (!AST) {
      std::unique_ptr<CompilerInvocation> Invocation =
          buildCompilerInvocation(FileInputs);
      // Try rebuilding the AST.
      llvm::Optional<ParsedAST> NewAST =
          Invocation
              ? buildAST(FileName,
                         llvm::make_unique<CompilerInvocation>(*Invocation),
                         FileInputs, getPossiblyStalePreamble(), PCHs)
              : None;
      AST = NewAST ? llvm::make_unique<ParsedAST>(std::move(*NewAST)) : nullptr;
    }
    // Make sure we put the AST back into the LRU cache.
    auto _ = llvm::make_scope_exit(
        [&AST, this]() { IdleASTs.put(this, std::move(*AST)); });
    // Run the user-provided action.
    if (!*AST)
      return Action(llvm::make_error<llvm::StringError>(
          "invalid AST", llvm::errc::invalid_argument));
    Action(InputsAndAST{FileInputs, **AST});
  };
  startTask(Name, Bind(Task, std::move(Action)),
            /*UpdateType=*/None);
}

std::shared_ptr<const PreambleData>
ASTWorker::getPossiblyStalePreamble() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return LastBuiltPreamble;
}

void ASTWorker::getCurrentPreamble(
    llvm::unique_function<void(std::shared_ptr<const PreambleData>)> Callback) {
  // We could just call startTask() to throw the read on the queue, knowing
  // it will run after any updates. But we know this task is cheap, so to
  // improve latency we cheat: insert it on the queue after the last update.
  std::unique_lock<std::mutex> Lock(Mutex);
  auto LastUpdate =
      std::find_if(Requests.rbegin(), Requests.rend(),
                   [](const Request &R) { return R.UpdateType.hasValue(); });
  // If there were no writes in the queue, the preamble is ready now.
  if (LastUpdate == Requests.rend()) {
    Lock.unlock();
    return Callback(getPossiblyStalePreamble());
  }
  assert(!RunSync && "Running synchronously, but queue is non-empty!");
  Requests.insert(LastUpdate.base(),
                  Request{Bind(
                              [this](decltype(Callback) Callback) {
                                Callback(getPossiblyStalePreamble());
                              },
                              std::move(Callback)),
                          "GetPreamble", steady_clock::now(),
                          Context::current().clone(),
                          /*UpdateType=*/None});
  Lock.unlock();
  RequestsCV.notify_all();
}

void ASTWorker::waitForFirstPreamble() const { PreambleWasBuilt.wait(); }

std::size_t ASTWorker::getUsedBytes() const {
  // Note that we don't report the size of ASTs currently used for processing
  // the in-flight requests. We used this information for debugging purposes
  // only, so this should be fine.
  std::size_t Result = IdleASTs.getUsedBytes(this);
  if (auto Preamble = getPossiblyStalePreamble())
    Result += Preamble->Preamble.getSize();
  return Result;
}

bool ASTWorker::isASTCached() const { return IdleASTs.getUsedBytes(this) != 0; }

void ASTWorker::stop() {
  {
    std::lock_guard<std::mutex> Lock(DiagsMu);
    ReportDiagnostics = false;
  }
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "stop() called twice");
    Done = true;
  }
  RequestsCV.notify_all();
}

void ASTWorker::startTask(llvm::StringRef Name,
                          llvm::unique_function<void()> Task,
                          llvm::Optional<WantDiagnostics> UpdateType) {
  if (RunSync) {
    assert(!Done && "running a task after stop()");
    trace::Span Tracer(Name + ":" + llvm::sys::path::filename(FileName));
    Task();
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "running a task after stop()");
    Requests.push_back(
        {std::move(Task), Name, steady_clock::now(),
         Context::current().derive(kFileBeingProcessed, FileName), UpdateType});
  }
  RequestsCV.notify_all();
}

void ASTWorker::emitTUStatus(TUAction Action,
                             const TUStatus::BuildDetails *Details) {
  Status.Action = std::move(Action);
  if (Details)
    Status.Details = *Details;
  std::lock_guard<std::mutex> Lock(DiagsMu);
  // Do not emit TU statuses when the ASTWorker is shutting down.
  if (ReportDiagnostics) {
    Callbacks.onFileUpdated(FileName, Status);
  }
}

void ASTWorker::run() {
  while (true) {
    Request Req;
    {
      std::unique_lock<std::mutex> Lock(Mutex);
      for (auto Wait = scheduleLocked(); !Wait.expired();
           Wait = scheduleLocked()) {
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
            emitTUStatus({TUAction::Queued, Req.Name});
            SPAN_ATTACH(*Tracer, "sleep_ms",
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            Wait.time() - steady_clock::now())
                            .count());
          }
        }

        wait(Lock, RequestsCV, Wait);
      }
      Req = std::move(Requests.front());
      // Leave it on the queue for now, so waiters don't see an empty queue.
    } // unlock Mutex

    {
      std::unique_lock<Semaphore> Lock(Barrier, std::try_to_lock);
      if (!Lock.owns_lock()) {
        emitTUStatus({TUAction::Queued, Req.Name});
        Lock.lock();
      }
      WithContext Guard(std::move(Req.Ctx));
      trace::Span Tracer(Req.Name);
      emitTUStatus({TUAction::RunningAction, Req.Name});
      Req.Action();
    }

    bool IsEmpty = false;
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      Requests.pop_front();
      IsEmpty = Requests.empty();
    }
    if (IsEmpty)
      emitTUStatus({TUAction::Idle, /*Name*/ ""});
    RequestsCV.notify_all();
  }
}

Deadline ASTWorker::scheduleLocked() {
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

  while (shouldSkipHeadLocked())
    Requests.pop_front();
  assert(!Requests.empty() && "skipped the whole queue");
  // Some updates aren't dead yet, but never end up being used.
  // e.g. the first keystroke is live until obsoleted by the second.
  // We debounce "maybe-unused" writes, sleeping 500ms in case they become dead.
  // But don't delay reads (including updates where diagnostics are needed).
  for (const auto &R : Requests)
    if (R.UpdateType == None || R.UpdateType == WantDiagnostics::Yes)
      return Deadline::zero();
  // Front request needs to be debounced, so determine when we're ready.
  Deadline D(Requests.front().AddTime + UpdateDebounce);
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
  std::unique_lock<std::mutex> Lock(Mutex);
  return wait(Lock, RequestsCV, Timeout, [&] { return Requests.empty(); });
}

// Render a TUAction to a user-facing string representation.
// TUAction represents clangd-internal states, we don't intend to expose them
// to users (say C++ programmers) directly to avoid confusion, we use terms that
// are familiar by C++ programmers.
std::string renderTUAction(const TUAction &Action) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  switch (Action.S) {
  case TUAction::Queued:
    OS << "file is queued";
    break;
  case TUAction::RunningAction:
    OS << "running " << Action.Name;
    break;
  case TUAction::BuildingPreamble:
    OS << "parsing includes";
    break;
  case TUAction::BuildingFile:
    OS << "parsing main file";
    break;
  case TUAction::Idle:
    OS << "idle";
    break;
  }
  return OS.str();
}

} // namespace

unsigned getDefaultAsyncThreadsCount() {
  unsigned HardwareConcurrency = std::thread::hardware_concurrency();
  // C++ standard says that hardware_concurrency()
  // may return 0, fallback to 1 worker thread in
  // that case.
  if (HardwareConcurrency == 0)
    return 1;
  return HardwareConcurrency;
}

FileStatus TUStatus::render(PathRef File) const {
  FileStatus FStatus;
  FStatus.uri = URIForFile::canonicalize(File, /*TUPath=*/File);
  FStatus.state = renderTUAction(Action);
  return FStatus;
}

struct TUScheduler::FileData {
  /// Latest inputs, passed to TUScheduler::update().
  std::string Contents;
  tooling::CompileCommand Command;
  ASTWorkerHandle Worker;
};

TUScheduler::TUScheduler(unsigned AsyncThreadsCount,
                         bool StorePreamblesInMemory,
                         std::unique_ptr<ParsingCallbacks> Callbacks,
                         std::chrono::steady_clock::duration UpdateDebounce,
                         ASTRetentionPolicy RetentionPolicy)
    : StorePreamblesInMemory(StorePreamblesInMemory),
      PCHOps(std::make_shared<PCHContainerOperations>()),
      Callbacks(Callbacks ? move(Callbacks)
                          : llvm::make_unique<ParsingCallbacks>()),
      Barrier(AsyncThreadsCount),
      IdleASTs(llvm::make_unique<ASTCache>(RetentionPolicy.MaxRetainedASTs)),
      UpdateDebounce(UpdateDebounce) {
  if (0 < AsyncThreadsCount) {
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

void TUScheduler::update(PathRef File, ParseInputs Inputs,
                         WantDiagnostics WantDiags) {
  std::unique_ptr<FileData> &FD = Files[File];
  if (!FD) {
    // Create a new worker to process the AST-related tasks.
    ASTWorkerHandle Worker = ASTWorker::create(
        File, *IdleASTs, WorkerThreads ? WorkerThreads.getPointer() : nullptr,
        Barrier, UpdateDebounce, PCHOps, StorePreamblesInMemory, *Callbacks);
    FD = std::unique_ptr<FileData>(new FileData{
        Inputs.Contents, Inputs.CompileCommand, std::move(Worker)});
  } else {
    FD->Contents = Inputs.Contents;
    FD->Command = Inputs.CompileCommand;
  }
  FD->Worker->update(std::move(Inputs), WantDiags);
}

void TUScheduler::remove(PathRef File) {
  bool Removed = Files.erase(File);
  if (!Removed)
    elog("Trying to remove file from TUScheduler that is not tracked: {0}",
         File);
}

void TUScheduler::run(llvm::StringRef Name,
                      llvm::unique_function<void()> Action) {
  if (!PreambleTasks)
    return Action();
  PreambleTasks->runAsync(Name, std::move(Action));
}

void TUScheduler::runWithAST(
    llvm::StringRef Name, PathRef File,
    llvm::unique_function<void(llvm::Expected<InputsAndAST>)> Action) {
  auto It = Files.find(File);
  if (It == Files.end()) {
    Action(llvm::make_error<LSPError>(
        "trying to get AST for non-added document", ErrorCode::InvalidParams));
    return;
  }

  It->second->Worker->runWithAST(Name, std::move(Action));
}

void TUScheduler::runWithPreamble(
    llvm::StringRef Name, PathRef File, PreambleConsistency Consistency,
    llvm::unique_function<void(llvm::Expected<InputsAndPreamble>)> Action) {
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
    Action(InputsAndPreamble{It->second->Contents, It->second->Command,
                             Preamble.get()});
    return;
  }

  // Future is populated if the task needs a specific preamble.
  std::future<std::shared_ptr<const PreambleData>> ConsistentPreamble;
  if (Consistency == Consistent) {
    std::promise<std::shared_ptr<const PreambleData>> Promise;
    ConsistentPreamble = Promise.get_future();
    It->second->Worker->getCurrentPreamble(Bind(
        [](decltype(Promise) Promise,
           std::shared_ptr<const PreambleData> Preamble) {
          Promise.set_value(std::move(Preamble));
        },
        std::move(Promise)));
  }

  std::shared_ptr<const ASTWorker> Worker = It->second->Worker.lock();
  auto Task = [Worker, this](std::string Name, std::string File,
                             std::string Contents,
                             tooling::CompileCommand Command, Context Ctx,
                             decltype(ConsistentPreamble) ConsistentPreamble,
                             decltype(Action) Action) mutable {
    std::shared_ptr<const PreambleData> Preamble;
    if (ConsistentPreamble.valid()) {
      Preamble = ConsistentPreamble.get();
    } else {
      // We don't want to be running preamble actions before the preamble was
      // built for the first time. This avoids extra work of processing the
      // preamble headers in parallel multiple times.
      Worker->waitForFirstPreamble();
      Preamble = Worker->getPossiblyStalePreamble();
    }

    std::lock_guard<Semaphore> BarrierLock(Barrier);
    WithContext Guard(std::move(Ctx));
    trace::Span Tracer(Name);
    SPAN_ATTACH(Tracer, "file", File);
    Action(InputsAndPreamble{Contents, Command, Preamble.get()});
  };

  PreambleTasks->runAsync(
      "task:" + llvm::sys::path::filename(File),
      Bind(Task, std::string(Name), std::string(File), It->second->Contents,
           It->second->Command,
           Context::current().derive(kFileBeingProcessed, File),
           std::move(ConsistentPreamble), std::move(Action)));
}

std::vector<std::pair<Path, std::size_t>>
TUScheduler::getUsedBytesPerFile() const {
  std::vector<std::pair<Path, std::size_t>> Result;
  Result.reserve(Files.size());
  for (auto &&PathAndFile : Files)
    Result.push_back(
        {PathAndFile.first(), PathAndFile.second->Worker->getUsedBytes()});
  return Result;
}

std::vector<Path> TUScheduler::getFilesWithCachedAST() const {
  std::vector<Path> Result;
  for (auto &&PathAndFile : Files) {
    if (!PathAndFile.second->Worker->isASTCached())
      continue;
    Result.push_back(PathAndFile.first());
  }
  return Result;
}

} // namespace clangd
} // namespace clang
