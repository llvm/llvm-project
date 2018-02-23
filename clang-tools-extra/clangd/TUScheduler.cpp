//===--- TUScheduler.cpp -----------------------------------------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
// To limit the concurrent load that clangd produces we mantain a semaphore that
// keeps more than a fixed number of threads from running concurrently.
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
#include "Logger.h"
#include "Trace.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"
#include <memory>
#include <queue>
#include <thread>

namespace clang {
namespace clangd {
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
  ASTWorker(llvm::StringRef File, Semaphore &Barrier, CppFile AST,
            bool RunSync);

public:
  /// Create a new ASTWorker and return a handle to it.
  /// The processing thread is spawned using \p Tasks. However, when \p Tasks
  /// is null, all requests will be processed on the calling thread
  /// synchronously instead. \p Barrier is acquired when processing each
  /// request, it is be used to limit the number of actively running threads.
  static ASTWorkerHandle Create(llvm::StringRef File, AsyncTaskRunner *Tasks,
                                Semaphore &Barrier, CppFile AST);
  ~ASTWorker();

  void update(ParseInputs Inputs, WantDiagnostics,
              UniqueFunction<void(std::vector<DiagWithFixIts>)> OnUpdated);
  void runWithAST(llvm::StringRef Name,
                  UniqueFunction<void(llvm::Expected<InputsAndAST>)> Action);
  bool blockUntilIdle(Deadline Timeout) const;

  std::shared_ptr<const PreambleData> getPossiblyStalePreamble() const;
  std::size_t getUsedBytes() const;

private:
  // Must be called exactly once on processing thread. Will return after
  // stop() is called on a separate thread and all pending requests are
  // processed.
  void run();
  /// Signal that run() should finish processing pending requests and exit.
  void stop();
  /// Adds a new task to the end of the request queue.
  void startTask(llvm::StringRef Name, UniqueFunction<void()> Task,
                 llvm::Optional<WantDiagnostics> UpdateType);
  /// Should the first task in the queue be skipped instead of run?
  bool shouldSkipHeadLocked() const;

  struct Request {
    UniqueFunction<void()> Action;
    std::string Name;
    Context Ctx;
    llvm::Optional<WantDiagnostics> UpdateType;
  };

  std::string File;
  const bool RunSync;
  Semaphore &Barrier;
  // AST and FileInputs are only accessed on the processing thread from run().
  CppFile AST;
  // Inputs, corresponding to the current state of AST.
  ParseInputs FileInputs;
  // Guards members used by both TUScheduler and the worker thread.
  mutable std::mutex Mutex;
  std::shared_ptr<const PreambleData> LastBuiltPreamble; /* GUARDED_BY(Mutex) */
  // Result of getUsedBytes() after the last rebuild or read of AST.
  std::size_t LastASTSize; /* GUARDED_BY(Mutex) */
  // Set to true to signal run() to finish processing.
  bool Done;                           /* GUARDED_BY(Mutex) */
  std::deque<Request> Requests;        /* GUARDED_BY(Mutex) */
  mutable std::condition_variable RequestsCV;
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

ASTWorkerHandle ASTWorker::Create(llvm::StringRef File, AsyncTaskRunner *Tasks,
                                  Semaphore &Barrier, CppFile AST) {
  std::shared_ptr<ASTWorker> Worker(
      new ASTWorker(File, Barrier, std::move(AST), /*RunSync=*/!Tasks));
  if (Tasks)
    Tasks->runAsync("worker:" + llvm::sys::path::filename(File),
                    [Worker]() { Worker->run(); });

  return ASTWorkerHandle(std::move(Worker));
}

ASTWorker::ASTWorker(llvm::StringRef File, Semaphore &Barrier, CppFile AST,
                     bool RunSync)
    : File(File), RunSync(RunSync), Barrier(Barrier), AST(std::move(AST)),
      Done(false) {
  if (RunSync)
    return;
}

ASTWorker::~ASTWorker() {
#ifndef NDEBUG
  std::lock_guard<std::mutex> Lock(Mutex);
  assert(Done && "handle was not destroyed");
  assert(Requests.empty() && "unprocessed requests when destroying ASTWorker");
#endif
}

void ASTWorker::update(
    ParseInputs Inputs, WantDiagnostics WantDiags,
    UniqueFunction<void(std::vector<DiagWithFixIts>)> OnUpdated) {
  auto Task = [=](decltype(OnUpdated) OnUpdated) mutable {
    FileInputs = Inputs;
    auto Diags = AST.rebuild(std::move(Inputs));

    {
      std::lock_guard<std::mutex> Lock(Mutex);
      if (AST.getPreamble())
        LastBuiltPreamble = AST.getPreamble();
      LastASTSize = AST.getUsedBytes();
    }
    // We want to report the diagnostics even if this update was cancelled.
    // It seems more useful than making the clients wait indefinitely if they
    // spam us with updates.
    if (Diags && WantDiags != WantDiagnostics::No)
      OnUpdated(std::move(*Diags));
  };

  startTask("Update", Bind(Task, std::move(OnUpdated)), WantDiags);
}

void ASTWorker::runWithAST(
    llvm::StringRef Name,
    UniqueFunction<void(llvm::Expected<InputsAndAST>)> Action) {
  auto Task = [=](decltype(Action) Action) {
    ParsedAST *ActualAST = AST.getAST();
    if (!ActualAST) {
      Action(llvm::make_error<llvm::StringError>("invalid AST",
                                                 llvm::errc::invalid_argument));
      return;
    }
    Action(InputsAndAST{FileInputs, *ActualAST});

    // Size of the AST might have changed after reads too, e.g. if some decls
    // were deserialized from preamble.
    std::lock_guard<std::mutex> Lock(Mutex);
    LastASTSize = ActualAST->getUsedBytes();
  };

  startTask(Name, Bind(Task, std::move(Action)),
            /*UpdateType=*/llvm::None);
}

std::shared_ptr<const PreambleData>
ASTWorker::getPossiblyStalePreamble() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return LastBuiltPreamble;
}

std::size_t ASTWorker::getUsedBytes() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return LastASTSize;
}

void ASTWorker::stop() {
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "stop() called twice");
    Done = true;
  }
  RequestsCV.notify_all();
}

void ASTWorker::startTask(llvm::StringRef Name, UniqueFunction<void()> Task,
                          llvm::Optional<WantDiagnostics> UpdateType) {
  if (RunSync) {
    assert(!Done && "running a task after stop()");
    trace::Span Tracer(Name + ":" + llvm::sys::path::filename(File));
    Task();
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "running a task after stop()");
    Requests.push_back(
        {std::move(Task), Name, Context::current().clone(), UpdateType});
  }
  RequestsCV.notify_all();
}

void ASTWorker::run() {
  while (true) {
    Request Req;
    {
      std::unique_lock<std::mutex> Lock(Mutex);
      RequestsCV.wait(Lock, [&]() { return Done || !Requests.empty(); });
      if (Requests.empty()) {
        assert(Done);
        return;
      }
      // Even when Done is true, we finish processing all pending requests
      // before exiting the processing loop.

      while (shouldSkipHeadLocked())
        Requests.pop_front();
      assert(!Requests.empty() && "skipped the whole queue");
      Req = std::move(Requests.front());
      // Leave it on the queue for now, so waiters don't see an empty queue.
    } // unlock Mutex

    {
      std::lock_guard<Semaphore> BarrierLock(Barrier);
      WithContext Guard(std::move(Req.Ctx));
      trace::Span Tracer(Req.Name);
      Req.Action();
    }

    {
      std::lock_guard<std::mutex> Lock(Mutex);
      Requests.pop_front();
    }
    RequestsCV.notify_all();
  }
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

struct TUScheduler::FileData {
  /// Latest inputs, passed to TUScheduler::update().
  ParseInputs Inputs;
  ASTWorkerHandle Worker;
};

TUScheduler::TUScheduler(unsigned AsyncThreadsCount,
                         bool StorePreamblesInMemory,
                         ASTParsedCallback ASTCallback)
    : StorePreamblesInMemory(StorePreamblesInMemory),
      PCHOps(std::make_shared<PCHContainerOperations>()),
      ASTCallback(std::move(ASTCallback)), Barrier(AsyncThreadsCount) {
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

void TUScheduler::update(
    PathRef File, ParseInputs Inputs, WantDiagnostics WantDiags,
    UniqueFunction<void(std::vector<DiagWithFixIts>)> OnUpdated) {
  std::unique_ptr<FileData> &FD = Files[File];
  if (!FD) {
    // Create a new worker to process the AST-related tasks.
    ASTWorkerHandle Worker = ASTWorker::Create(
        File, WorkerThreads ? WorkerThreads.getPointer() : nullptr, Barrier,
        CppFile(File, StorePreamblesInMemory, PCHOps, ASTCallback));
    FD = std::unique_ptr<FileData>(new FileData{Inputs, std::move(Worker)});
  } else {
    FD->Inputs = Inputs;
  }
  FD->Worker->update(std::move(Inputs), WantDiags, std::move(OnUpdated));
}

void TUScheduler::remove(PathRef File) {
  bool Removed = Files.erase(File);
  if (!Removed)
    log("Trying to remove file from TUScheduler that is not tracked. File:" +
        File);
}

void TUScheduler::runWithAST(
    llvm::StringRef Name, PathRef File,
    UniqueFunction<void(llvm::Expected<InputsAndAST>)> Action) {
  auto It = Files.find(File);
  if (It == Files.end()) {
    Action(llvm::make_error<llvm::StringError>(
        "trying to get AST for non-added document",
        llvm::errc::invalid_argument));
    return;
  }

  It->second->Worker->runWithAST(Name, std::move(Action));
}

void TUScheduler::runWithPreamble(
    llvm::StringRef Name, PathRef File,
    UniqueFunction<void(llvm::Expected<InputsAndPreamble>)> Action) {
  auto It = Files.find(File);
  if (It == Files.end()) {
    Action(llvm::make_error<llvm::StringError>(
        "trying to get preamble for non-added document",
        llvm::errc::invalid_argument));
    return;
  }

  if (!PreambleTasks) {
    trace::Span Tracer(Name);
    SPAN_ATTACH(Tracer, "file", File);
    std::shared_ptr<const PreambleData> Preamble =
        It->second->Worker->getPossiblyStalePreamble();
    Action(InputsAndPreamble{It->second->Inputs, Preamble.get()});
    return;
  }

  ParseInputs InputsCopy = It->second->Inputs;
  std::shared_ptr<const ASTWorker> Worker = It->second->Worker.lock();
  auto Task = [InputsCopy, Worker, this](std::string Name, std::string File,
                                         Context Ctx,
                                         decltype(Action) Action) mutable {
    std::lock_guard<Semaphore> BarrierLock(Barrier);
    WithContext Guard(std::move(Ctx));
    trace::Span Tracer(Name);
    SPAN_ATTACH(Tracer, "file", File);
    std::shared_ptr<const PreambleData> Preamble =
        Worker->getPossiblyStalePreamble();
    Action(InputsAndPreamble{InputsCopy, Preamble.get()});
  };

  PreambleTasks->runAsync("task:" + llvm::sys::path::filename(File),
                          Bind(Task, std::string(Name), std::string(File),
                               Context::current().clone(), std::move(Action)));
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

} // namespace clangd
} // namespace clang
