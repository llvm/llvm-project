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
#include "clang/Frontend/PCHContainerOperations.h"
#include "llvm/Support/Errc.h"
#include <memory>
#include <queue>

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
  ASTWorker(Semaphore &Barrier, std::shared_ptr<CppFile> AST, bool RunSync);

public:
  /// Create a new ASTWorker and return a handle to it.
  /// The processing thread is spawned using \p Tasks. However, when \p Tasks
  /// is null, all requests will be processed on the calling thread
  /// synchronously instead. \p Barrier is acquired when processing each
  /// request, it is be used to limit the number of actively running threads.
  static ASTWorkerHandle Create(AsyncTaskRunner *Tasks, Semaphore &Barrier,
                                std::shared_ptr<CppFile> AST);
  ~ASTWorker();

  void update(ParseInputs Inputs,
              UniqueFunction<void(llvm::Optional<std::vector<DiagWithFixIts>>)>
                  OnUpdated);
  void runWithAST(UniqueFunction<void(llvm::Expected<InputsAndAST>)> Action);

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
  void startTask(UniqueFunction<void()> Task, bool isUpdate,
                 llvm::Optional<CancellationFlag> CF);

  using RequestWithCtx = std::pair<UniqueFunction<void()>, Context>;

  const bool RunSync;
  Semaphore &Barrier;
  // AST and FileInputs are only accessed on the processing thread from run().
  const std::shared_ptr<CppFile> AST;
  // Inputs, corresponding to the current state of AST.
  ParseInputs FileInputs;
  // Guards members used by both TUScheduler and the worker thread.
  mutable std::mutex Mutex;
  // Set to true to signal run() to finish processing.
  bool Done;                           /* GUARDED_BY(Mutex) */
  std::queue<RequestWithCtx> Requests; /* GUARDED_BY(Mutex) */
  // Only set when last request is an update. This allows us to cancel an update
  // that was never read, if a subsequent update comes in.
  llvm::Optional<CancellationFlag> LastUpdateCF; /* GUARDED_BY(Mutex) */
  std::condition_variable RequestsCV;
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

ASTWorkerHandle ASTWorker::Create(AsyncTaskRunner *Tasks, Semaphore &Barrier,
                                  std::shared_ptr<CppFile> AST) {
  std::shared_ptr<ASTWorker> Worker(
      new ASTWorker(Barrier, std::move(AST), /*RunSync=*/!Tasks));
  if (Tasks)
    Tasks->runAsync([Worker]() { Worker->run(); });

  return ASTWorkerHandle(std::move(Worker));
}

ASTWorker::ASTWorker(Semaphore &Barrier, std::shared_ptr<CppFile> AST,
                     bool RunSync)
    : RunSync(RunSync), Barrier(Barrier), AST(std::move(AST)), Done(false) {
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
    ParseInputs Inputs,
    UniqueFunction<void(llvm::Optional<std::vector<DiagWithFixIts>>)>
        OnUpdated) {
  auto Task = [=](CancellationFlag CF, decltype(OnUpdated) OnUpdated) mutable {
    if (CF.isCancelled()) {
      OnUpdated(llvm::None);
      return;
    }
    FileInputs = Inputs;
    auto Diags = AST->rebuild(std::move(Inputs));
    // We want to report the diagnostics even if this update was cancelled.
    // It seems more useful than making the clients wait indefinitely if they
    // spam us with updates.
    OnUpdated(std::move(Diags));
  };

  CancellationFlag UpdateCF;
  startTask(BindWithForward(Task, UpdateCF, std::move(OnUpdated)),
            /*isUpdate=*/true, UpdateCF);
}

void ASTWorker::runWithAST(
    UniqueFunction<void(llvm::Expected<InputsAndAST>)> Action) {
  auto Task = [=](decltype(Action) Action) {
    auto ASTWrapper = this->AST->getAST().get();
    // FIXME: no need to lock here, cleanup the CppFile interface to get rid of
    // them.
    ASTWrapper->runUnderLock([&](ParsedAST *AST) {
      if (!AST) {
        Action(llvm::make_error<llvm::StringError>(
            "invalid AST", llvm::errc::invalid_argument));
        return;
      }
      Action(InputsAndAST{FileInputs, *AST});
    });
  };

  startTask(BindWithForward(Task, std::move(Action)), /*isUpdate=*/false,
            llvm::None);
}

std::shared_ptr<const PreambleData>
ASTWorker::getPossiblyStalePreamble() const {
  return AST->getPossiblyStalePreamble();
}

std::size_t ASTWorker::getUsedBytes() const {
  // FIXME(ibiryukov): we'll need to take locks here after we remove
  // thread-safety from CppFile. For now, CppFile is thread-safe and we can
  // safely call methods on it without acquiring a lock.
  return AST->getUsedBytes();
}

void ASTWorker::stop() {
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "stop() called twice");
    Done = true;
  }
  RequestsCV.notify_one();
}

void ASTWorker::startTask(UniqueFunction<void()> Task, bool isUpdate,
                          llvm::Optional<CancellationFlag> CF) {
  assert(isUpdate == CF.hasValue() &&
         "Only updates are expected to pass CancellationFlag");

  if (RunSync) {
    assert(!Done && "running a task after stop()");
    Task();
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    assert(!Done && "running a task after stop()");
    if (isUpdate) {
      if (!Requests.empty() && LastUpdateCF) {
        // There were no reads for the last unprocessed update, let's cancel it
        // to not waste time on it.
        LastUpdateCF->cancel();
      }
      LastUpdateCF = std::move(*CF);
    } else {
      LastUpdateCF = llvm::None;
    }
    Requests.emplace(std::move(Task), Context::current().clone());
  } // unlock Mutex.
  RequestsCV.notify_one();
}

void ASTWorker::run() {
  while (true) {
    RequestWithCtx Req;
    {
      std::unique_lock<std::mutex> Lock(Mutex);
      RequestsCV.wait(Lock, [&]() { return Done || !Requests.empty(); });
      if (Requests.empty()) {
        assert(Done);
        return;
      }
      // Even when Done is true, we finish processing all pending requests
      // before exiting the processing loop.

      Req = std::move(Requests.front());
      Requests.pop();
    } // unlock Mutex

    std::lock_guard<Semaphore> BarrierLock(Barrier);
    WithContext Guard(std::move(Req.second));
    Req.first();
  }
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
  if (0 < AsyncThreadsCount)
    Tasks.emplace();
}

TUScheduler::~TUScheduler() {
  // Notify all workers that they need to stop.
  Files.clear();

  // Wait for all in-flight tasks to finish.
  if (Tasks)
    Tasks->waitForAll();
}

void TUScheduler::update(
    PathRef File, ParseInputs Inputs,
    UniqueFunction<void(llvm::Optional<std::vector<DiagWithFixIts>>)>
        OnUpdated) {
  std::unique_ptr<FileData> &FD = Files[File];
  if (!FD) {
    // Create a new worker to process the AST-related tasks.
    ASTWorkerHandle Worker = ASTWorker::Create(
        Tasks ? Tasks.getPointer() : nullptr, Barrier,
        CppFile::Create(File, StorePreamblesInMemory, PCHOps, ASTCallback));
    FD = std::unique_ptr<FileData>(new FileData{Inputs, std::move(Worker)});
  } else {
    FD->Inputs = Inputs;
  }
  FD->Worker->update(std::move(Inputs), std::move(OnUpdated));
}

void TUScheduler::remove(PathRef File) {
  bool Removed = Files.erase(File);
  if (!Removed)
    log("Trying to remove file from TUScheduler that is not tracked. File:" +
        File);
}

void TUScheduler::runWithAST(
    PathRef File, UniqueFunction<void(llvm::Expected<InputsAndAST>)> Action) {
  auto It = Files.find(File);
  if (It == Files.end()) {
    Action(llvm::make_error<llvm::StringError>(
        "trying to get AST for non-added document",
        llvm::errc::invalid_argument));
    return;
  }

  It->second->Worker->runWithAST(std::move(Action));
}

void TUScheduler::runWithPreamble(
    PathRef File,
    UniqueFunction<void(llvm::Expected<InputsAndPreamble>)> Action) {
  auto It = Files.find(File);
  if (It == Files.end()) {
    Action(llvm::make_error<llvm::StringError>(
        "trying to get preamble for non-added document",
        llvm::errc::invalid_argument));
    return;
  }

  if (!Tasks) {
    std::shared_ptr<const PreambleData> Preamble =
        It->second->Worker->getPossiblyStalePreamble();
    Action(InputsAndPreamble{It->second->Inputs, Preamble.get()});
    return;
  }

  ParseInputs InputsCopy = It->second->Inputs;
  std::shared_ptr<const ASTWorker> Worker = It->second->Worker.lock();
  auto Task = [InputsCopy, Worker, this](Context Ctx,
                                         decltype(Action) Action) mutable {
    std::lock_guard<Semaphore> BarrierLock(Barrier);
    WithContext Guard(std::move(Ctx));
    std::shared_ptr<const PreambleData> Preamble =
        Worker->getPossiblyStalePreamble();
    Action(InputsAndPreamble{InputsCopy, Preamble.get()});
  };

  Tasks->runAsync(
      BindWithForward(Task, Context::current().clone(), std::move(Action)));
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
