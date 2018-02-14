#include "Threading.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include <thread>

namespace clang {
namespace clangd {

CancellationFlag::CancellationFlag()
    : WasCancelled(std::make_shared<std::atomic<bool>>(false)) {}

Semaphore::Semaphore(std::size_t MaxLocks) : FreeSlots(MaxLocks) {}

void Semaphore::lock() {
  std::unique_lock<std::mutex> Lock(Mutex);
  SlotsChanged.wait(Lock, [&]() { return FreeSlots > 0; });
  --FreeSlots;
}

void Semaphore::unlock() {
  std::unique_lock<std::mutex> Lock(Mutex);
  ++FreeSlots;
  Lock.unlock();

  SlotsChanged.notify_one();
}

AsyncTaskRunner::~AsyncTaskRunner() { wait(); }

bool AsyncTaskRunner::wait(Deadline D) const {
  std::unique_lock<std::mutex> Lock(Mutex);
  return clangd::wait(Lock, TasksReachedZero, D,
                      [&] { return InFlightTasks == 0; });
}

void AsyncTaskRunner::runAsync(UniqueFunction<void()> Action) {
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    ++InFlightTasks;
  }

  auto CleanupTask = llvm::make_scope_exit([this]() {
    std::lock_guard<std::mutex> Lock(Mutex);
    int NewTasksCnt = --InFlightTasks;
    if (NewTasksCnt == 0) {
      // Note: we can't unlock here because we don't want the object to be
      // destroyed before we notify.
      TasksReachedZero.notify_one();
    }
  });

  std::thread(
      [](decltype(Action) Action, decltype(CleanupTask)) {
        Action();
        // Make sure function stored by Action is destroyed before CleanupTask
        // is run.
        Action = nullptr;
      },
      std::move(Action), std::move(CleanupTask))
      .detach();
}

Deadline timeoutSeconds(llvm::Optional<double> Seconds) {
  using namespace std::chrono;
  if (!Seconds)
    return llvm::None;
  return steady_clock::now() +
         duration_cast<steady_clock::duration>(duration<double>(*Seconds));
}

} // namespace clangd
} // namespace clang
