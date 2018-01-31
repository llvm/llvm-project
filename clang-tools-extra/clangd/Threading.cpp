#include "Threading.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"

namespace clang {
namespace clangd {
ThreadPool::ThreadPool(unsigned AsyncThreadsCount)
    : RunSynchronously(AsyncThreadsCount == 0) {
  if (RunSynchronously) {
    // Don't start the worker thread if we're running synchronously
    return;
  }

  Workers.reserve(AsyncThreadsCount);
  for (unsigned I = 0; I < AsyncThreadsCount; ++I) {
    Workers.push_back(std::thread([this, I]() {
      llvm::set_thread_name(llvm::formatv("scheduler/{0}", I));
      while (true) {
        UniqueFunction<void()> Request;
        Context Ctx;

        // Pick request from the queue
        {
          std::unique_lock<std::mutex> Lock(Mutex);
          // Wait for more requests.
          RequestCV.wait(Lock,
                         [this] { return !RequestQueue.empty() || Done; });
          if (RequestQueue.empty()) {
            assert(Done);
            return;
          }

          // We process requests starting from the front of the queue. Users of
          // ThreadPool have a way to prioritise their requests by putting
          // them to the either side of the queue (using either addToEnd or
          // addToFront).
          std::tie(Request, Ctx) = std::move(RequestQueue.front());
          RequestQueue.pop_front();
        } // unlock Mutex

        WithContext WithCtx(std::move(Ctx));
        Request();
      }
    }));
  }
}

ThreadPool::~ThreadPool() {
  if (RunSynchronously)
    return; // no worker thread is running in that case

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    // Wake up the worker thread
    Done = true;
  } // unlock Mutex
  RequestCV.notify_all();

  for (auto &Worker : Workers)
    Worker.join();
}
} // namespace clangd
} // namespace clang
