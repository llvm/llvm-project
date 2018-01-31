//===--- ThreadPool.h --------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_THREADING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_THREADING_H

#include "Context.h"
#include "Function.h"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace clang {
namespace clangd {
/// A simple fixed-size thread pool implementation.
class ThreadPool {
public:
  /// If \p AsyncThreadsCount is 0, requests added using addToFront and addToEnd
  /// will be processed synchronously on the calling thread.
  // Otherwise, \p AsyncThreadsCount threads will be created to schedule the
  // requests.
  ThreadPool(unsigned AsyncThreadsCount);
  /// Destructor blocks until all requests are processed and worker threads are
  /// terminated.
  ~ThreadPool();

  /// Add a new request to run function \p F with args \p As to the start of the
  /// queue. The request will be run on a separate thread.
  template <class Func, class... Args>
  void addToFront(Func &&F, Args &&... As) {
    if (RunSynchronously) {
      std::forward<Func>(F)(std::forward<Args>(As)...);
      return;
    }

    {
      std::lock_guard<std::mutex> Lock(Mutex);
      RequestQueue.emplace_front(
          BindWithForward(std::forward<Func>(F), std::forward<Args>(As)...),
          Context::current().clone());
    }
    RequestCV.notify_one();
  }

  /// Add a new request to run function \p F with args \p As to the end of the
  /// queue. The request will be run on a separate thread.
  template <class Func, class... Args> void addToEnd(Func &&F, Args &&... As) {
    if (RunSynchronously) {
      std::forward<Func>(F)(std::forward<Args>(As)...);
      return;
    }

    {
      std::lock_guard<std::mutex> Lock(Mutex);
      RequestQueue.emplace_back(
          BindWithForward(std::forward<Func>(F), std::forward<Args>(As)...),
          Context::current().clone());
    }
    RequestCV.notify_one();
  }

private:
  bool RunSynchronously;
  mutable std::mutex Mutex;
  /// We run some tasks on separate threads(parsing, CppFile cleanup).
  /// These threads looks into RequestQueue to find requests to handle and
  /// terminate when Done is set to true.
  std::vector<std::thread> Workers;
  /// Setting Done to true will make the worker threads terminate.
  bool Done = false;
  /// A queue of requests.
  std::deque<std::pair<UniqueFunction<void()>, Context>> RequestQueue;
  /// Condition variable to wake up worker threads.
  std::condition_variable RequestCV;
};
} // namespace clangd
} // namespace clang
#endif
