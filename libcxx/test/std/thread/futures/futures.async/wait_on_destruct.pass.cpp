//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// This test uses std::atomic interfaces that are only available in C++20
// UNSUPPORTED: c++11, c++14, c++17

// Make sure that the `future` destructor keeps the data alive until the thread finished. This test fails by triggering
// TSan. It may not be observable by normal means.

// See https://github.com/llvm/llvm-project/pull/125433#issuecomment-2703618927 for more details.

#include <atomic>
#include <future>
#include <mutex>
#include <condition_variable>
#include <thread>

std::mutex mux;

int main(int, char**) {
  std::condition_variable cond;
  std::unique_lock lock(mux);
  auto v = std::async(std::launch::async, [&cond, value = 1]() mutable {
    std::unique_lock thread_lock(mux);
    cond.notify_all();
    thread_lock.unlock();

    value = 4;
    (void)value;
  });
  cond.wait(lock);

  return 0;
}
