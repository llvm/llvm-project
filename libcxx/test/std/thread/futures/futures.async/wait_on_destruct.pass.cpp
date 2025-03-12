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

// Make sure that the `future` destructor keeps the data alive until the thread finished. This test fails by triggering
// TSan. It may not be observable by normal means.

#include <atomic>
#include <future>
#include <mutex>

std::mutex mux;

int main() {
  using namespace std::chrono_literals;
  mux.lock();
  std::atomic<bool> in_async = false;
  auto v = std::async(std::launch::async, [&in_async, value = 1]() mutable {
    in_async = true;
    in_async.notify_all();
    std::scoped_lock thread_lock(mux);
    value = 4;
    (void)value;
  });
  in_async.wait(true);
  mux.unlock();
}
