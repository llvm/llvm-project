//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-threads

#include <atomic>
#include <cassert>
#include <memory>
#include <thread>

int main(int, char**) {
  std::atomic<std::shared_ptr<int>> a;

  auto p1 = std::make_shared<int>(1);
  auto p2 = std::make_shared<int>(2);

  a.store(p1);
  {
    auto got = a.load();
    assert(got && *got == 1);
  }

  {
    auto old = a.exchange(p2);
    assert(old && *old == 1);
    auto got = a.load();
    assert(got && *got == 2);
  }

  {
    auto expected = p2;
    bool ok       = a.compare_exchange_strong(expected, p1);
    assert(ok);
    auto got = a.load();
    assert(got && *got == 1);
  }

  {
    auto expected = p2;
    bool ok       = a.compare_exchange_strong(expected, p2);
    assert(!ok);
    assert(expected && *expected == 1);
  }

#if __cpp_lib_atomic_wait >= 201907L
  {
    std::atomic<bool> started{false};
    std::thread t([&] {
      auto old = a.load();
      started.store(true, std::memory_order_release);
      a.wait(old);
    });

    while (!started.load(std::memory_order_acquire)) {
    }

    a.store(p2);
    a.notify_all();
    t.join();
  }
#endif

  return 0;
}
