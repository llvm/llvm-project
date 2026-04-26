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
  std::atomic<std::weak_ptr<int>> a;

  auto sp1               = std::make_shared<int>(1);
  auto sp2               = std::make_shared<int>(2);
  std::weak_ptr<int> wp1 = sp1;
  std::weak_ptr<int> wp2 = sp2;

  a.store(wp1);
  {
    std::weak_ptr<int> got = a.load();
    auto locked            = got.lock();
    assert(locked && *locked == 1);
  }

  {
    std::weak_ptr<int> old = a.exchange(wp2);
    auto locked            = old.lock();
    assert(locked && *locked == 1);
    std::weak_ptr<int> got = a.load();
    auto locked2           = got.lock();
    assert(locked2 && *locked2 == 2);
  }

  {
    std::weak_ptr<int> expected = wp2;
    bool ok                     = a.compare_exchange_strong(expected, wp1);
    assert(ok);
    std::weak_ptr<int> got = a.load();
    auto locked            = got.lock();
    assert(locked && *locked == 1);
  }

  {
    std::weak_ptr<int> expected = wp2;
    bool ok                     = a.compare_exchange_strong(expected, wp2);
    assert(!ok);
    auto locked = expected.lock();
    assert(locked && *locked == 1);
  }

  // Expired weak pointer remains representable and loadable.
  {
    auto sp3               = std::make_shared<int>(3);
    std::weak_ptr<int> wp3 = sp3;
    a.store(wp3);
    sp3.reset();
    std::weak_ptr<int> got = a.load();
    assert(got.expired());
  }

#if __cpp_lib_atomic_wait >= 201907L
  {
    auto sp_for_wait               = std::make_shared<int>(42);
    std::weak_ptr<int> wp_for_wait = sp_for_wait;
    a.store(wp_for_wait);

    std::atomic<bool> started{false};
    std::thread t([&] {
      std::weak_ptr<int> old = a.load();
      started.store(true, std::memory_order_release);
      a.wait(old);
    });

    while (!started.load(std::memory_order_acquire)) {
    }

    a.store(wp1);
    a.notify_all();
    t.join();
  }
#endif

  return 0;
}
