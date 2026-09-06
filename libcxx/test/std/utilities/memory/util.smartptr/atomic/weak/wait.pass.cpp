//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void wait(weak_ptr<T> old, memory_order order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <memory>
#include <thread>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_wait_weak() {
  std::atomic<std::weak_ptr<T>> a;

#if __cpp_lib_atomic_wait >= 201907L
  auto sp_for_wait             = SpValues<T>::state_a();
  std::weak_ptr<T> wp_for_wait = sp_for_wait;
  a.store(std::weak_ptr<T>(wp_for_wait));

  auto sp1             = SpValues<T>::state_b();
  std::weak_ptr<T> wp1 = sp1;

  std::atomic<bool> started{false};
  std::thread t([&] {
    std::weak_ptr<T> old = a.load();
    started.store(true, std::memory_order_release);
    a.wait(old);
  });

  while (!started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  a.store(std::weak_ptr<T>(wp1));
  a.notify_all();
  t.join();

  {
    auto locked = a.load().lock();
    assert(locked && *locked == *sp1);
  }
#endif

  static_assert(noexcept(a.wait(std::weak_ptr<T>(), std::memory_order_seq_cst)));
}

template <class T>
struct TestWaitWeak {
  void operator()() const { test_wait_weak<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestWaitWeak>();
  return 0;
}
