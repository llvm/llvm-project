//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void notify_all() noexcept;

#include <atomic>
#include <cassert>
#include <memory>
#include <thread>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_notify_all() {
  std::atomic<std::shared_ptr<T>> a;

  static_assert(noexcept(a.notify_all()));

#if __cpp_lib_atomic_wait >= 201907L
  auto p1 = SpValues<T>::state_a();
  a.store(std::shared_ptr<T>(p1));

  std::atomic<int> phase{0};
  std::thread t1([&] {
    std::shared_ptr<T> old = a.load();
    phase.fetch_add(1, std::memory_order_acq_rel);
    while (phase.load(std::memory_order_acquire) < 2) {
      std::this_thread::yield();
    }
    a.wait(old);
  });
  std::thread t2([&] {
    std::shared_ptr<T> old = a.load();
    phase.fetch_add(1, std::memory_order_acq_rel);
    while (phase.load(std::memory_order_acquire) < 2) {
      std::this_thread::yield();
    }
    a.wait(old);
  });

  while (phase.load(std::memory_order_acquire) < 2) {
    std::this_thread::yield();
  }

  a.store(std::make_shared<T>(*SpValues<T>::state_b()));
  a.notify_all();
  t1.join();
  t2.join();
#endif
}

template <class T>
struct TestNotifyAll {
  void operator()() const { test_notify_all<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestNotifyAll>();
  return 0;
}
