//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void notify_one() noexcept;

#include <atomic>
#include <cassert>
#include <memory>
#include <thread>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_notify_one_weak() {
  std::atomic<std::weak_ptr<T>> a;

  static_assert(noexcept(a.notify_one()));

#if __cpp_lib_atomic_wait >= 201907L
  auto sp1             = SpValues<T>::state_a();
  std::weak_ptr<T> wp1 = sp1;
  a.store(std::weak_ptr<T>(wp1));

  std::atomic<bool> started{false};
  std::thread t([&] {
    std::weak_ptr<T> old = a.load();
    started.store(true, std::memory_order_release);
    a.wait(old);
  });

  while (!started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  auto spB = SpValues<T>::state_b();
  a.store(std::weak_ptr<T>(spB));
  a.notify_one();
  t.join();
#endif
}

template <class T>
struct TestNotifyOneWeak {
  void operator()() const { test_notify_one_weak<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestNotifyOneWeak>();
  return 0;
}
