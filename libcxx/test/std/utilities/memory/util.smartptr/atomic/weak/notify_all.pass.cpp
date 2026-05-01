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
void test_notify_all_weak() {
  using libcxx_atomic_smart_ptr_test::SpValues;
  std::atomic<std::weak_ptr<T>> a;

  ASSERT_NOEXCEPT(a.notify_all());

#if __cpp_lib_atomic_wait >= 201907L
  auto sp1             = SpValues<T>::state_a();
  std::weak_ptr<T> wp1 = sp1;
  a.store(std::weak_ptr<T>(wp1));

  std::atomic<int> phase{0};
  std::thread t1([&] {
    std::weak_ptr<T> old = a.load();
    phase.fetch_add(1, std::memory_order_acq_rel);
    while (phase.load(std::memory_order_acquire) < 2) {
      std::this_thread::yield();
    }
    a.wait(old);
  });
  std::thread t2([&] {
    std::weak_ptr<T> old = a.load();
    phase.fetch_add(1, std::memory_order_acq_rel);
    while (phase.load(std::memory_order_acquire) < 2) {
      std::this_thread::yield();
    }
    a.wait(old);
  });

  while (phase.load(std::memory_order_acquire) < 2) {
    std::this_thread::yield();
  }

  auto sp2 = SpValues<T>::state_b();
  a.store(std::weak_ptr<T>(sp2));
  a.notify_all();
  t1.join();
  t2.join();
#endif
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_W_NA(T) test_notify_all_weak<T>();
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_W_NA)
#undef LIBCXX_ATOMIC_SP_RUN_W_NA
  return 0;
}
