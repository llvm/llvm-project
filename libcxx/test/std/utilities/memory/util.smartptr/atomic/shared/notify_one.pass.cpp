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
#include <memory>
#include <thread>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_notify_one() {
  using libcxx_atomic_smart_ptr_test::SpValues;
  std::atomic<std::shared_ptr<T>> a;

  ASSERT_NOEXCEPT(a.notify_one());

#if __cpp_lib_atomic_wait >= 201907L
  auto p1 = SpValues<T>::state_a();
  a.store(std::shared_ptr<T>(p1));

  std::atomic<bool> started{false};
  std::thread t([&] {
    std::shared_ptr<T> old = a.load();
    started.store(true, std::memory_order_release);
    a.wait(old);
  });

  while (!started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  a.store(std::make_shared<T>(*SpValues<T>::state_b()));
  a.notify_one();
  t.join();
#endif
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_NOTIFY_ONE(T) test_notify_one<T>();
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_NOTIFY_ONE)
#undef LIBCXX_ATOMIC_SP_RUN_NOTIFY_ONE
  return 0;
}
