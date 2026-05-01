//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void wait(shared_ptr<T> old, memory_order order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <memory>
#include <thread>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_wait() {
  using libcxx_atomic_smart_ptr_test::SpValues;
  std::atomic<std::shared_ptr<T>> a;

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

  auto pWake = std::make_shared<T>(*SpValues<T>::state_c());
  a.store(std::move(pWake));
  a.notify_all();
  t.join();

  assert(*a.load() == *SpValues<T>::state_c());
#endif

  ASSERT_NOEXCEPT(a.wait(std::shared_ptr<T>(), std::memory_order_seq_cst));
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_WAIT(T) test_wait<T>();
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_WAIT)
#undef LIBCXX_ATOMIC_SP_RUN_WAIT
  return 0;
}
