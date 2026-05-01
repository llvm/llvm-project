//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void store(weak_ptr<T> desired, memory_order order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_store_weak() {
  using libcxx_atomic_smart_ptr_test::SpValues;
  auto sp1             = SpValues<T>::state_a();
  auto sp2             = SpValues<T>::state_b();
  std::weak_ptr<T> wp1 = sp1;
  std::weak_ptr<T> wp2 = sp2;

  std::atomic<std::weak_ptr<T>> a;
  a.store(std::weak_ptr<T>(wp1));
  {
    auto locked = a.load().lock();
    assert(locked && *locked == *sp1);
  }

  a.store(std::weak_ptr<T>(wp2), std::memory_order_relaxed);
  {
    auto locked = a.load().lock();
    assert(locked && *locked == *sp2);
  }

  ASSERT_NOEXCEPT(a.store(std::weak_ptr<T>(wp1)));
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_W_STORE(T) test_store_weak<T>();
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_W_STORE)
#undef LIBCXX_ATOMIC_SP_RUN_W_STORE
  return 0;
}
