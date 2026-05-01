//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void store(shared_ptr<T> desired, memory_order order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_store() {
  using libcxx_atomic_smart_ptr_test::SpValues;
  std::atomic<std::shared_ptr<T>> a;

  auto p = SpValues<T>::state_a();
  a.store(std::shared_ptr<T>(p));
  assert(a.load().get() == p.get());
  assert(*a.load() == *p);

  a.store(std::make_shared<T>(*SpValues<T>::state_b()));
  assert(*a.load() == *SpValues<T>::state_b());

  a.store(nullptr, std::memory_order_relaxed);
  assert(!a.load());

  ASSERT_NOEXCEPT(a.store(nullptr));
  {
    std::shared_ptr<T> desired = std::make_shared<T>(*SpValues<T>::state_c());
    ASSERT_NOEXCEPT(a.store(std::move(desired), std::memory_order_seq_cst));
  }
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_STORE(T) test_store<T>();
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_STORE)
#undef LIBCXX_ATOMIC_SP_RUN_STORE
  return 0;
}
