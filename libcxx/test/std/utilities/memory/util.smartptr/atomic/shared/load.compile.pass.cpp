//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// shared_ptr<T> load(memory_order order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void check(const std::atomic<std::shared_ptr<T>>& asp) {
  std::same_as<std::shared_ptr<T>> decltype(auto) no_arg = asp.load();
  ASSERT_SAME_TYPE(decltype(asp.load()), std::shared_ptr<T>);
  ASSERT_NOEXCEPT(asp.load());

  std::same_as<std::shared_ptr<T>> decltype(auto) with_order = asp.load(std::memory_order_seq_cst);
  ASSERT_SAME_TYPE(decltype(asp.load(std::memory_order_acquire)), std::shared_ptr<T>);
  ASSERT_NOEXCEPT(asp.load(std::memory_order_seq_cst));
  static_cast<void>(no_arg);
  static_cast<void>(with_order);

  {
    const std::atomic<std::shared_ptr<T>> const_a;
    static_assert(noexcept(const_a.load()));
    ASSERT_SAME_TYPE(decltype(const_a.load()), std::shared_ptr<T>);
  }
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_LOAD(T) check<T>(std::atomic<std::shared_ptr<T>>());
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_LOAD)
#undef LIBCXX_ATOMIC_SP_RUN_LOAD
  return 0;
}
