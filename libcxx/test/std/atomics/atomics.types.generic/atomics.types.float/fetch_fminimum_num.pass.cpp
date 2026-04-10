//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// XFAIL: !has-64-bit-atomics

// floating-point-type fetch_fminimum_num(floating-point-type, memory_order = memory_order::seq_cst) volatile noexcept;
// floating-point-type fetch_fminimum_num(floating-point-type, memory_order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "../../atomics.types.operations/atomics.types.operations.req/atomic_fetch_fminimum_num_helper.h"

template <class T>
concept HasVolatileFetchFMinimumNum = requires(volatile std::atomic<T>& a, T t) { a.fetch_fminimum_num(t); };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  static_assert(HasVolatileFetchFMinimumNum<T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>().fetch_fminimum_num(T(0))));

  MaybeVolatile<std::atomic<T>> a;

  auto load         = [&]() { return a.load(); };
  auto store        = [&](T val) { a.store(val); };
  auto fminimum_num = [&](T val, auto order) { return a.fetch_fminimum_num(val, order); };

  test_fetch_fminimum_num<T>(load, store, fminimum_num);
}

template <class T>
void test() {
  test_impl<T>();
  if constexpr (std::atomic<T>::is_always_lock_free) {
    test_impl<T, std::add_volatile_t>();
  }
}

int main(int, char**) {
  test<float>();
  test<double>();

  return 0;
}
