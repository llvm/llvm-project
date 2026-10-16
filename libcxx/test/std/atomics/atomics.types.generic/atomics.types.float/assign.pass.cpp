//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

//  floating-point-type operator=(floating-point-type) volatile noexcept;
//  floating-point-type operator=(floating-point-type) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "test_helper.h"
#include "test_macros.h"

template <class T>
concept HasVolatileAssign = requires(volatile std::atomic<T>& a, T t) { a = t; };

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  static_assert(HasVolatileAssign<T> == std::atomic<T>::is_always_lock_free);

  static_assert(noexcept(std::declval<MaybeVolatile<std::atomic<T>>&>() = (T(0))));

  // assignment
  {
    MaybeVolatile<std::atomic<T>> a(T(3.1));
    std::same_as<T> decltype(auto) r = (a = T(1.2));
    assert(a.load() == T(1.2));
    assert(r == T(1.2));
  }

  // memory_order::seq_cst
  {
    auto assign = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x = new_val; };
    auto load   = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(); };
    test_seq_cst<T, MaybeVolatile>(assign, load);
  }
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
  // TODO https://llvm.org/PR48634
  // test<long double>();

  return 0;
}
