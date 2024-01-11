//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// UNSUPPORTED: !non-lockfree-atomics

//  operator floating-point-type() volatile noexcept;
//  operator floating-point-type() noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_helper.h"
#include "test_macros.h"

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  // Uncomment the test after P1831R1 is implemented
  // static_assert(std::is_convertible_v<volatile std::atomic<T>&, T> == std::atomic<T>::is_always_lock_free);
  static_assert(noexcept(T(std::declval<MaybeVolatile<std::atomic<T>>&>())));

  // operator float
  {
    MaybeVolatile<std::atomic<T>> a(T(3.1));
    T r = a;
    assert(r == T(3.1));
  }

  // memory_order::seq_cst
  {
    auto store    = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val); };
    auto op_float = [](MaybeVolatile<std::atomic<T>>& x) -> T { return x; };
    test_seq_cst<T, MaybeVolatile>(store, op_float);
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
  // TODO https://github.com/llvm/llvm-project/issues/47978
  // test<long double>();

  return 0;
}
