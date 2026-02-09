//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// XFAIL: !has-64-bit-atomics

// <atomic>

// template<class T>
//   T atomic_fetch_max_explicit(volatile atomic<T>*, typename atomic<T>::value_type,
//                               memory_order) noexcept;
// template<class T>
//   constexpr T atomic_fetch_max_explicit(atomic<T>*, typename atomic<T>::value_type,
//                                         memory_order) noexcept;

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    {
      std::atomic<T> t(T(1));
      assert(std::atomic_fetch_max_explicit(&t, T(2), std::memory_order_seq_cst) == T(1));
      assert(t == T(2));

      ASSERT_NOEXCEPT(std::atomic_fetch_max_explicit(&t, T(2), std::memory_order_seq_cst));
    }
    {
      volatile std::atomic<T> t(T(3));
      assert(std::atomic_fetch_max_explicit(&t, T(2), std::memory_order_seq_cst) == T(3));
      assert(t == T(3));

      ASSERT_NOEXCEPT(std::atomic_fetch_max_explicit(&t, T(2), std::memory_order_seq_cst));
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFn>()();

  return 0;
}
