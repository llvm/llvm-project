//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// bool compare_exchange_strong(T&, T, memory_order, memory_order) const noexcept;
// bool compare_exchange_strong(T&, T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
struct TestCompareExchangeStrong {
  void operator()() const {
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      T t(T(1));
      std::same_as<bool> auto y = a.compare_exchange_strong(t, T(2));
      assert(y == true);
      assert(a == T(2));
      assert(t == T(1));
      y = a.compare_exchange_strong(t, T(3));
      assert(y == false);
      assert(a == T(2));
      assert(t == T(2));

      ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2)));
    }
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      T t(T(1));
      std::same_as<bool> auto y = a.compare_exchange_strong(t, T(2), std::memory_order_seq_cst);
      assert(y == true);
      assert(a == T(2));
      assert(t == T(1));
      y = a.compare_exchange_strong(t, T(3), std::memory_order_seq_cst);
      assert(y == false);
      assert(a == T(2));
      assert(t == T(2));

      ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2), std::memory_order_seq_cst));
    }
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      T t(T(1));
      std::same_as<bool> auto y =
          a.compare_exchange_strong(t, T(2), std::memory_order_release, std::memory_order_relaxed);
      assert(y == true);
      assert(a == T(2));
      assert(t == T(1));
      y = a.compare_exchange_strong(t, T(3), std::memory_order_release, std::memory_order_relaxed);
      assert(y == false);
      assert(a == T(2));
      assert(t == T(2));

      ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2), std::memory_order_release, std::memory_order_relaxed));
    }
  }
};

void test() { TestEachAtomicType<TestCompareExchangeStrong>()(); }

int main(int, char**) {
  test();
  return 0;
}
