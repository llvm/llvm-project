//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// XFAIL: !has-64-bit-atomics

// integral-type fetch_max(integral-type, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
concept has_fetch_max = requires {
  std::declval<T const>().fetch_max(std::declval<T>());
  std::declval<T const>().fetch_max(std::declval<T>(), std::declval<std::memory_order>());
};

template <typename T>
struct TestDoesNotHaveFetchMax {
  void operator()() const { static_assert(!has_fetch_max<std::atomic_ref<T>>); }
};

template <typename T>
struct TestFetchMax {
  void operator()() const {
    static_assert(std::is_integral_v<T>);

    alignas(std::atomic_ref<T>::required_alignment) T x(T(1));
    std::atomic_ref<T> const a(x);

    {
      std::same_as<T> decltype(auto) y = a.fetch_max(T(2));
      assert(y == T(1));
      assert(x == T(2));
      ASSERT_NOEXCEPT(a.fetch_max(T(0)));
    }

    {
      std::same_as<T> decltype(auto) y = a.fetch_max(T(1), std::memory_order_relaxed);
      assert(y == T(2));
      assert(x == T(2));
      ASSERT_NOEXCEPT(a.fetch_max(T(0), std::memory_order_relaxed));
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFetchMax>()();

  TestEachFloatingPointType<TestDoesNotHaveFetchMax>()();

  TestEachPointerType<TestDoesNotHaveFetchMax>()();

  TestDoesNotHaveFetchMax<bool>()();
  TestDoesNotHaveFetchMax<UserAtomicType>()();
  TestDoesNotHaveFetchMax<LargeUserAtomicType>()();

  return 0;
}
