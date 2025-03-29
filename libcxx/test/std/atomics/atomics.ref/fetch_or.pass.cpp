//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// integral-type fetch_or(integral-type, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
concept has_fetch_or = requires {
  std::declval<T const>().fetch_or(std::declval<T>());
  std::declval<T const>().fetch_or(std::declval<T>(), std::declval<std::memory_order>());
};

template <typename T>
struct TestDoesNotHaveFetchOr {
  void operator()() const { static_assert(!has_fetch_or<std::atomic_ref<T>>); }
};

template <typename T>
struct TestFetchOr {
  void operator()() const {
    static_assert(std::is_integral_v<T>);

    T x(T(1));
    std::atomic_ref<T> const a(x);

    {
      std::same_as<T> decltype(auto) y = a.fetch_or(T(2));
      assert(y == T(1));
      assert(x == T(3));
      ASSERT_NOEXCEPT(a.fetch_or(T(0)));
    }

    {
      std::same_as<T> decltype(auto) y = a.fetch_or(T(2), std::memory_order_relaxed);
      assert(y == T(3));
      assert(x == T(3));
      ASSERT_NOEXCEPT(a.fetch_or(T(0), std::memory_order_relaxed));
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFetchOr>()();

  TestEachFloatingPointType<TestDoesNotHaveFetchOr>()();

  TestEachPointerType<TestDoesNotHaveFetchOr>()();

  TestDoesNotHaveFetchOr<bool>()();
  TestDoesNotHaveFetchOr<UserAtomicType>()();
  TestDoesNotHaveFetchOr<LargeUserAtomicType>()();

  return 0;
}
