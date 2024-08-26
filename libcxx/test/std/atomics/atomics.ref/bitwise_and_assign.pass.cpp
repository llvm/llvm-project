//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// integral-type operator&=(integral-type) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
concept has_bitwise_and_assign = requires { std::declval<T const>() &= std::declval<T>(); };

template <typename T>
struct TestDoesNotHaveBitwiseAndAssign {
  void operator()() const { static_assert(!has_bitwise_and_assign<std::atomic_ref<T>>); }
};

template <typename T>
struct TestBitwiseAndAssign {
  void operator()() const {
    static_assert(std::is_integral_v<T>);

    T x(T(1));
    std::atomic_ref<T> const a(x);

    std::same_as<T> decltype(auto) y = (a &= T(1));
    assert(y == T(1));
    assert(x == T(1));
    ASSERT_NOEXCEPT(a &= T(0));

    y = (a &= T(2));
    assert(y == T(0));
    assert(x == T(0));
  }
};

int main(int, char**) {
  TestEachIntegralType<TestBitwiseAndAssign>()();

  TestEachFloatingPointType<TestDoesNotHaveBitwiseAndAssign>()();

  TestEachPointerType<TestDoesNotHaveBitwiseAndAssign>()();

  TestDoesNotHaveBitwiseAndAssign<bool>()();
  TestDoesNotHaveBitwiseAndAssign<UserAtomicType>()();
  TestDoesNotHaveBitwiseAndAssign<LargeUserAtomicType>()();

  return 0;
}
