//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// integral-type operator++(int) const noexcept;
// integral-type operator--(int) const noexcept;
// integral-type operator++() const noexcept;
// integral-type operator--() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
concept has_pre_increment_operator = requires { ++std::declval<T const>(); };

template <typename T>
concept has_post_increment_operator = requires { std::declval<T const>()++; };

template <typename T>
concept has_pre_decrement_operator = requires { --std::declval<T const>(); };

template <typename T>
concept has_post_decrement_operator = requires { std::declval<T const>()--; };

template <typename T>
constexpr bool does_not_have_increment_nor_decrement_operators() {
  return !has_pre_increment_operator<T> && !has_pre_decrement_operator<T> && !has_post_increment_operator<T> &&
         !has_post_decrement_operator<T>;
}

template <typename T>
struct TestDoesNotHaveIncrementDecrement {
  void operator()() const { static_assert(does_not_have_increment_nor_decrement_operators<T>()); }
};

template <typename T>
struct TestIncrementDecrement {
  void operator()() const {
    static_assert(std::is_integral_v<T>);

    T x(T(1));
    std::atomic_ref<T> const a(x);

    {
      std::same_as<T> decltype(auto) y = ++a;
      assert(y == T(2));
      assert(x == T(2));
      ASSERT_NOEXCEPT(++a);
    }

    {
      std::same_as<T> decltype(auto) y = --a;
      assert(y == T(1));
      assert(x == T(1));
      ASSERT_NOEXCEPT(--a);
    }

    {
      std::same_as<T> decltype(auto) y = a++;
      assert(y == T(1));
      assert(x == T(2));
      ASSERT_NOEXCEPT(a++);
    }

    {
      std::same_as<T> decltype(auto) y = a--;
      assert(y == T(2));
      assert(x == T(1));
      ASSERT_NOEXCEPT(a--);
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestIncrementDecrement>()();

  TestEachFloatingPointType<TestDoesNotHaveIncrementDecrement>()();

  TestEachPointerType<TestDoesNotHaveIncrementDecrement>()();

  TestDoesNotHaveIncrementDecrement<bool>()();
  TestDoesNotHaveIncrementDecrement<UserAtomicType>()();
  TestDoesNotHaveIncrementDecrement<LargeUserAtomicType>()();

  return 0;
}
