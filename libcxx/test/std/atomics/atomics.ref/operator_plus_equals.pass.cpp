//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// integral-type operator+=(integral-type) const noexcept;
// floating-point-type operator+=(floating-point-type) const noexcept;
// T* operator+=(difference_type) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
concept has_operator_plus_equals = requires { std::declval<T const>() += std::declval<T>(); };

template <typename T>
struct TestDoesNotHaveOperatorPlusEquals {
  void operator()() const { static_assert(!has_operator_plus_equals<std::atomic_ref<T>>); }
};

template <typename T>
struct TestOperatorPlusEquals {
  void operator()() const {
    if constexpr (std::is_arithmetic_v<T>) {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      std::same_as<T> decltype(auto) y = (a += T(2));
      assert(y == T(3));
      assert(x == T(3));
      ASSERT_NOEXCEPT(a += T(0));
    } else if constexpr (std::is_pointer_v<T>) {
      using U = std::remove_pointer_t<T>;
      U t[9]  = {};
      T p{&t[1]};
      std::atomic_ref<T> const a(p);

      std::same_as<T> decltype(auto) y = (a += 2);
      assert(y == &t[3]);
      assert(a == &t[3]);
      ASSERT_NOEXCEPT(a += 0);
    } else {
      static_assert(std::is_void_v<T>);
    }

    // memory_order::seq_cst
    {
      auto plus_equals = [](std::atomic_ref<T> const& x, T old_val, T new_val) { x += (new_val - old_val); };
      auto load        = [](std::atomic_ref<T> const& x) { return x.load(); };
      test_seq_cst<T>(plus_equals, load);
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestOperatorPlusEquals>()();

  TestOperatorPlusEquals<float>()();
  TestOperatorPlusEquals<double>()();

  TestEachPointerType<TestOperatorPlusEquals>()();

  TestDoesNotHaveOperatorPlusEquals<bool>()();
  TestDoesNotHaveOperatorPlusEquals<UserAtomicType>()();
  TestDoesNotHaveOperatorPlusEquals<LargeUserAtomicType>()();

  return 0;
}
