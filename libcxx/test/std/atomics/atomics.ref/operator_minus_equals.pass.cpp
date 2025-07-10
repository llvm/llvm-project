//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// integral-type operator-=(integral-type) const noexcept;
// floating-point-type operator-=(floating-point-type) const noexcept;
// T* operator-=(difference_type) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
concept has_operator_minus_equals = requires { std::declval<T const>() -= std::declval<T>(); };

template <typename T>
struct TestDoesNotHaveOperatorMinusEquals {
  void operator()() const { static_assert(!has_operator_minus_equals<std::atomic_ref<T>>); }
};

template <typename T>
struct TestOperatorMinusEquals {
  void operator()() const {
    if constexpr (std::is_arithmetic_v<T>) {
      T x(T(3));
      std::atomic_ref<T> const a(x);

      std::same_as<T> decltype(auto) y = (a -= T(2));
      assert(y == T(1));
      assert(x == T(1));
      ASSERT_NOEXCEPT(a -= T(0));
    } else if constexpr (std::is_pointer_v<T>) {
      using U = std::remove_pointer_t<T>;
      U t[9]  = {};
      T p{&t[3]};
      std::atomic_ref<T> const a(p);

      std::same_as<T> decltype(auto) y = (a -= 2);
      assert(y == &t[1]);
      assert(a == &t[1]);
      ASSERT_NOEXCEPT(a -= 0);
    } else {
      static_assert(std::is_void_v<T>);
    }

    // memory_order::seq_cst
    {
      auto minus_equals = [](std::atomic_ref<T> const& x, T old_val, T new_val) { x -= (old_val - new_val); };
      auto load         = [](std::atomic_ref<T> const& x) { return x.load(); };
      test_seq_cst<T>(minus_equals, load);
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestOperatorMinusEquals>()();

  TestOperatorMinusEquals<float>()();
  TestOperatorMinusEquals<double>()();

  TestEachPointerType<TestOperatorMinusEquals>()();

  TestDoesNotHaveOperatorMinusEquals<bool>()();
  TestDoesNotHaveOperatorMinusEquals<UserAtomicType>()();
  TestDoesNotHaveOperatorMinusEquals<LargeUserAtomicType>()();

  return 0;
}
