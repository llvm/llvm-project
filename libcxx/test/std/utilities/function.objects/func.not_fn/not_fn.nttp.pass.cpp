//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <functional>

// template<auto f> constexpr unspecified not_fn() noexcept;

#include <functional>

#include <bit>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "test_macros.h"

class BooleanTestable {
  bool val_;

public:
  constexpr explicit BooleanTestable(bool val) : val_(val) {}
  constexpr operator bool() const { return val_; }
  constexpr BooleanTestable operator!() const { return BooleanTestable{!val_}; }
};

LIBCPP_STATIC_ASSERT(std::__boolean_testable<BooleanTestable>);

class FakeBool {
  int val_;

public:
  constexpr FakeBool(int val) : val_(val) {}
  constexpr FakeBool operator!() const { return FakeBool{-val_}; }
  constexpr bool operator==(int other) const { return val_ == other; }
};

template <bool IsNoexcept>
struct MaybeNoexceptFn {
  bool operator()() const noexcept(IsNoexcept); // not defined
};

constexpr void basic_tests() {
  { // Test constant functions
    auto false_fn = std::not_fn<std::false_type{}>();
    assert(false_fn());

    auto true_fn = std::not_fn<std::true_type{}>();
    assert(!true_fn());

    static_assert(noexcept(std::not_fn<std::false_type{}>()));
    static_assert(noexcept(std::not_fn<std::true_type{}>()));
  }

  { // Test function with one argument
    auto is_odd = std::not_fn<[](auto x) { return x % 2 == 0; }>();
    assert(is_odd(1));
    assert(!is_odd(2));
    assert(is_odd(3));
    assert(!is_odd(4));
    assert(is_odd(5));
  }

  { // Test function with multiple arguments
    auto at_least_10 = [](auto... vals) { return (vals + ... + 0) >= 10; };
    auto at_most_9   = std::not_fn<at_least_10>();
    assert(at_most_9());
    assert(at_most_9(1));
    assert(at_most_9(1, 2, 3, 4, -1));
    assert(at_most_9(3, 3, 2, 1, -2));
    assert(!at_most_9(10, -1, 2));
    assert(!at_most_9(5, 5));
    static_assert(noexcept(std::not_fn<at_least_10>()));
  }

  { // Test function that returns boolean-testable type other than bool
    auto is_product_even = [](auto... vals) { return BooleanTestable{(vals * ... * 1) % 2 == 0}; };
    auto is_product_odd  = std::not_fn<is_product_even>();
    assert(is_product_odd());
    assert(is_product_odd(1, 3, 5, 9));
    assert(is_product_odd(3, 3, 3, 3));
    assert(!is_product_odd(3, 5, 9, 11, 0));
    assert(!is_product_odd(11, 7, 5, 3, 2));
    static_assert(noexcept(std::not_fn<is_product_even>()));
  }

  { // Test function that returns non-boolean-testable type
    auto sum         = [](auto... vals) -> FakeBool { return (vals + ... + 0); };
    auto negated_sum = std::not_fn<sum>();
    assert(negated_sum() == 0);
    assert(negated_sum(3) == -3);
    assert(negated_sum(4, 5, 1, 3) == -13);
    assert(negated_sum(4, 2, 5, 6, 1) == -18);
    assert(negated_sum(-1, 3, 2, -8) == 4);
    static_assert(noexcept(std::not_fn<sum>()));
  }

  { // Test member pointers
    struct MemberPointerTester {
      bool value = true;
      constexpr bool not_value() const { return !value; }
      constexpr bool value_and(bool other) noexcept { return value && other; }
    };

    MemberPointerTester tester;

    auto not_mem_object = std::not_fn<&MemberPointerTester::value>();
    assert(!not_mem_object(tester));
    assert(!not_mem_object(std::as_const(tester)));
    static_assert(noexcept(not_mem_object(tester)));
    static_assert(noexcept(not_mem_object(std::as_const(tester))));

    auto not_nullary_mem_fn = std::not_fn<&MemberPointerTester::not_value>();
    assert(not_nullary_mem_fn(tester));
    static_assert(!noexcept(not_nullary_mem_fn(tester)));

    auto not_unary_mem_fn = std::not_fn<&MemberPointerTester::value_and>();
    assert(not_unary_mem_fn(tester, false));
    static_assert(noexcept(not_unary_mem_fn(tester, false)));
    static_assert(!std::is_invocable_v<decltype(not_unary_mem_fn), const MemberPointerTester&, bool>);
  }
}

constexpr void test_perfect_forwarding_call_wrapper() {
  { // Make sure we call the correctly cv-ref qualified operator()
    // based on the value category of the not_fn<NTTP> unspecified-type.
    struct X {
      constexpr FakeBool operator()() & { return 1; }
      constexpr FakeBool operator()() const& { return 2; }
      constexpr FakeBool operator()() && { return 3; }
      constexpr FakeBool operator()() const&& { return 4; }
    };

    auto f  = std::not_fn<X{}>();
    using F = decltype(f);
    assert(static_cast<F&>(f)() == -2);
    assert(static_cast<const F&>(f)() == -2);
    assert(static_cast<F&&>(f)() == -2);
    assert(static_cast<const F&&>(f)() == -2);
  }

  // Call to `not_fn<NTTP>` unspecified-type's operator() should always result in call to const& overload of .
  {
    { // Make sure unspecified-type is still callable when we delete & overload.
      struct X {
        FakeBool operator()() & = delete;
        FakeBool operator()() const&;
        FakeBool operator()() &&;
        FakeBool operator()() const&&;
      };

      using F = decltype(std::not_fn<X{}>());
      static_assert(std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(std::invocable<F>);
      static_assert(std::invocable<const F>);
    }

    { // Make sure unspecified-type is not callable when we delete const& overload.
      struct X {
        FakeBool operator()() &;
        FakeBool operator()() const& = delete;
        FakeBool operator()() &&;
        FakeBool operator()() const&&;
      };

      using F = decltype(std::not_fn<X{}>());
      static_assert(!std::invocable<F&>);
      static_assert(!std::invocable<const F&>);
      static_assert(!std::invocable<F>);
      static_assert(!std::invocable<const F>);
    }

    { // Make sure unspecified-type is still callable when we delete && overload.
      struct X {
        FakeBool operator()() &;
        FakeBool operator()() const&;
        FakeBool operator()() && = delete;
        FakeBool operator()() const&&;
      };

      using F = decltype(std::not_fn<X{}>());
      static_assert(std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(std::invocable<F>);
      static_assert(std::invocable<const F>);
    }

    { // Make sure unspecified-type is still callable when we delete const&& overload.
      struct X {
        FakeBool operator()() &;
        FakeBool operator()() const&;
        FakeBool operator()() &&;
        FakeBool operator()() const&& = delete;
      };

      using F = decltype(std::not_fn<X{}>());
      static_assert(std::invocable<F&>);
      static_assert(std::invocable<const F&>);
      static_assert(std::invocable<F>);
      static_assert(std::invocable<const F>);
    }
  }

  { // Test perfect forwarding
    auto f = [](int& val) {
      val = 5;
      return false;
    };

    auto not_f = std::not_fn<f>();
    int val    = 0;
    assert(not_f(val));
    assert(val == 5);

    using NotF = decltype(not_f);
    static_assert(std::invocable<NotF, int&>);
    static_assert(!std::invocable<NotF, int>);
  }
}

constexpr void test_return_type() {
  { // Test constructors and assignment operators
    struct IsPowerOfTwo {
      constexpr bool operator()(unsigned int x) const { return std::has_single_bit(x); }
    };

    auto is_not_power_of_2 = std::not_fn<IsPowerOfTwo{}>();
    assert(is_not_power_of_2(5));
    assert(!is_not_power_of_2(4));

    auto moved = std::move(is_not_power_of_2);
    assert(moved(5));
    assert(!moved(4));

    auto copied = is_not_power_of_2;
    assert(copied(7));
    assert(!copied(8));

    moved = std::move(copied);
    assert(copied(9));
    assert(!copied(16));

    copied = moved;
    assert(copied(11));
    assert(!copied(32));
  }

  { // Make sure `not_fn<NTTP>` unspecified type's operator() is SFINAE-friendly.
    using F = decltype(std::not_fn<[](int x) { return !x; }>());
    static_assert(!std::is_invocable<F>::value);
    static_assert(std::is_invocable<F, int>::value);
    static_assert(!std::is_invocable<F, void*>::value);
    static_assert(!std::is_invocable<F, int, int>::value);
  }

  { // Test noexceptness
    auto always_noexcept = std::not_fn<MaybeNoexceptFn<true>{}>();
    static_assert(noexcept(always_noexcept()));

    auto never_noexcept = std::not_fn<MaybeNoexceptFn<false>{}>();
    static_assert(!noexcept(never_noexcept()));
  }

  { // Test calling volatile wrapper
    using NotFn = decltype(std::not_fn<std::false_type{}>());
    static_assert(!std::invocable<volatile NotFn&>);
    static_assert(!std::invocable<const volatile NotFn&>);
    static_assert(!std::invocable<volatile NotFn>);
    static_assert(!std::invocable<const volatile NotFn>);
  }
}

constexpr bool test() {
  basic_tests();
  test_perfect_forwarding_call_wrapper();
  test_return_type();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
