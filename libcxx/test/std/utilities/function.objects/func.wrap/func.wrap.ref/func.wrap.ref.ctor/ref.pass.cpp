//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<class F> constexpr function_ref(F&&) noexcept;

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

// Constraints:
// - remove_cvref_t<F> is not the same type as function_ref,
// - is_member_pointer_v<T> is false, and
// - is-invocable-using<cv T&> is true.

struct A {
  int i;
  void f() {}
  void operator()(auto...) const {}
};

constexpr auto l1 = [] {};
using L1          = std::remove_cvref_t<decltype(l1)>;

constexpr auto l2 = [] { return A{5}; };
using L2          = std::remove_cvref_t<decltype(l2)>;

constexpr auto l2_noexcept = [] noexcept { return A{5}; };
using L2Noexcept           = std::remove_cvref_t<decltype(l2_noexcept)>;

constexpr auto l3 = [](int x, double d) { return x + d; };
using L3          = std::remove_cvref_t<decltype(l3)>;

constexpr auto l3_noexcept = [](int x, double d) noexcept { return x + d; };
using L3Noexcept           = std::remove_cvref_t<decltype(l3_noexcept)>;

struct NonConstInvocable {
  int i;

  int operator()(int x, double y) {
    ++i;
    return x + y + i;
  }
};

// non-const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void()>, L1&>);
static_assert(std::is_constructible_v<std::function_ref<void()>, L1 const&>);
static_assert(std::is_constructible_v<std::function_ref<void()>, L1&&>);
static_assert(std::is_constructible_v<std::function_ref<void()>, L1 const&&>);
static_assert(std::is_constructible_v<std::function_ref<void()>, L2&>);
static_assert(std::is_constructible_v<std::function_ref<void()>, std::function_ref<int()>&>);

static_assert(!std::is_constructible_v<std::function_ref<void()>, L3&>);
static_assert(!std::is_constructible_v<std::function_ref<void()>>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, void (A::*)()>);
static_assert(!std::is_constructible_v<std::function_ref<void(A*)>, void (A::*)()>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, L1&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, L1 const&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, L1&&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, L1 const&&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, std::function_ref<int()>&>);

// non-const noexcept(true)
static_assert(std::is_constructible_v<std::function_ref<A() noexcept>, L2Noexcept&>);
static_assert(std::is_constructible_v<std::function_ref<A() noexcept>, L2Noexcept const&>);
static_assert(std::is_constructible_v<std::function_ref<A() noexcept>, L2Noexcept&&>);
static_assert(std::is_constructible_v<std::function_ref<A() noexcept>, L2Noexcept const&&>);
static_assert(std::is_constructible_v<std::function_ref<A() noexcept>, std::function_ref<A&() noexcept>>);

static_assert(!std::is_constructible_v<std::function_ref<A() noexcept>, L2&>);
static_assert(!std::is_constructible_v<std::function_ref<A() noexcept>, L2 const&>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, L3&>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, void (A::*)() noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void(A*) noexcept>, void (A::*)() noexcept>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<A() noexcept>, L2Noexcept&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<A() noexcept>, L2Noexcept const&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<A() noexcept>, L2Noexcept&&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<A() noexcept>, L2Noexcept const&&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<A() noexcept>, std::function_ref<A&() noexcept>>);

// const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const>, L3&>);
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const>, const L3&>);
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const>, L3&&>);
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const>, const L3&&>);
static_assert(
    std::is_constructible_v<std::function_ref<void(int, double) const>, std::function_ref<int(int, double) const>>);

static_assert(std::is_constructible_v<std::function_ref<void(int, double)>, NonConstInvocable&>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double)>, const NonConstInvocable&>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const>, NonConstInvocable&>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const>>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const>, void (A::*)()>);
static_assert(!std::is_constructible_v<std::function_ref<void(A*) const>, void (A::*)()>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const>, L3&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const>, const L3&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const>, L3&&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const>, const L3&&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const>,
                                              std::function_ref<int(int, double) const>>);

// const noexcept(true)
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, L3Noexcept&>);
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, const L3Noexcept&>);
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, L3Noexcept&&>);
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, const L3Noexcept&&>);
static_assert(std::is_constructible_v<std::function_ref<void(int, double) const noexcept>,
                                      std::function_ref<int(int, double) const noexcept>>);

static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, L3&>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, const L3&>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, L3&&>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, const L3&&>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, void*>);
static_assert(!std::is_constructible_v<std::function_ref<void(int, double) const noexcept>, void (A::*)() noexcept>);
static_assert(!std::is_constructible_v<std::function_ref<void(A*) const>, void (A::*)() noexcept>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const noexcept>, L3Noexcept&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const noexcept>, const L3Noexcept&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const noexcept>, L3Noexcept&&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const noexcept>, const L3Noexcept&&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int, double) const noexcept>,
                                              std::function_ref<int(int, double) const noexcept>>);

struct F {
  int i;
  int operator()(auto&&...) { return 5 + i; }

  int operator()(auto&&...) const { return 6 + i; }
};

struct Int {
  int i;
  constexpr Int(int ii) noexcept : i(ii) {}
};

struct NeedsConversion {
  int operator()(Int x, Int y, Int z) const noexcept { return x.i + y.i + z.i; }
};

int fn() { return 5; }
constexpr auto fn_noexcept = []() noexcept { return 6; };

const auto one = []() noexcept { return 1; };
const auto two = []() noexcept { return 2; };

struct VolatileCallable {
  int operator()() volatile const { return 42; }
};

constexpr bool test() {
  {
    std::function_ref<void()> f(l1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f();
    }
  }
  {
    // explicit(false)
    std::function_ref<void()> f = l1;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f();
    }
  }
  {
    // noexcept
    std::function_ref<A() noexcept> f(l2_noexcept);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      auto a = f();
      assert(a.i == 5);
    }
  }
  {
    // const
    std::function_ref<double(int, double) const> f(l3);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2.0) == 3.0);
    }
  }
  {
    // const noexcept
    std::function_ref<double(int, double) const noexcept> f(l3_noexcept);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2.0) == 3.0);
    }
  }
  {
    // no copies of original callable
    auto local = [i = 5] mutable { return i++; };
    std::function_ref<int()> f(local);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f() == 5);
      assert(local() == 6);
      assert(f() == 7);
    }
  }
  {
    // const correctness
    F f{5};

    std::function_ref<int()> f1(f);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f1() == 10);
      assert(std::as_const(f1)() == 10);
    }

    std::function_ref<int() const> f2(f);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 11);
      assert(std::as_const(f2)() == 11);
    }
  }
  {
    // with conversions
    NeedsConversion c{};

    std::function_ref<Int(int, int, int)> f(c);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const> f2(c);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) noexcept> f3(c);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f3(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const noexcept> f4(c);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f4(1, 2, 3).i == 6);
    }
  }

  {
    // noexcept conversion
    std::function_ref<int() noexcept> f1(fn_noexcept);
    std::function_ref<int()> f2(f1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 6);
    }
  }

  {
    // const conversion
    std::function_ref<int() const> f1(fn_noexcept);
    std::function_ref<int()> f2(f1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 6);
    }
  }

  {
    // const noexcept conversion
    std::function_ref<int() const noexcept> f1(fn_noexcept);
    std::function_ref<int()> f2(f1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 6);
    }
  }

  {
    // const noexcept to noexcept conversion
    std::function_ref<int() const noexcept> f1(fn_noexcept);
    std::function_ref<int() noexcept> f2(f1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 6);
    }
  }

  {
    // const noexcept to const conversion
    std::function_ref<int() const noexcept> f1(fn_noexcept);
    std::function_ref<int() const> f2(f1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 6);
    }
  }

  {
    // P3961R1 Less double indirection in function_ref
    // double unwrapping
    std::function_ref<int() const noexcept> f1(std::cw<one>);
    std::function_ref<int()> f2(f1);

    f1 = std::cw<two>;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 1 || f2() == 2);
      LIBCPP_ASSERT(f2() == 1);
    }
  }
  {
    // P3961R1 Less double indirection in function_ref
    // is-convertible-from-specialization<remove_cv_t<T>> is false
    std::function_ref<int()> f1(std::cw<one>);
    std::function_ref<int() const> f2(f1);

    f1 = std::cw<two>;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 2);
    }
  }
  {
    // volatile objects
    const volatile VolatileCallable vc{};
    std::function_ref<int()> f(vc);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f() == 42);
    }
  }
  {
    // static call operator
    struct StaticCall {
      static int operator()(int x, int y) { return x + y; }
    };
    StaticCall sc;
    std::function_ref<int(int, int)> f(sc);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2) == 3);
    }
  }
  {
    // overload set with static and non-static call operator
    struct OverloadSet {
      static int operator()(int x) { return x + 1; }
      int operator()(long x) const { return x + 2; }
    };
    OverloadSet os;
    std::function_ref<int(int)> f1(os);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f1(1) == 2);
    }

    LIBCPP_STATIC_ASSERT(!std::__statically_callable<OverloadSet, long>);
    std::function_ref<int(long)> f2(os);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(1L) == 3);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
