//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<auto f, class U>
//   constexpr function_ref(constant_wrapper<f>, U&& obj) noexcept;

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

// Constraints:
// - is_rvalue_reference_v<U&&> is false, and
// - is-invocable-using<const F&, cv T&> is true.

auto l1          = [](int) {};
auto l1_noexcept = [](int) noexcept {};
auto l2          = [](int, double) {};
auto l2_noexcept = [](int, double) noexcept {};
auto l3          = [](int&) {};
auto l3_noexcept = [](int&) noexcept {};

struct NonConstInvocable {
  void operator()(long) noexcept {}
};

struct A {
  int i;
  void f() {}
  void f_const() const {}
  void f_noexcept() noexcept {}
  void f_const_noexcept() const noexcept {}
  void g(int&) {}
  void g_const(int&) const {}
  void g_noexcept(int&) noexcept {}
  void g_const_noexcept(int&) const noexcept {}
};

// non-const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<l1>, int&>);

static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<l1>, int&&>);

// LWG issue 4256
// https://cplusplus.github.io/LWG/issue4256
static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<NonConstInvocable{}>, long&>);

static_assert(std::is_constructible_v<std::function_ref<void(double)>, std::constant_wrapper<l2>, int&>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<l2>, int&>);

static_assert(std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<l3>, int&>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<l3>, const int&>);

static_assert(std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<&A::f>, A&>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<&A::g>, A&>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, std::constant_wrapper<l1>, int&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(double)>, std::constant_wrapper<l2>, int&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, std::constant_wrapper<l3>, int&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, std::constant_wrapper<&A::f>, A&>);

// non-const noexcept
static_assert(std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l1_noexcept>, int&>);

static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l1>, int&>);

static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l1_noexcept>, int&&>);

static_assert(
    !std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<NonConstInvocable{}>, long&>);

static_assert(
    std::is_constructible_v<std::function_ref<void(double) noexcept>, std::constant_wrapper<l2_noexcept>, int&>);
static_assert(!std::is_constructible_v<std::function_ref<void(double) noexcept>, std::constant_wrapper<l2>, int&>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l2_noexcept>, int&>);

static_assert(std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l3_noexcept>, int&>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l3>, const int&>);
static_assert(
    !std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l3_noexcept>, const int&>);

static_assert(std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<&A::f_noexcept>, A&>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<&A::f>, A&>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<&A::g_noexcept>, A&>);

// the constructor is noexcept
static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l1_noexcept>, int&>);
static_assert(
    std::
        is_nothrow_constructible_v<std::function_ref<void(double) noexcept>, std::constant_wrapper<l2_noexcept>, int&>);
static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l3_noexcept>, int&>);
static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<&A::f_noexcept>, A&>);

// const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l1>, int&>);

static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l1>, int&&>);

static_assert(
    !std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<NonConstInvocable{}>, long&>);

static_assert(std::is_constructible_v<std::function_ref<void(double) const>, std::constant_wrapper<l2>, int&>);
static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l2>, int&>);

static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l3>, int&>);
static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l3>, const int&>);

static_assert(std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<&A::f_const>, A&>);
static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<&A::f>, A&>);
static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<&A::g_const>, A&>);

// the constructor is noexcept
static_assert(std::is_nothrow_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l1>, int&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(double) const>, std::constant_wrapper<l2>, int&>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void() const>, std::constant_wrapper<&A::f_const>, A&>);

// const noexcept
static_assert(
    std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l1_noexcept>, int&>);

static_assert(!std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l1>, int&>);

static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l1_noexcept>, int&&>);

static_assert(!std::is_constructible_v<std::function_ref<void() const noexcept>,
                                       std::constant_wrapper<NonConstInvocable{}>,
                                       long&>);

static_assert(
    std::is_constructible_v<std::function_ref<void(double) const noexcept>, std::constant_wrapper<l2_noexcept>, int&>);
static_assert(
    !std::is_constructible_v<std::function_ref<void(double) const noexcept>, std::constant_wrapper<l2>, int&>);
static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l2_noexcept>, int&>);

static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l3_noexcept>, int&>);
static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l3>, const int&>);
static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l3_noexcept>, const int&>);

static_assert(
    std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<&A::f_const_noexcept>, A&>);
static_assert(!std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<&A::f>, A&>);
static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<&A::f_const>, A&>);
static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<&A::f_noexcept>, A&>);
static_assert(!std::is_constructible_v<std::function_ref<void() const noexcept>,
                                       std::constant_wrapper<&A::g_const_noexcept>,
                                       A&>);

// the constructor is noexcept
static_assert(
    std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l1_noexcept>, int&>);
static_assert(
    std::is_constructible_v<std::function_ref<void(double) const noexcept>, std::constant_wrapper<l2_noexcept>, int&>);
static_assert(
    std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<&A::f_const_noexcept>, A&>);

double f1(int x, double y) noexcept { return x + y; }

struct M {
  int i;
  int f() { return i; }
  int f_const() const { return i + 5; }
  int f_noexcept() noexcept { return i + 7; }
  int f_const_noexcept() const noexcept { return i + 9; }
  int g(int& j) {
    j = 42;
    return i + j;
  }
  int g_const(int& j) const {
    j = 42;
    return i + j + 1;
  }
  int g_noexcept(int& j) noexcept {
    j = 42;
    return i + j + 2;
  }
  int g_const_noexcept(int& j) const noexcept {
    j = 42;
    return i + j + 3;
  }
};

struct Int {
  int i;
  constexpr Int(int ii) noexcept : i(ii) {}
};

struct NeedsConversion {
  int operator()(Int x, Int y, Int z) const noexcept { return x.i + y.i + z.i; }
};

int needs_conversion(Int x, Int y, Int z) noexcept { return x.i + y.i + z.i; }

constexpr bool test() {
  {
    int i = 0;
    std::function_ref<void()> f(std::cw<[](int) {}>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f();
    }
  }
  {
    // explicit
    int i                       = 0;
    std::function_ref<void()> f = {std::cw<[](int) {}>, i};
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f();
    }
  }
  {
    // mutate
    int i;
    std::function_ref<double(double)> f(
        std::cw<[](int& j, double d) {
          j = 5;
          return j + d;
        }>,
        i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(3.3) == 8.3);
      assert(i == 5);
    }
  }
  {
    // const
    int i = 5;
    std::function_ref<int() const> f(std::cw<[](int j) { return j + 42; }>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f() == 47);
    }
  }
  {
    // noexcept
    int i = 5;
    std::function_ref<double(double) noexcept> f(std::cw<&f1>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(2.0) == 7.0);
    }
  }
  {
    // const noexcept
    int i = 5;
    std::function_ref<double(double) const noexcept> f(std::cw<&f1>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(2.0) == 7.0);
    }
  }
  {
    // member ptr
    M m{3};
    std::function_ref<int()> f(std::cw<&M::f>, m);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f() == 3);
    }

    int j = 0;
    std::function_ref<int(int&)> g(std::cw<&M::g>, m);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(g(j) == 45);
      assert(j == 42);
    }

    std::function_ref<int() const> f_const(std::cw<&M::f_const>, m);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_const() == 8);
    }

    j = 0;
    std::function_ref<int(int&)> g_const(std::cw<&M::g_const>, m);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(g_const(j) == 46);
      assert(j == 42);
    }

    std::function_ref<int() noexcept> f_noexcept(std::cw<&M::f_noexcept>, m);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_noexcept() == 10);
    }

    j = 0;
    std::function_ref<int(int&) noexcept> g_noexcept(std::cw<&M::g_noexcept>, m);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(g_noexcept(j) == 47);
      assert(j == 42);
    }

    std::function_ref<int() const noexcept> f_const_noexcept(std::cw<&M::f_const_noexcept>, m);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_const_noexcept() == 12);
    }

    j = 0;
    std::function_ref<int(int&) const noexcept> g_const_noexcept(std::cw<&M::g_const_noexcept>, m);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(g_const_noexcept(j) == 48);
      assert(j == 42);
    }
  }
  {
    // with conversions
    int i = 1;

    std::function_ref<Int(int, int)> f(std::cw<NeedsConversion{}>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(2, 3).i == 6);
    }

    std::function_ref<Int(int, int) const> f2(std::cw<NeedsConversion{}>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(2, 3).i == 6);
    }

    std::function_ref<Int(int, int) noexcept> f3(std::cw<NeedsConversion{}>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f3(2, 3).i == 6);
    }

    std::function_ref<Int(int, int) const noexcept> f4(std::cw<NeedsConversion{}>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f4(2, 3).i == 6);
    }
  }
  {
    // with conversions function pointer
    int i = 1;
    std::function_ref<Int(int, int)> f(std::cw<&needs_conversion>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(2, 3).i == 6);
    }

    std::function_ref<Int(int, int) const> f2(std::cw<&needs_conversion>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(2, 3).i == 6);
    }

    std::function_ref<Int(int, int) noexcept> f3(std::cw<&needs_conversion>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f3(2, 3).i == 6);
    }

    std::function_ref<Int(int, int) const noexcept> f4(std::cw<&needs_conversion>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f4(2, 3).i == 6);
    }
  }
  {
    // volatile objects
    volatile int i = 5;
    std::function_ref<int() > f(std::cw<[](int j) { return j + 42; }>, i);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f() == 47);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
