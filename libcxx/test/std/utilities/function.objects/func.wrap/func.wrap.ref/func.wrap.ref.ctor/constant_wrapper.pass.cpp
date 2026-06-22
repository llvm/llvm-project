//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<auto f> constexpr function_ref(constant_wrapper<f>) noexcept;

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

// Constraints: is-invocable-using<const F&> is true.

auto l1          = [] {};
auto l1_noexcept = [] noexcept {};
auto l2          = [](int) {};
auto l2_noexcept = [](int) noexcept {};

struct NonConstInvocable {
  void operator()() noexcept {}
};

// non-const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<l1>>);
// LWG issue 4256
// https://cplusplus.github.io/LWG/issue4256
static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<NonConstInvocable{}>>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_wrapper<l2>>);
static_assert(std::is_constructible_v<std::function_ref<void(int)>, std::constant_wrapper<l2>>);

static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, std::constant_wrapper<l1>>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int)>, std::constant_wrapper<l2>>);

// non-const noexcept
static_assert(std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l1_noexcept>>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l1>>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<NonConstInvocable{}>>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l2_noexcept>>);
static_assert(std::is_constructible_v<std::function_ref<void(int) noexcept>, std::constant_wrapper<l2_noexcept>>);

static_assert(std::is_nothrow_constructible_v<std::function_ref<void() noexcept>, std::constant_wrapper<l1_noexcept>>);
static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void(int) noexcept>, std::constant_wrapper<l2_noexcept>>);

// const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l1>>);
static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<NonConstInvocable{}>>);
static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l2>>);
static_assert(std::is_constructible_v<std::function_ref<void(int) const>, std::constant_wrapper<l2>>);

static_assert(std::is_nothrow_constructible_v<std::function_ref<void() const>, std::constant_wrapper<l1>>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int) const>, std::constant_wrapper<l2>>);

// const noexcept
static_assert(std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l1_noexcept>>);
static_assert(!std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l1>>);
static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<NonConstInvocable{}>>);
static_assert(!std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l2_noexcept>>);
static_assert(std::is_constructible_v<std::function_ref<void(int) const noexcept>, std::constant_wrapper<l2_noexcept>>);

static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void() const noexcept>, std::constant_wrapper<l1_noexcept>>);
static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void(int) const noexcept>, std::constant_wrapper<l2_noexcept>>);

double f1(int x, double y) noexcept { return x + y; }

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
    std::function_ref<void()> f(std::cw<[] {}>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f();
    }
  }
  {
    // explicit
    std::function_ref<void()> f = std::cw<[] {}>;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f();
    }
  }
  {
    // const
    std::function_ref<int() const> f(std::cw<[] { return 42; }>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f() == 42);
    }
  }
  {
    // noexcept
    std::function_ref<double(int, double) noexcept> f(std::cw<&f1>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2.0) == 3.0);
    }
  }
  {
    // const noexcept
    std::function_ref<double(int, double) const noexcept> f(std::cw<&f1>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2.0) == 3.0);
    }
  }
  {
    // with conversions
    std::function_ref<Int(int, int, int)> f(std::cw<NeedsConversion{}>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const> f2(std::cw<NeedsConversion{}>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) noexcept> f3(std::cw<NeedsConversion{}>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f3(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const noexcept> f4(std::cw<NeedsConversion{}>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f4(1, 2, 3).i == 6);
    }
  }
  {
    // with conversions function pointer
    std::function_ref<Int(int, int, int)> f(std::cw<&needs_conversion>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const> f2(std::cw<&needs_conversion>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) noexcept> f3(std::cw<&needs_conversion>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f3(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const noexcept> f4(std::cw<&needs_conversion>);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f4(1, 2, 3).i == 6);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
