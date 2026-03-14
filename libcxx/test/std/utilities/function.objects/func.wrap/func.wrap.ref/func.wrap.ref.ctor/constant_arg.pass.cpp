//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<auto f> constexpr function_ref(constant_arg_t<f>) noexcept;

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
static_assert(std::is_constructible_v<std::function_ref<void()>, std::constant_arg_t<l1>>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_arg_t<NonConstInvocable{}>>);
static_assert(!std::is_constructible_v<std::function_ref<void()>, std::constant_arg_t<l2>>);
static_assert(std::is_constructible_v<std::function_ref<void(int)>, std::constant_arg_t<l2>>);

static_assert(std::is_nothrow_constructible_v<std::function_ref<void()>, std::constant_arg_t<l1>>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int)>, std::constant_arg_t<l2>>);

// non-const noexcept
static_assert(std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_arg_t<l1_noexcept>>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_arg_t<l1>>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_arg_t<NonConstInvocable{}>>);
static_assert(!std::is_constructible_v<std::function_ref<void() noexcept>, std::constant_arg_t<l2_noexcept>>);
static_assert(std::is_constructible_v<std::function_ref<void(int) noexcept>, std::constant_arg_t<l2_noexcept>>);

static_assert(std::is_nothrow_constructible_v<std::function_ref<void() noexcept>, std::constant_arg_t<l1_noexcept>>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int) noexcept>, std::constant_arg_t<l2_noexcept>>);

// const noexcept(false)
static_assert(std::is_constructible_v<std::function_ref<void() const>, std::constant_arg_t<l1>>);
static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_arg_t<NonConstInvocable{}>>);
static_assert(!std::is_constructible_v<std::function_ref<void() const>, std::constant_arg_t<l2>>);
static_assert(std::is_constructible_v<std::function_ref<void(int) const>, std::constant_arg_t<l2>>);

static_assert(std::is_nothrow_constructible_v<std::function_ref<void() const>, std::constant_arg_t<l1>>);
static_assert(std::is_nothrow_constructible_v<std::function_ref<void(int) const>, std::constant_arg_t<l2>>);

// const noexcept
static_assert(std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_arg_t<l1_noexcept>>);
static_assert(!std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_arg_t<l1>>);
static_assert(
    !std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_arg_t<NonConstInvocable{}>>);
static_assert(!std::is_constructible_v<std::function_ref<void() const noexcept>, std::constant_arg_t<l2_noexcept>>);
static_assert(std::is_constructible_v<std::function_ref<void(int) const noexcept>, std::constant_arg_t<l2_noexcept>>);

static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void() const noexcept>, std::constant_arg_t<l1_noexcept>>);
static_assert(
    std::is_nothrow_constructible_v<std::function_ref<void(int) const noexcept>, std::constant_arg_t<l2_noexcept>>);

constexpr double f1(int x, double y) noexcept { return x + y; }

struct Int {
  int i;
  constexpr Int(int ii) noexcept : i(ii) {}
};

struct NeedsConversion {
  constexpr int operator()(Int x, Int y, Int z) const noexcept { return x.i + y.i + z.i; }
};

constexpr int needs_conversion(Int x, Int y, Int z) noexcept { return x.i + y.i + z.i; }

constexpr bool test() {
  {
    std::function_ref<void()> f(std::constant_arg<[] {}>);
    f();
  }
  {
    // explicit
    std::function_ref<void()> f = std::constant_arg<[] {}>;
    f();
  }
  {
    // const
    std::function_ref<int() const> f(std::constant_arg<[] { return 42; }>);
    assert(f() == 42);
  }
  {
    // noexcept
    std::function_ref<double(int, double) noexcept> f(std::constant_arg<&f1>);
    assert(f(1, 2.0) == 3.0);
  }
  {
    // const noexcept
    std::function_ref<double(int, double) const noexcept> f(std::constant_arg<&f1>);
    assert(f(1, 2.0) == 3.0);
  }
  {
    // with conversions
    std::function_ref<Int(int, int, int)> f(std::constant_arg<NeedsConversion{}>);
    assert(f(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const> f2(std::constant_arg<NeedsConversion{}>);
    assert(f2(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) noexcept> f3(std::constant_arg<NeedsConversion{}>);
    assert(f3(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const noexcept> f4(std::constant_arg<NeedsConversion{}>);
    assert(f4(1, 2, 3).i == 6);
  }
  {
    // with conversions function pointer
    std::function_ref<Int(int, int, int)> f(std::constant_arg<&needs_conversion>);
    assert(f(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const> f2(std::constant_arg<&needs_conversion>);
    assert(f2(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) noexcept> f3(std::constant_arg<&needs_conversion>);
    assert(f3(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const noexcept> f4(std::constant_arg<&needs_conversion>);
    assert(f4(1, 2, 3).i == 6);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
