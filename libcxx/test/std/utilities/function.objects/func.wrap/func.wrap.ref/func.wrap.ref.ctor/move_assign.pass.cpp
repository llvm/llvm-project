//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_move_assignable_v<std::function_ref<void()>>);
static_assert(std::is_move_assignable_v<std::function_ref<void() const>>);
static_assert(std::is_move_assignable_v<std::function_ref<void() noexcept>>);
static_assert(std::is_move_assignable_v<std::function_ref<void() const noexcept>>);

constexpr double plus(int x, double y) noexcept { return x + y; }
constexpr double minus(int x, double y) noexcept { return x - y; }

struct Int {
  int i;
  constexpr Int(int ii) noexcept : i(ii) {}
};

struct NeedsConversion {
  constexpr int operator()(Int x, Int y, Int z) const noexcept { return x.i + y.i + z.i; }
};

constexpr int needs_conversion(Int x, Int y, Int z) noexcept { return x.i + y.i + z.i; }
constexpr int zero(Int, Int, Int) noexcept { return 0; }

constexpr bool test() {
  {
    std::function_ref<void()> f(std::constant_arg<[] {}>);
    std::function_ref<void()> f2(std::constant_arg<[] {}>);
    f2 = std::move(f);
    f();
    f2();
  }
  {
    // const
    std::function_ref<int() const> f(std::constant_arg<[] { return 42; }>);
    std::function_ref<int() const> f2(std::constant_arg<[] { return 41; }>);
    f2 = std::move(f);

    assert(f() == 42);
    assert(f2() == 42);
  }
  {
    // noexcept
    std::function_ref<double(int, double) noexcept> f(std::constant_arg<&plus>);
    std::function_ref<double(int, double) noexcept> f2(std::constant_arg<&minus>);
    f2 = std::move(f);
    assert(f(1, 2.0) == 3.0);
    assert(f2(1, 2.0) == 3.0);
  }
  {
    // const noexcept
    std::function_ref<double(int, double) const noexcept> f(std::constant_arg<&plus>);
    std::function_ref<double(int, double) const noexcept> f2(std::constant_arg<&minus>);
    f2 = std::move(f);
    assert(f(1, 2.0) == 3.0);
    assert(f2(1, 2.0) == 3.0);
  }
  {
    // with conversions
    std::function_ref<Int(int, int, int)> f(std::constant_arg<[](int, int, int) { return Int{1}; }>);
    std::function_ref<Int(int, int, int)> f2(std::constant_arg<NeedsConversion{}>);
    f = std::move(f2);
    assert(f(1, 2, 3).i == 6);
    assert(f2(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const> f_const(std::constant_arg<[](int, int, int) { return Int{1}; }>);
    std::function_ref<Int(int, int, int) const> f2_const(std::constant_arg<NeedsConversion{}>);
    f_const = std::move(f2_const);
    assert(f_const(1, 2, 3).i == 6);
    assert(f2_const(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) noexcept> f_noexcept(
        std::constant_arg<[](int, int, int) noexcept { return Int{1}; }>);
    std::function_ref<Int(int, int, int) noexcept> f2_noexcept(std::constant_arg<NeedsConversion{}>);
    f_noexcept = std::move(f2_noexcept);
    assert(f_noexcept(1, 2, 3).i == 6);
    assert(f2_noexcept(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const noexcept> f_const_noexcept(
        std::constant_arg<[](int, int, int) noexcept { return Int{1}; }>);
    std::function_ref<Int(int, int, int) const noexcept> f2_const_noexcept(std::constant_arg<NeedsConversion{}>);
    f_const_noexcept = std::move(f2_const_noexcept);
    assert(f_const_noexcept(1, 2, 3).i == 6);
    assert(f2_const_noexcept(1, 2, 3).i == 6);
  }
  {
    // with conversions function pointer
    std::function_ref<Int(int, int, int)> f(std::constant_arg<&zero>);
    std::function_ref<Int(int, int, int)> f2(std::constant_arg<&needs_conversion>);
    f = std::move(f2);
    assert(f(1, 2, 3).i == 6);
    assert(f2(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const> f_const(std::constant_arg<&zero>);
    std::function_ref<Int(int, int, int) const> f2_const(std::constant_arg<&needs_conversion>);
    f_const = std::move(f2_const);
    assert(f_const(1, 2, 3).i == 6);
    assert(f2_const(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) noexcept> f_noexcept(std::constant_arg<&zero>);
    std::function_ref<Int(int, int, int) noexcept> f2_noexcept(std::constant_arg<&needs_conversion>);
    f_noexcept = std::move(f2_noexcept);
    assert(f_noexcept(1, 2, 3).i == 6);
    assert(f2_noexcept(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const noexcept> f_const_noexcept(std::constant_arg<&zero>);
    std::function_ref<Int(int, int, int) const noexcept> f2_const_noexcept(std::constant_arg<&needs_conversion>);
    f_const_noexcept = std::move(f2_const_noexcept);
    assert(f_const_noexcept(1, 2, 3).i == 6);
    assert(f2_const_noexcept(1, 2, 3).i == 6);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
