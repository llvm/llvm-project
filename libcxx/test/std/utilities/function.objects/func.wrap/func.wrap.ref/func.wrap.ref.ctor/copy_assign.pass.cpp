//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr function_ref& operator=(const function_ref&) noexcept = default;

#include <cassert>
#include <concepts>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_copy_assignable_v<std::function_ref<void()>>);
static_assert(std::is_copy_assignable_v<std::function_ref<void() const>>);
static_assert(std::is_copy_assignable_v<std::function_ref<void() noexcept>>);
static_assert(std::is_copy_assignable_v<std::function_ref<void() const noexcept>>);

static_assert(std::is_trivially_copy_assignable_v<std::function_ref<void()>>);
static_assert(std::is_trivially_copy_assignable_v<std::function_ref<void() const>>);
static_assert(std::is_trivially_copy_assignable_v<std::function_ref<void() noexcept>>);
static_assert(std::is_trivially_copy_assignable_v<std::function_ref<void() const noexcept>>);

double plus(int x, double y) noexcept { return x + y; }
double minus(int x, double y) noexcept { return x - y; }

struct Int {
  int i;
  constexpr Int(int ii) noexcept : i(ii) {}
};

struct NeedsConversion {
  int operator()(Int x, Int y, Int z) const noexcept { return x.i + y.i + z.i; }
};

int needs_conversion(Int x, Int y, Int z) noexcept { return x.i + y.i + z.i; }
int zero(Int, Int, Int) noexcept { return 0; }

constexpr bool test() {
  {
    std::function_ref<void()> f(std::cw<[] {}>);
    std::function_ref<void()> f2(std::cw<[] {}>);
    std::same_as<std::function_ref<void()>&> decltype(auto) result = f2 = f;
    assert(&result == &f2);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f();
      f2();
    }
  }
  {
    // const
    std::function_ref<int() const> f(std::cw<[] { return 42; }>);
    std::function_ref<int() const> f2(std::cw<[] { return 41; }>);
    std::same_as<std::function_ref<int() const>&> decltype(auto) result = f2 = f;
    assert(&result == &f2);

    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f() == 42);
      assert(f2() == 42);
    }
  }
  {
    // noexcept
    std::function_ref<double(int, double) noexcept> f(std::cw<&plus>);
    std::function_ref<double(int, double) noexcept> f2(std::cw<&minus>);
    std::same_as<std::function_ref<double(int, double) noexcept>&> decltype(auto) result = f2 = f;
    assert(&result == &f2);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2.0) == 3.0);
      assert(f2(1, 2.0) == 3.0);
    }
  }
  {
    // const noexcept
    std::function_ref<double(int, double) const noexcept> f(std::cw<&plus>);
    std::function_ref<double(int, double) const noexcept> f2(std::cw<&minus>);
    std::same_as<std::function_ref<double(int, double) const noexcept>&> decltype(auto) result = f2 = f;
    assert(&result == &f2);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2.0) == 3.0);
      assert(f2(1, 2.0) == 3.0);
    }
  }
  {
    // with conversions
    std::function_ref<Int(int, int, int)> f(std::cw<[](int, int, int) { return Int{1}; }>);
    std::function_ref<Int(int, int, int)> f2(std::cw<NeedsConversion{}>);
    std::same_as<std::function_ref<Int(int, int, int)>&> decltype(auto) result = f = f2;
    assert(&result == &f);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2, 3).i == 6);
      assert(f2(1, 2, 3).i == 6);
    }
  }

  {
    // with conversions
    // const
    std::function_ref<Int(int, int, int) const> f_const(std::cw<[](int, int, int) { return Int{1}; }>);
    std::function_ref<Int(int, int, int) const> f2_const(std::cw<NeedsConversion{}>);
    std::same_as<std::function_ref<Int(int, int, int) const>&> decltype(auto) result = f_const = f2_const;
    assert(&result == &f_const);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_const(1, 2, 3).i == 6);
      assert(f2_const(1, 2, 3).i == 6);
    }
  }
  {
    // with conversions
    // noexcept
    std::function_ref<Int(int, int, int) noexcept> f_noexcept(std::cw<[](int, int, int) noexcept { return Int{1}; }>);
    std::function_ref<Int(int, int, int) noexcept> f2_noexcept(std::cw<NeedsConversion{}>);
    std::same_as<std::function_ref<Int(int, int, int) noexcept>&> decltype(auto) result = f_noexcept = f2_noexcept;
    assert(&result == &f_noexcept);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_noexcept(1, 2, 3).i == 6);
      assert(f2_noexcept(1, 2, 3).i == 6);
    }
  }
  {
    // with conversions
    // const noexcept
    std::function_ref<Int(int, int, int) const noexcept> f_const_noexcept(
        std::cw<[](int, int, int) noexcept { return Int{1}; }>);
    std::function_ref<Int(int, int, int) const noexcept> f2_const_noexcept(std::cw<NeedsConversion{}>);
    std::same_as<std::function_ref<Int(int, int, int) const noexcept>&> decltype(auto) result = f_const_noexcept =
        f2_const_noexcept;
    assert(&result == &f_const_noexcept);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_const_noexcept(1, 2, 3).i == 6);
      assert(f2_const_noexcept(1, 2, 3).i == 6);
    }
  }
  {
    // with conversions function pointer
    std::function_ref<Int(int, int, int)> f(std::cw<&zero>);
    std::function_ref<Int(int, int, int)> f2(std::cw<&needs_conversion>);
    std::same_as<std::function_ref<Int(int, int, int)>&> decltype(auto) result = f = f2;
    assert(&result == &f);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f(1, 2, 3).i == 6);
      assert(f2(1, 2, 3).i == 6);
    }
  }

  {
    // with conversions function pointer
    // const
    std::function_ref<Int(int, int, int) const> f_const(std::cw<&zero>);
    std::function_ref<Int(int, int, int) const> f2_const(std::cw<&needs_conversion>);
    std::same_as<std::function_ref<Int(int, int, int) const>&> decltype(auto) result = f_const = f2_const;
    assert(&result == &f_const);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_const(1, 2, 3).i == 6);
      assert(f2_const(1, 2, 3).i == 6);
    }
  }
  {
    // with conversions function pointer
    // noexcept
    std::function_ref<Int(int, int, int) noexcept> f_noexcept(std::cw<&zero>);
    std::function_ref<Int(int, int, int) noexcept> f2_noexcept(std::cw<&needs_conversion>);
    std::same_as<std::function_ref<Int(int, int, int) noexcept>&> decltype(auto) result = f_noexcept = f2_noexcept;
    assert(&result == &f_noexcept);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_noexcept(1, 2, 3).i == 6);
      assert(f2_noexcept(1, 2, 3).i == 6);
    }
  }

  {
    // with conversions function pointer
    // const noexcept
    std::function_ref<Int(int, int, int) const noexcept> f_const_noexcept(std::cw<&zero>);
    std::function_ref<Int(int, int, int) const noexcept> f2_const_noexcept(std::cw<&needs_conversion>);
    std::same_as<std::function_ref<Int(int, int, int) const noexcept>&> decltype(auto) result = f_const_noexcept =
        f2_const_noexcept;
    assert(&result == &f_const_noexcept);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_const_noexcept(1, 2, 3).i == 6);
      assert(f2_const_noexcept(1, 2, 3).i == 6);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
