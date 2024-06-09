//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr function_ref(const function_ref&) noexcept = default;

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_copy_constructible_v<std::function_ref<void()>>);
static_assert(std::is_copy_constructible_v<std::function_ref<void() const>>);
static_assert(std::is_copy_constructible_v<std::function_ref<void() noexcept>>);
static_assert(std::is_copy_constructible_v<std::function_ref<void() const noexcept>>);

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
    auto f2 = f;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f2();
    }
  }
  {
    // const
    std::function_ref<int() const> f(std::cw<[] { return 42; }>);
    auto f2 = f;

    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 42);
    }
  }
  {
    // noexcept
    std::function_ref<double(int, double) noexcept> f(std::cw<&f1>);
    auto f2 = f;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(1, 2.0) == 3.0);
    }
  }
  {
    // const noexcept
    std::function_ref<double(int, double) const noexcept> f(std::cw<&f1>);
    auto f2 = f;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(1, 2.0) == 3.0);
    }
  }
  {
    // with conversions
    std::function_ref<Int(int, int, int)> f(std::cw<NeedsConversion{}>);
    auto f_copy = f;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_copy(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const> f2(std::cw<NeedsConversion{}>);
    auto f2_copy = f2;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2_copy(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) noexcept> f3(std::cw<NeedsConversion{}>);
    auto f3_copy = f3;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f3_copy(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const noexcept> f4(std::cw<NeedsConversion{}>);
    auto f4_copy = f4;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f4_copy(1, 2, 3).i == 6);
    }
  }
  {
    // with conversions function pointer
    std::function_ref<Int(int, int, int)> f(std::cw<&needs_conversion>);
    auto f_copy = f;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f_copy(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const> f2(std::cw<&needs_conversion>);
    auto f2_copy = f2;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2_copy(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) noexcept> f3(std::cw<&needs_conversion>);
    auto f3_copy = f3;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f3_copy(1, 2, 3).i == 6);
    }

    std::function_ref<Int(int, int, int) const noexcept> f4(std::cw<&needs_conversion>);
    auto f4_copy = f4;
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f4_copy(1, 2, 3).i == 6);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
