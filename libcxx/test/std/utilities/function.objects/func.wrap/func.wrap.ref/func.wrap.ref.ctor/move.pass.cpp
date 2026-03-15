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

static_assert(std::is_move_constructible_v<std::function_ref<void()>>);
static_assert(std::is_move_constructible_v<std::function_ref<void() const>>);
static_assert(std::is_move_constructible_v<std::function_ref<void() noexcept>>);
static_assert(std::is_move_constructible_v<std::function_ref<void() const noexcept>>);

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
    auto f2 = std::move(f);
    f2();
  }
  {
    // const
    std::function_ref<int() const> f(std::constant_arg<[] { return 42; }>);
    auto f2 = std::move(f);

    assert(f2() == 42);
  }
  {
    // noexcept
    std::function_ref<double(int, double) noexcept> f(std::constant_arg<&f1>);
    auto f2 = std::move(f);
    assert(f2(1, 2.0) == 3.0);
  }
  {
    // const noexcept
    std::function_ref<double(int, double) const noexcept> f(std::constant_arg<&f1>);
    auto f2 = std::move(f);
    assert(f2(1, 2.0) == 3.0);
  }
  {
    // with conversions
    std::function_ref<Int(int, int, int)> f(std::constant_arg<NeedsConversion{}>);
    auto f_copy = std::move(f);
    assert(f_copy(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const> f2(std::constant_arg<NeedsConversion{}>);
    auto f2_copy = std::move(f2);
    assert(f2_copy(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) noexcept> f3(std::constant_arg<NeedsConversion{}>);
    auto f3_copy = std::move(f3);
    assert(f3_copy(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const noexcept> f4(std::constant_arg<NeedsConversion{}>);
    auto f4_copy = std::move(f4);
    assert(f4_copy(1, 2, 3).i == 6);
  }
  {
    // with conversions function pointer
    std::function_ref<Int(int, int, int)> f(std::constant_arg<&needs_conversion>);
    auto f_copy = std::move(f);
    assert(f_copy(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const> f2(std::constant_arg<&needs_conversion>);
    auto f2_copy = std::move(f2);
    assert(f2_copy(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) noexcept> f3(std::constant_arg<&needs_conversion>);
    auto f3_copy = std::move(f3);
    assert(f3_copy(1, 2, 3).i == 6);

    std::function_ref<Int(int, int, int) const noexcept> f4(std::constant_arg<&needs_conversion>);
    auto f4_copy = std::move(f4);
    assert(f4_copy(1, 2, 3).i == 6);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
