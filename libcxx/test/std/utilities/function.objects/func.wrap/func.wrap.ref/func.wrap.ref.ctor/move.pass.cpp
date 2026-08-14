//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// Each specialization of function_ref is a trivially copyable type ([basic.types.general]) that models copyable.

#include <cassert>
#include <functional>
#include <utility>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_move_constructible_v<std::function_ref<void()>>);
static_assert(std::is_move_constructible_v<std::function_ref<void() const>>);
static_assert(std::is_move_constructible_v<std::function_ref<void() noexcept>>);
static_assert(std::is_move_constructible_v<std::function_ref<void() const noexcept>>);

static_assert(std::is_trivially_move_constructible_v<std::function_ref<void()>>);
static_assert(std::is_trivially_move_constructible_v<std::function_ref<void() const>>);
static_assert(std::is_trivially_move_constructible_v<std::function_ref<void() noexcept>>);
static_assert(std::is_trivially_move_constructible_v<std::function_ref<void() const noexcept>>);

double f1(int x, double y) noexcept { return x + y; }

constexpr bool test() {
  {
    std::function_ref<void()> f(std::cw<[] {}>);
    auto f2 = std::move(f);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      f2();
    }
  }
  {
    // const
    std::function_ref<int() const> f(std::cw<[] { return 42; }>);
    auto f2 = std::move(f);

    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2() == 42);
    }
  }
  {
    // noexcept
    std::function_ref<double(int, double) noexcept> f(std::cw<&f1>);
    auto f2 = std::move(f);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(1, 2.0) == 3.0);
    }
  }
  {
    // const noexcept
    std::function_ref<double(int, double) const noexcept> f(std::cw<&f1>);
    auto f2 = std::move(f);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(f2(1, 2.0) == 3.0);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
