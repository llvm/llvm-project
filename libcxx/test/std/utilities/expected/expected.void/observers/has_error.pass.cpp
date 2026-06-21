//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// constexpr bool has_error() const noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "../../types.h"

static_assert(noexcept(std::expected<void, int>().has_error()));

constexpr bool test() {
  {
    const std::expected<void, int> e(std::unexpect, 5);
    assert(e.has_error());
  }

  {
    const std::expected<void, int> e;
    assert(!e.has_error());
  }

  return true;
}

constexpr bool test_nodiscard() {
  std::expected<void, int> e;
  e.has_error(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  test_nodiscard();
  static_assert(test_nodiscard());

  return 0;
}
