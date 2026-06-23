//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr bool has_error() const noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "../../types.h"

constexpr bool test() {
  {
    const std::expected<int, int> e(std::unexpect, 5);
    static_assert(noexcept(e.has_error()));
    std::same_as<bool> decltype(auto) has_err = e.has_error();
    assert(has_err);
  }

  {
    const std::expected<int, int> e(5);
    assert(!e.has_error());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
