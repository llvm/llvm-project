//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <optional>

// [optional.nullops], comparison with nullopt

// template<class T>
//   constexpr strong_ordering operator<=>(const optional<T>&, nullopt_t) noexcept;

#include <cassert>
#include <compare>
#include <optional>

#include "test_comparisons.h"

constexpr bool test() {
  {
    std::optional<int> op;
    assert((std::nullopt <=> op) == std::strong_ordering::equal);
    assert(testOrder(std::nullopt, op, std::strong_ordering::equal));
    assert((op <=> std::nullopt) == std::strong_ordering::equal);
    assert(testOrder(op, std::nullopt, std::strong_ordering::equal));
  }
  {
    std::optional<int> op{1};
    assert((std::nullopt <=> op) == std::strong_ordering::less);
    assert(testOrder(std::nullopt, op, std::strong_ordering::less));
  }
  {
    std::optional<int> op{1};
    assert((op <=> std::nullopt) == std::strong_ordering::greater);
    assert(testOrder(op, std::nullopt, std::strong_ordering::greater));
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
