//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <optional>

// [optional.relops], relational operators

// template<class T, three_way_comparable_with<T> U>
//   constexpr compare_three_way_result_t<T, U>
//     operator<=>(const optional<T>&, const optional<U>&);

#include <cassert>
#include <compare>
#include <optional>

#include "test_comparisons.h"

constexpr bool test() {
  {
    std::optional<int> op1;
    std::optional<int> op2;

    assert((op1 <=> op2) == std::strong_ordering::equal);
    assert(testOrder(op1, op2, std::strong_ordering::equal));
  }
  {
    std::optional<int> op1{3};
    std::optional<int> op2{3};
    assert((op1 <=> op1) == std::strong_ordering::equal);
    assert(testOrder(op1, op1, std::strong_ordering::equal));
    assert((op1 <=> op2) == std::strong_ordering::equal);
    assert(testOrder(op1, op2, std::strong_ordering::equal));
    assert((op2 <=> op1) == std::strong_ordering::equal);
    assert(testOrder(op2, op1, std::strong_ordering::equal));
  }
  {
    std::optional<int> op;
    std::optional<int> op1{2};
    std::optional<int> op2{3};
    assert((op <=> op2) == std::strong_ordering::less);
    assert(testOrder(op, op2, std::strong_ordering::less));
    assert((op1 <=> op2) == std::strong_ordering::less);
    assert(testOrder(op1, op2, std::strong_ordering::less));
  }
  {
    std::optional<int> op;
    std::optional<int> op1{3};
    std::optional<int> op2{2};
    assert((op1 <=> op) == std::strong_ordering::greater);
    assert(testOrder(op1, op, std::strong_ordering::greater));
    assert((op1 <=> op2) == std::strong_ordering::greater);
    assert(testOrder(op1, op2, std::strong_ordering::greater));
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
