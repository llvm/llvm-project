//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// class std::ranges::subrange;

#include <ranges>

#include "types.h"
#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

static_assert( std::is_constructible_v<ForwardSubrange, ForwardBorrowedRange>); // Default case.
static_assert(!std::is_constructible_v<ForwardSubrange, ForwardRange>); // Not borrowed.
// Iter convertible to sentinel (pointer) type.
static_assert( std::is_constructible_v<ConvertibleForwardSubrange, ConvertibleForwardBorrowedRange>);
// Where neither iter or sentinel are pointers, but they are different.
static_assert( std::is_constructible_v<DifferentSentinelSubrange, ForwardBorrowedRangeDifferentSentinel>);
static_assert( std::is_constructible_v<DifferentSentinelWithSizeMemberSubrange, DifferentSentinelWithSizeMember>);

constexpr bool test() {
  ForwardSubrange a{ForwardBorrowedRange()};
  assert(a.begin().base() == globalBuff);
  assert(a.end().base() == globalBuff + 8);

  ConvertibleForwardSubrange b{ConvertibleForwardBorrowedRange()};
  assert(b.begin() == globalBuff);
  assert(b.end() == globalBuff + 8);

  DifferentSentinelSubrange c{ForwardBorrowedRangeDifferentSentinel()};
  assert(c.begin().base() == globalBuff);
  assert(c.end().value == globalBuff + 8);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
