//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// class std::ranges::subrange;

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"
#include "../subrange_test_types.h"

constexpr bool test() {
  std::ranges::subrange<int*> a(globalBuff, globalBuff + 8, 8);
  auto a1 = a.next();
  assert(a1.begin() == globalBuff + 1);
  assert(a1.size() == 7);
  auto a5 = a.next(5);
  assert(a5.begin() == globalBuff + 5);
  assert(a5.size() == 3);
  auto a4 = a5.prev();
  assert(a4.begin() == globalBuff + 4);
  assert(a4.size() == 4);

  std::ranges::subrange<InputIter> b(InputIter(globalBuff), InputIter(globalBuff + 8));
  auto b1 = std::move(b).next();
  assert(b1.begin().base() == globalBuff + 1);

  std::ranges::subrange<BidirIter> c(BidirIter(globalBuff + 4), BidirIter(globalBuff + 8));
  auto c1 = c.prev();
  assert(c1.begin().base() == globalBuff + 3);
  auto c2 = c.prev(4);
  assert(c2.begin().base() == globalBuff);

  std::ranges::subrange<BidirIter> d(BidirIter(globalBuff + 4), BidirIter(globalBuff + 8));
  auto d1 = d.advance(4);
  assert(d1.begin().base() == globalBuff + 8);
  assert(d1.empty());
  auto d2 = d1.advance(-4);
  assert(d2.begin().base() == globalBuff + 4);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
