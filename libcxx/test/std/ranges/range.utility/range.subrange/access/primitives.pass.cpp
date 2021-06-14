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

// class std::ranges::subrange;

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"
#include "../subrange_test_types.h"

// Note: begin and end tested in range.subrange.ctor.pass.cpp.

constexpr bool test() {
  std::ranges::subrange<MoveOnlyForwardIter, int*> a(MoveOnlyForwardIter(globalBuff), globalBuff + 8, 8);
  assert(a.begin().base == globalBuff);
  assert(!a.empty());
  assert(a.size() == 8);

  std::ranges::subrange<ForwardIter> b(ForwardIter(nullptr), ForwardIter(nullptr));
  assert(b.empty());

  std::ranges::subrange<ForwardIter> c{ForwardIter(globalBuff), ForwardIter(globalBuff)};
  assert(c.empty());

  std::ranges::subrange<ForwardIter> d(ForwardIter(globalBuff), ForwardIter(globalBuff + 1));
  assert(!d.empty());

  std::ranges::subrange<SizedSentinelForwardIter> e(SizedSentinelForwardIter(globalBuff),
                                                    SizedSentinelForwardIter(globalBuff + 8), 8);
  assert(!e.empty());
  assert(e.size() == 8);

  // Make sure that operator- is used to calculate size when possible.
  if (!std::is_constant_evaluated())
    assert(SizedSentinelForwardIter::minusCount == 1);

  return true;
}


int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
