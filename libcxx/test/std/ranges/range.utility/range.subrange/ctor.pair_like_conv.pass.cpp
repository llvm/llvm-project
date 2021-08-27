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
#include <tuple>
#include <utility>

static_assert( std::is_convertible_v<ForwardSubrange, std::pair<ForwardIter, ForwardIter>>);
static_assert( std::is_convertible_v<ForwardSubrange, std::tuple<ForwardIter, ForwardIter>>);
static_assert(!std::is_convertible_v<ForwardSubrange, std::tuple<ForwardIter, ForwardIter>&>);
static_assert(!std::is_convertible_v<ForwardSubrange, std::tuple<ForwardIter, ForwardIter, ForwardIter>>);
static_assert( std::is_convertible_v<ConvertibleForwardSubrange, std::tuple<ConvertibleForwardIter, int*>>);
static_assert(!std::is_convertible_v<SizedIntPtrSubrange, std::tuple<long*, int*>>);
static_assert( std::is_convertible_v<SizedIntPtrSubrange, std::tuple<int*, int*>>);

constexpr bool test() {
  ForwardSubrange a(ForwardIter(globalBuff), ForwardIter(globalBuff + 8));
  std::pair<ForwardIter, ForwardIter> aPair = a;
  assert(aPair.first.base() == globalBuff);
  assert(aPair.second.base() == globalBuff + 8);
  std::tuple<ForwardIter, ForwardIter> aTuple = a;
  assert(get<0>(aTuple).base() == globalBuff);
  assert(get<1>(aTuple).base() == globalBuff + 8);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
