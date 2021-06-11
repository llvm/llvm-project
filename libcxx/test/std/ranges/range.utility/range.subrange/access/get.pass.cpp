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

template<size_t I, class S>
concept GetInvocable = requires {
  std::get<I>(std::declval<S>());
};

static_assert( GetInvocable<0, std::ranges::subrange<int*>>);
static_assert( GetInvocable<1, std::ranges::subrange<int*>>);
static_assert(!GetInvocable<2, std::ranges::subrange<int*>>);
static_assert(!GetInvocable<3, std::ranges::subrange<int*>>);

constexpr bool test() {
  std::ranges::subrange<int*> a(globalBuff, globalBuff + 8, 8);
  assert(std::get<0>(a) == a.begin());
  assert(std::get<1>(a) == a.end());

  assert(a.begin() == std::get<0>(std::move(a)));
  std::ranges::subrange<int*> b(globalBuff, globalBuff + 8, 8);
  assert(b.end() == std::get<1>(std::move(b)));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
