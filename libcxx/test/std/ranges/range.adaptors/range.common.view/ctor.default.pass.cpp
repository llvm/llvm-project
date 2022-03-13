//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// common_view() requires default_initializable<V> = default;

#include <ranges>

#include <cassert>
#include <concepts>

#include "test_iterators.h"
#include "types.h"

int main(int, char**) {
  static_assert(!std::default_initializable<std::ranges::common_view<MoveOnlyView>>);
  static_assert( std::default_initializable<std::ranges::common_view<DefaultConstructibleView>>);

  std::ranges::common_view<DefaultConstructibleView> common;
  assert(common.begin() == nullptr);
  assert(common.end() == nullptr);

  return 0;
}
