//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <utility>

// inline constexpr auto __for_each_index_sequence = []<size_t... _Index>(index_sequence<_Index...>, auto __func)

#include <utility>
#include <cassert>

#include "test_macros.h"

constexpr bool test() {
  int count = 0;
  std::__for_each_index_sequence(std::make_index_sequence<8>(), [&]<std::size_t _Index> { count += _Index; });
  assert(count == 28);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
