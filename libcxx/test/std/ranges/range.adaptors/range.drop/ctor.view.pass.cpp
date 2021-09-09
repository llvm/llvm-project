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

// constexpr drop_view(V base, range_difference_t<V> count);

#include <ranges>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  std::ranges::drop_view dropView1(ContiguousView(), 4);
  assert(dropView1.size() == 4);
  assert(dropView1.begin() == globalBuff + 4);

  std::ranges::drop_view dropView2(ForwardView(), 4);
  assert(dropView2.begin().base() == globalBuff + 4);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
