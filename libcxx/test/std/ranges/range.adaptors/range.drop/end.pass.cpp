//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto end()
//   requires (!simple-view<V>)
// constexpr auto end() const
//   requires range<const V>

#include <ranges>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  // range<const V>
  std::ranges::drop_view dropView1(MoveOnlyView(), 4);
  assert(dropView1.end() == globalBuff + 8);

  // !simple-view<V>
  std::ranges::drop_view dropView2(InputView(), 4);
  assert(dropView2.end() == globalBuff + 8);

  // range<const V>
  const std::ranges::drop_view dropView3(MoveOnlyView(), 0);
  assert(dropView3.end() == globalBuff + 8);

  // !simple-view<V>
  const std::ranges::drop_view dropView4(InputView(), 2);
  assert(dropView4.end() == globalBuff + 8);

  // range<const V>
  std::ranges::drop_view dropView5(MoveOnlyView(), 10);
  assert(dropView5.end() == globalBuff + 8);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
