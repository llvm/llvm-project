//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// class transform_view::<sentinel>;

#include <ranges>

#include "test_macros.h"
#include "../types.h"

template<class T>
concept EndIsIter = requires(T t) { ++t.end(); };

constexpr bool test() {
  std::ranges::transform_view<SizedSentinelView, PlusOne> transformView1;
  // Going to const and back.
  auto sent1 = transformView1.end();
  std::ranges::sentinel_t<const std::ranges::transform_view<SizedSentinelView, PlusOne>> sent2{sent1};
  std::ranges::sentinel_t<const std::ranges::transform_view<SizedSentinelView, PlusOne>> sent3{sent2};
  (void)sent3;

  static_assert(!EndIsIter<decltype(sent1)>);
  static_assert(!EndIsIter<decltype(sent2)>);
  assert(sent1.base() == globalBuff + 8);

  std::ranges::transform_view transformView2(SizedSentinelView{4}, PlusOne());
  auto sent4 = transformView2.end();
  auto iter = transformView1.begin();
  {
    assert(iter != sent1);
    assert(iter != sent2);
    assert(iter != sent4);
  }

  {
    assert(iter + 8 == sent1);
    assert(iter + 8 == sent2);
    assert(iter + 4 == sent4);
  }

  {
    assert(sent1 - iter  == 8);
    assert(sent4 - iter  == 4);
    assert(iter  - sent1 == -8);
    assert(iter  - sent4 == -4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
