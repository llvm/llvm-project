//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <cassert>
#include "test_iterators.h"
#include "types.h"

constexpr bool test() {
  int buffer1[5] = {1, 2, 3, 4, 5};
  int buffer2[2] = {6, 7};

  {
    std::ranges::concat_view v(SimpleCommon{buffer1}, SimpleCommon{buffer2});
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
    static_assert(std::ranges::common_range<decltype(v)>);
    assert(v.begin() + 7 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }

  {
    std::ranges::concat_view v(SimpleCommon{buffer1}, SimpleNonCommon{buffer2});
    assert(v.begin() + 7 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
    static_assert(std::is_same_v<decltype(v.end()), std::default_sentinel_t>);
  }

  {
    std::ranges::concat_view v(SimpleCommon{buffer1}, NonSimpleCommon{buffer2});
    assert(v.begin() + 7 == v.end());
    static_assert(!std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
  }

  {
    std::ranges::concat_view v(SimpleCommon{buffer1}, NonSimpleNonCommon{buffer2});
    assert(v.begin() + 7 == v.end());
    static_assert(std::is_same_v<decltype(v.end()), decltype(std::as_const(v).end())>);
    static_assert(std::is_same_v<decltype(v.end()), std::default_sentinel_t>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
