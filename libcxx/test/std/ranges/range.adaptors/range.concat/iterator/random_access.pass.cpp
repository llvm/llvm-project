//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto operator[](difference_type n) const requires
//        all_random_access<Const, Views...>

#include <ranges>
#include <cassert>

#include "../../range_adaptor_types.h"

constexpr bool test() {
  int buffer1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int buffer2[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // random_access_range and common range
    std::ranges::concat_view v(SimpleCommonRandomAccessSized{buffer1});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

    static_assert(std::is_same_v<decltype(it[2]), const int&>);
  }

  {
    // random_access_range and common range, with last view is not a common range
    std::ranges::concat_view v(SimpleCommonRandomAccessSized{buffer1}, SimpleNonCommonRandomAccessSized{buffer2});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

    static_assert(std::is_same_v<decltype(it[2]), const int&>);
  }

  {
    // random_access_range and non common range
    std::ranges::concat_view v(SimpleNonCommonRandomAccessSized{buffer1}, NonSimpleCommonRandomAccessSized{buffer2});
    auto it                 = v.begin();
    const auto canSubscript = [](auto&& jt) { return requires { jt[0]; }; };
    static_assert(!canSubscript(it));

    static_assert(std::is_same_v<decltype(*it), const int&>);
  }

  {
    // contiguous_range
    std::ranges::concat_view v(ContiguousCommonView{buffer1}, ContiguousCommonView{buffer2});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

    static_assert(std::is_same_v<decltype(it[2]), int&>);
  }

  {
    // non random_access_range
    std::ranges::concat_view v(BidiCommonView{buffer1});
    auto iter               = v.begin();
    const auto canSubscript = [](auto&& it) { return requires { it[0]; }; };
    static_assert(!canSubscript(iter));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
