//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  constexpr decltype(auto) operator[](difference_type n) const
//    requires random_access_range<Base>;

#include <ranges>
#include <cassert>

#include "../types.h"

template <class T>
concept CanSubscript = requires(T t) { t[0]; };

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    // F returns PR value
    std::ranges::zip_transform_view v(MakeTuple{}, SizedRandomAccessView{buffer}, std::views::iota(0));
    auto it    = v.begin();
    using Iter = decltype(it);
    static_assert(CanSubscript<Iter>);

    std::same_as<std::tuple<int, int>> decltype(auto) val = it[0];
    assert(val == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));
  }

  {
    // F return by reference
    std::ranges::zip_transform_view v(GetFirst{}, ContiguousCommonView{buffer}, ContiguousCommonView{buffer});
    auto it    = v.begin();
    using Iter = decltype(it);
    static_assert(CanSubscript<Iter>);

    std::same_as<int&> decltype(auto) val = it[0];
    assert(&val == &buffer[0]);
    assert(val == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));
  }

  {
    // non random_access_range
    std::ranges::zip_transform_view v(GetFirst{}, BidiCommonView{buffer});
    auto it    = v.begin();
    using Iter = decltype(it);
    static_assert(!CanSubscript<Iter>);
  }

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    auto it = v.begin();
    assert(it[2] == std::make_tuple(3));
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, std::views::iota(0));
    auto it = v.begin();
    assert(it[0] == 1);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(Tie{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::repeat_view(2.));
    auto it = v.begin();
    assert(&std::get<0>(it[1]) == &buffer[1]);
    assert(&std::get<1>(it[2]) == &buffer[2]);
    assert(std::get<2>(it[3]) == 2.0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
