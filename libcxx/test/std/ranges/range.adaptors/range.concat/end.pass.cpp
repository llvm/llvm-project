//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto end()
//   requires(!(__simple_view<_Views> && ...))

// constexpr auto end() const
//   requires((range<const _Views> && ...) && __concatable<const _Views...>)

#include <array>
#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "types.h"
#include "../range_adaptor_types.h"

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

  {
    // all the ranges but the last one are input ranges, the last range is common => end() returns sentinel
    // https://cplusplus.github.io/LWG/issue4166

    using Iter      = cpp20_input_iterator<const int*>;
    using Sentinel  = sentinel_wrapper<Iter>;
    using InputView = minimal_view<Iter, Sentinel>;

    std::array<int, 3> arr{1, 2, 3};
    auto v = std::views::concat(InputView{Iter{buffer1}, Sentinel{Iter{buffer1 + 5}}}, arr);
    static_assert(std::same_as<decltype(v.end()), std::default_sentinel_t>);
    auto it = v.begin();
    it++;
    it++;
    it++;
    it++;
    it++;
    it++;
    it++;
    it++;
    assert(it == v.end());
  }

  {
    // first range is forward range, the last range is common => end() does not return sentinel
    using Iter        = forward_iterator<const int*>;
    using Sentinel    = sentinel_wrapper<Iter>;
    using ForwardView = minimal_view<Iter, Sentinel>;

    std::array<int, 2> arr{6, 7};
    auto v = std::views::concat(ForwardView{Iter{buffer1}, Sentinel{Iter{buffer1 + 5}}}, arr);
    static_assert(!std::same_as<decltype(v.end()), std::default_sentinel_t>);
    static_assert(std::same_as<decltype(v.end()), decltype(v.begin())>);

    auto it   = v.begin();
    auto last = v.end();
    int sum   = 0;
    for (; it != last; it++) {
      sum += *it;
    }
    assert(sum == 1 + 2 + 3 + 4 + 5 + 6 + 7);
  }

  {
    // testing concatable constraint
    static_assert(!ConcatableConstViews<ViewWithNoConstBegin>);
    static_assert(ConcatableConstViews<ViewWithConstBegin>);
    static_assert(!ConcatableConstViews<ViewWithNoConstBegin, ViewWithConstBegin>);
    static_assert(!ConcatableConstViews<ViewWithNoConstBegin, ViewWithConstBegin, SizedViewWithConstBegin>);
    static_assert(ConcatableConstViews<ViewWithConstBegin, SizedViewWithConstBegin>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
