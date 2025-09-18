//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator(iterator<!Const> i)
//     requires Const && convertible_to<ziperator<false>, ziperator<Const>>;

#include <array>
#include <ranges>

#include <cassert>

#include "../types.h"

using ConstIterIncompatibleView =
    BasicView<forward_iterator<int*>,
              forward_iterator<int*>,
              random_access_iterator<const int*>,
              random_access_iterator<const int*>>;
static_assert(!std::convertible_to<std::ranges::iterator_t<ConstIterIncompatibleView>,
                                   std::ranges::iterator_t<const ConstIterIncompatibleView>>);

constexpr bool test() {
  int buffer[3] = {1, 2, 3};

  {
    std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleCommon{buffer});
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    assert(iter1 == iter2);

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);

    // We cannot create a non-const iterator from a const iterator.
    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
  }

  {
    // Check when we can't perform a non-const-to-const conversion of the ziperator
    std::ranges::zip_transform_view v(MakeTuple{}, ConstIterIncompatibleView{buffer});
    auto iter1 = v.begin();
    auto iter2 = std::as_const(v).begin();

    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);

    static_assert(!std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!std::constructible_from<decltype(iter2), decltype(iter1)>);
  }

  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, NonSimpleCommon{buffer});
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(*iter2 == std::tuple(1));
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, NonSimpleCommon{buffer}, std::views::iota(0));
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(*iter2 == 1);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(
        Tie{}, NonSimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::single_view(2.));
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(*iter2 == std::tuple(1, 1, 2.0));
  }

  {
    // single empty range
    std::array<int, 0> buffer2{};
    std::ranges::zip_transform_view v(MakeTuple{}, buffer2);
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(iter2 == v.end());
  }

  {
    // empty range at the beginning
    std::ranges::zip_transform_view v(
        MakeTuple{}, std::ranges::empty_view<int>(), NonSimpleCommon{buffer}, SimpleCommon{buffer});
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(iter2 == v.end());
  }

  {
    // empty range in the middle
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, std::ranges::empty_view<int>(), NonSimpleCommon{buffer});
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(iter2 == v.end());
  }

  {
    // empty range at the end
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, NonSimpleCommon{buffer}, std::ranges::empty_view<int>());
    auto iter1                                       = v.begin();
    std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    static_assert(!std::is_same_v<decltype(iter1), decltype(iter2)>);
    assert(iter2 == v.end());
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
