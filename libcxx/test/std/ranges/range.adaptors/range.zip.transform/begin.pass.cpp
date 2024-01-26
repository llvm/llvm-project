//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto begin();
// constexpr auto begin() const
//   requires range<const InnerView> &&
//            regular_invocable<const F&, range_reference_t<const Views>...>;

#include <ranges>

#include <cassert>
#include <concepts>
#include <tuple>
#include <utility>

#include "types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // all underlying iterators should be at the begin position
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, std::views::iota(0), std::ranges::single_view(2.));
    auto it = v.begin();
    assert(*it == std::make_tuple(1, 0, 2.0));

    auto const_it = std::as_const(v).begin();
    assert(*const_it == *it);

    static_assert(!std::same_as<decltype(it), decltype(const_it)>);
  }

  {
    // with empty range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer}, std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
    assert(std::as_const(v).begin() == std::as_const(v).end());
  }

  {
    // underlying const R is not a range
    using View = std::ranges::zip_transform_view<MakeTuple, SimpleCommon, NoConstBeginView>;
    static_assert(HasBegin<View>);
    static_assert(!HasConstBegin<View>);
  }

  {
    // Fn cannot be invoked on const range
    using View = std::ranges::zip_transform_view<NonConstOnlyFn, ConstNonConstDifferentView>;
    static_assert(HasBegin<View>);
    static_assert(!HasConstBegin<View>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
