//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr __iterator<false> begin()
//     requires(!(__simple_view<_Views> && ...))

// constexpr __iterator<true> begin() const
//     requires((range<const _Views> && ...) && __concatable<const _Views...>)

#include <array>
#include <ranges>
#include <vector>

#include <cassert>
#include "test_iterators.h"
#include "types.h"
#include "../range_adaptor_types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin = HasConstBegin<T> && requires(T& t, const T& ct) {
  requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>;
};

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

constexpr bool test() {
  // check the case of simple view
  {
    int buffer[4] = {1, 2, 3, 4};
    std::ranges::concat_view v(SimpleCommon{buffer}, SimpleCommon{buffer});
    static_assert(std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());
    assert(*v.begin() == buffer[0]);
    assert(*std::as_const(v).begin() == buffer[0]);

    using View = decltype(v);
    static_assert(HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(!HasConstAndNonConstBegin<View>);
  }

  // not all underlying ranges model simple view
  {
    int buffer[4] = {1, 2, 3, 4};
    std::ranges::concat_view v(SimpleCommon{buffer}, NonSimpleNonCommon{buffer});
    static_assert(!std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());
    assert(*v.begin() == buffer[0]);
    assert(*std::as_const(v).begin() == buffer[0]);

    using View = decltype(v);
    static_assert(!HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(HasConstAndNonConstBegin<View>);
  }

  // first view is empty
  {
    std::vector<int> v1;
    std::vector<int> v2 = {1, 2, 3, 4};
    std::ranges::concat_view view(v1, v2);
    auto it = view.begin();
    assert(*it == 1);
    assert(it + 4 == view.end());
  }

  // first few views is empty, including different types
  {
    std::vector<int> v1;
    std::array<int, 0> v2;
    std::vector<int> v3 = {1, 2, 3, 4};
    std::ranges::concat_view view(v1, v2, v3);
    auto it = view.begin();
    assert(*it == 1);
    assert(it + 4 == view.end());
  }

  // all views are empty
  {
    std::array<int, 0> arr;
    std::vector<int> v;
    std::ranges::concat_view(arr, v);
    assert(v.begin() == v.end());
  }

  // testing concatable constraint
  {
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
