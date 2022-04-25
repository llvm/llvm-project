//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto begin() requires (!(simple-view<Views> && ...));
// constexpr auto begin() const requires (range<const Views> && ...);

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

template <class T>
concept HasConstAndNonConstBegin =
    HasConstBegin<T> &&
    requires(T& t, const T& ct) { requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>; };

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !
HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !
HasConstAndNonConstBegin<T>;

struct NoConstBeginView : std::ranges::view_base {
  int* begin();
  int* end();
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    // all underlying iterators should be at the begin position
    std::ranges::zip_view v(SizedRandomAccessView{buffer}, std::views::iota(0), std::ranges::single_view(2.));
    std::same_as<std::tuple<int&, int, double&>> decltype(auto) val = *v.begin();
    assert(val == std::make_tuple(1, 0, 2.0));
    assert(&(std::get<0>(val)) == &buffer[0]);
  }

  {
    // with empty range
    std::ranges::zip_view v(SizedRandomAccessView{buffer}, std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
  }

  {
    // underlying ranges all model simple-view
    std::ranges::zip_view v(SimpleCommon{buffer}, SimpleCommon{buffer});
    static_assert(std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());
    auto [x, y] = *std::as_const(v).begin();
    assert(&x == &buffer[0]);
    assert(&y == &buffer[0]);

    using View = decltype(v);
    static_assert(HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(!HasConstAndNonConstBegin<View>);
  }

  {
    // not all underlying ranges model simple-view
    std::ranges::zip_view v(SimpleCommon{buffer}, NonSimpleNonCommon{buffer});
    static_assert(!std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());
    auto [x, y] = *std::as_const(v).begin();
    assert(&x == &buffer[0]);
    assert(&y == &buffer[0]);

    using View = decltype(v);
    static_assert(!HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(HasConstAndNonConstBegin<View>);
  }

  {
    // underlying const R is not a range
    using View = std::ranges::zip_view<SimpleCommon, NoConstBeginView>;
    static_assert(!HasOnlyConstBegin<View>);
    static_assert(HasOnlyNonConstBegin<View>);
    static_assert(!HasConstAndNonConstBegin<View>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
