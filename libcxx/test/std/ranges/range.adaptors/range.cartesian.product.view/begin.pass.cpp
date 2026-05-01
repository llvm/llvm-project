//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr iterator<false> begin()       requires (!simple-view<First> || ... || !simple-view<Vs>);
// constexpr iterator<true>  begin() const requires (range<const First> && ... && range<const Vs>);

#include <cassert>
#include <concepts>
#include <ranges>
#include <tuple>
#include <utility>

#include "../range_adaptor_types.h"

template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasNonConstBegin = requires(T& t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin = HasConstBegin<T> && HasNonConstBegin<T> && requires(T& t, const T& ct) {
  requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>;
};

template <class T>
concept HasOnlyNonConstBegin = HasNonConstBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

struct NoConstBeginView : std::ranges::view_base {
  int* begin();
  int* end();
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  { // all underlying iterators are at the begin position
    std::ranges::cartesian_product_view v(
        SizedRandomAccessView{buffer}, std::views::iota(0, 5), std::ranges::single_view(2.0));
    std::same_as<std::tuple<int&, int, double&>> decltype(auto) val = *v.begin();
    assert(val == std::make_tuple(1, 0, 2.0));
    assert(&(std::get<0>(val)) == &buffer[0]);
  }

  { // empty inner range: begin() == end()
    std::ranges::cartesian_product_view v(SizedRandomAccessView{buffer}, std::ranges::empty_view<int>());
    assert(v.begin() == v.end());
    assert(std::as_const(v).begin() == std::as_const(v).end());
  }

  { // empty outer range: begin() == end()
    std::ranges::cartesian_product_view v(std::ranges::empty_view<int>(), SizedRandomAccessView{buffer});
    assert(v.begin() == v.end());
  }

  { // empty middle of a 3-range product: begin() == end()
    std::ranges::cartesian_product_view v(
        SizedRandomAccessView{buffer}, std::ranges::empty_view<int>(), SizedRandomAccessView{buffer});
    assert(v.begin() == v.end());
  }

  { // simple-view: const and non-const begin() return the same type
    std::ranges::cartesian_product_view v(SimpleCommon{buffer}, SimpleCommon{buffer});
    static_assert(std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());
    auto [x, y] = *std::as_const(v).begin();
    assert(&x == &buffer[0]);
    assert(&y == &buffer[0]);

    using View = decltype(v);
    static_assert(std::ranges::__simple_view<SimpleCommon>);
    static_assert(HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(!HasConstAndNonConstBegin<View>);
  }

  { // not all underlying ranges are simple-view: const and non-const begin() differ
    std::ranges::cartesian_product_view v(SimpleCommon{buffer}, NonSimpleNonCommon{buffer});
    static_assert(!std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());

    using View = decltype(v);
    static_assert(!HasOnlyConstBegin<View>);
    static_assert(!HasOnlyNonConstBegin<View>);
    static_assert(HasConstAndNonConstBegin<View>);
  }

  { // const-of-underlying is not a range: only non-const begin() is available
    using View = std::ranges::cartesian_product_view<SimpleCommon, NoConstBeginView>;
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
