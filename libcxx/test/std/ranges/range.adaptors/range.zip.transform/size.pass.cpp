//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto size() requires sized_range<InnerView>
// constexpr auto size() const requires sized_range<const InnerView>

#include <ranges>

#include <cassert>

#include "test_iterators.h"
#include "types.h"

int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
struct SizedView : std::ranges::view_base {
  std::size_t size_ = 0;
  constexpr SizedView(std::size_t s) : size_(s) {}
  constexpr auto begin() const { return buffer; }
  constexpr auto end() const { return buffer + size_; }
};

struct SizedNonConst : std::ranges::view_base {
  using iterator    = forward_iterator<int*>;
  std::size_t size_ = 0;
  constexpr SizedNonConst(std::size_t s) : size_(s) {}
  constexpr auto begin() const { return iterator{buffer}; }
  constexpr auto end() const { return iterator{buffer + size_}; }
  constexpr std::size_t size() { return size_; }
};

struct ConstNonConstDifferentSize : std::ranges::view_base {
  constexpr auto begin() const { return buffer; }
  constexpr auto end() const { return buffer + 8; }

  constexpr auto size() { return 5; }
  constexpr auto size() const { return 6; }
};

constexpr bool test() {
  {
    // one range
    std::ranges::zip_transform_view v(MakeTuple{}, SimpleCommon{buffer});
    assert(v.size() == 9);
    assert(std::as_const(v).size() == 9);
  }

  {
    // two ranges
    std::ranges::zip_transform_view v(GetFirst{}, SimpleCommon{buffer}, SizedView(3));
    assert(v.size() == 3);
    assert(std::as_const(v).size() == 3);
  }

  {
    // three ranges
    std::ranges::zip_transform_view v(Tie{}, SimpleCommon{buffer}, SizedView{6}, std::ranges::single_view(2.));
    assert(v.size() == 1);
    assert(std::as_const(v).size() == 1);
  }

  {
    // single empty range
    std::ranges::zip_transform_view v(MakeTuple{}, std::ranges::empty_view<int>());
    assert(v.size() == 0);
    assert(std::as_const(v).size() == 0);
  }

  {
    // empty range at the beginning
    std::ranges::zip_transform_view v(
        MakeTuple{}, std::ranges::empty_view<int>(), SimpleCommon{buffer}, SimpleCommon{buffer});
    assert(v.size() == 0);
    assert(std::as_const(v).size() == 0);
  }

  {
    // empty range in the middle
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, std::ranges::empty_view<int>(), SimpleCommon{buffer});
    assert(v.size() == 0);
    assert(std::as_const(v).size() == 0);
  }

  {
    // empty range at the end
    std::ranges::zip_transform_view v(
        MakeTuple{}, SimpleCommon{buffer}, SimpleCommon{buffer}, std::ranges::empty_view<int>());
    assert(v.size() == 0);
    assert(std::as_const(v).size() == 0);
  }

  {
    // const-view non-sized range
    std::ranges::zip_transform_view v(MakeTuple{}, SizedNonConst(2), SizedView(3));
    assert(v.size() == 2);
    static_assert(std::ranges::sized_range<decltype(v)>);
    static_assert(!std::ranges::sized_range<decltype(std::as_const(v))>);
  }

  {
    // const/non-const has different sizes
    std::ranges::zip_transform_view v(MakeTuple{}, ConstNonConstDifferentSize{});
    assert(v.size() == 5);
    assert(std::as_const(v).size() == 6);
  }

  {
    // underlying range not sized
    std::ranges::zip_transform_view v(MakeTuple{}, InputCommonView{buffer});
    static_assert(!std::ranges::sized_range<decltype(v)>);
    static_assert(!std::ranges::sized_range<decltype(std::as_const(v))>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
