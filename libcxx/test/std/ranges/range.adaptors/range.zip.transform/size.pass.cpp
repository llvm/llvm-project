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
    // single range
    std::ranges::zip_transform_view v(MakeTuple{}, SizedView(8));
    assert(v.size() == 8);
    assert(std::as_const(v).size() == 8);
  }

  {
    // multiple ranges
    std::ranges::zip_transform_view v(MakeTuple{}, SizedView(2), SizedView(3));
    assert(v.size() == 2);
    assert(std::as_const(v).size() == 2);
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
