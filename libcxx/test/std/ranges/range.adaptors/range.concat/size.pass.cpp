//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto size()
//     requires(sized_range<_Views> && ...)

// constexpr auto size() const
//     requires(sized_range<const _Views> && ...)

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "types.h"
#include "../range_adaptor_types.h"

int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
struct View : std::ranges::view_base {
  std::size_t size_ = 0;
  constexpr View(std::size_t s) : size_(s) {}
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

struct StrangeSizeView : std::ranges::view_base {
  constexpr auto begin() const { return buffer; }
  constexpr auto end() const { return buffer + 8; }

  constexpr auto size() { return 5; }
  constexpr auto size() const { return 6; }
};

struct NoSizeView : std::ranges::view_base {
  constexpr auto begin() const { return buffer; }
  constexpr auto end() const { return buffer + 8; }
};

struct IntSizeView : std::ranges::view_base {
  using iterator = forward_iterator<int*>;

  constexpr IntSizeView() {}
  constexpr auto begin() const { return iterator{buffer}; }
  constexpr auto end() const { return iterator{buffer + 9}; }
  constexpr int size() const { return 9; }
};

struct UnsignedSizeView : std::ranges::view_base {
  using iterator = forward_iterator<int*>;

  constexpr UnsignedSizeView() {}
  constexpr auto begin() const { return iterator{buffer}; }
  constexpr auto end() const { return iterator{buffer + 9}; }
  constexpr unsigned int size() const { return 9; }
};

constexpr bool test() {
  {
    // single range
    std::ranges::concat_view v(View(8));
    assert(v.size() == 8);
    assert(std::as_const(v).size() == 8);
  }

  {
    // multiple ranges same type
    std::ranges::concat_view v(View(2), View(3));
    assert(v.size() == 5);
    assert(std::as_const(v).size() == 5);
  }

  {
    // multiple ranges different types
    std::ranges::concat_view v(std::views::iota(0, 5), View(3));
    assert(v.size() == 8);
    assert(std::as_const(v).size() == 8);
  }

  {
    // const-view non-sized range
    std::ranges::concat_view v(SizedNonConst(2), View(3));
    assert(v.size() == 5);
    static_assert(std::ranges::sized_range<decltype(v)>);
    static_assert(!std::ranges::sized_range<decltype(std::as_const(v))>);
  }

  {
    // const/non-const has different sizes
    std::ranges::concat_view v(StrangeSizeView{});
    assert(v.size() == 5);
    assert(std::as_const(v).size() == 6);
  }

  {
    // underlying range not sized
    std::ranges::concat_view v(InputCommonView{buffer});
    static_assert(!std::ranges::sized_range<decltype(v)>);
    static_assert(!std::ranges::sized_range<decltype(std::as_const(v))>);
  }

  {
    //two ranges with different size type
    std::ranges::concat_view v(UnsignedSizeView{}, IntSizeView{});
    assert(v.size() == 18);
    // common type between size_t and int should be size_t
    ASSERT_SAME_TYPE(decltype(v.size()), unsigned int);
  }

  {
    // three ranges with different size type
    std::ranges::concat_view v(UnsignedSizeView{}, IntSizeView{}, StrangeSizeView{});
    assert(v.size() == 23);
    // common type between size_t and int should be size_t
    ASSERT_SAME_TYPE(decltype(v.size()), unsigned int);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
