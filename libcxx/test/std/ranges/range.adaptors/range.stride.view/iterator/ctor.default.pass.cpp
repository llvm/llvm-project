//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// ranges

// std::views::stride_view

#include "../test.h"
#include "__concepts/convertible_to.h"
#include "__ranges/access.h"
#include "__ranges/concepts.h"
#include "__ranges/stride_view.h"
#include <cassert>
#include <ranges>
#include <type_traits>

constexpr bool non_simple_view_iter_ctor_test() {
  using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentBegin<false>>;
  using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
  using NotSimpleStrideViewIterConst = std::ranges::iterator_t<const NotSimpleStrideView>;
  static_assert(!std::is_same_v<NotSimpleStrideViewIterConst, NotSimpleStrideViewIter>);
  return true;
}

struct NonDefaultConstructibleIterator : InputIterBase<NonDefaultConstructibleIterator> {
  NonDefaultConstructibleIterator() = delete;
  constexpr NonDefaultConstructibleIterator(int) {}
};

struct View : std::ranges::view_base {
  constexpr NonDefaultConstructibleIterator begin() const { return NonDefaultConstructibleIterator{5}; }
  constexpr std::default_sentinel_t end() const { return {}; }
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<View> = true;

constexpr bool iterator_default_constructible() {
  {
    // If the type of the iterator of the range being strided is non-default
    // constructible, then the stride view's iterator should not be default
    // constructible, either!
    constexpr View v{};
    constexpr auto stride   = std::ranges::stride_view(v, 1);
    using stride_iterator_t = decltype(stride.begin());
    static_assert(!std::is_default_constructible<stride_iterator_t>());
  }
  {
    // If the type of the iterator of the range being strided is default
    // constructible, then the stride view's iterator should be default
    // constructible, too!
    constexpr int arr[]     = {1, 2, 3};
    auto stride             = std::ranges::stride_view(arr, 1);
    using stride_iterator_t = decltype(stride.begin());
    static_assert(std::is_default_constructible<stride_iterator_t>());
  }

  return true;
}

constexpr bool non_const_iterator_copy_ctor() {
  {
    // Instantiate a stride view over a non-simple view whose const/non-const iterators are not-convertible.
    using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentBegin<false>>;
    using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
    using NotSimpleStrideViewConstIter = std::ranges::iterator_t<const NotSimpleStrideView>;

    // It should not be possible to construct a stride view iterator from a non-const stride view iterator
    // when the strided-over type has inconvertible iterator types.
    static_assert(!std::ranges::__simple_view<NotSimpleStrideView>);
    static_assert(!std::convertible_to<NotSimpleStrideViewIter, NotSimpleStrideViewConstIter>);
    static_assert(!std::is_constructible_v<NotSimpleStrideViewConstIter, NotSimpleStrideViewIter>);
  }
  {
    // Instantiate a stride view over a non-simple view whose const/non-const iterators are convertible.
    using NotSimpleStrideView          = std::ranges::stride_view<NotSimpleViewDifferentBegin<true>>;
    using NotSimpleStrideViewIter      = std::ranges::iterator_t<NotSimpleStrideView>;
    using NotSimpleStrideViewConstIter = std::ranges::iterator_t<const NotSimpleStrideView>;

    // It should be possible to construct a stride view iterator from a non-const stride view iterator
    // when the strided-over type has convertible iterator types.
    static_assert(!std::ranges::__simple_view<NotSimpleStrideView>);
    static_assert(std::convertible_to<NotSimpleStrideViewIter, NotSimpleStrideViewConstIter>);
    static_assert(std::is_constructible_v<NotSimpleStrideViewConstIter, NotSimpleStrideViewIter>);
  }
  return true;
}

int main(int, char**) {
  non_simple_view_iter_ctor_test();
  static_assert(non_simple_view_iter_ctor_test());
  static_assert(iterator_default_constructible());
  static_assert(non_const_iterator_copy_ctor());
  return 0;
}
