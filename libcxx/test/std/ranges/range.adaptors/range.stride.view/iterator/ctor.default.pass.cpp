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
#include <cassert>
#include <ranges>
#include <type_traits>

constexpr bool non_simple_view_iter_ctor_test() {
  using NotSimpleStrideView     = std::ranges::stride_view<NotSimpleView>;
  using NotSimpleStrideViewIter = std::ranges::iterator_t<NotSimpleStrideView>;

  using SimpleStrideView     = std::ranges::stride_view<ForwardTracedMoveView>;
  using SimpleStrideViewIter = std::ranges::iterator_t<SimpleStrideView>;

  NotSimpleStrideView nsv{NotSimpleView{}, 1};
  [[maybe_unused]] NotSimpleStrideViewIter nsv_iter = nsv.begin();

  SimpleStrideView sv{ForwardTracedMoveView{}, 1};
  [[maybe_unused]] SimpleStrideViewIter ssv_iter = sv.begin();

  using NotSimpleStrideViewIterConst = std::ranges::iterator_t<const NotSimpleStrideView>;
  using SimpleStrideViewIterConst    = std::ranges::iterator_t<const SimpleStrideView>;

  // .begin on a stride view over a non-simple view will give us a
  // stride_view iterator with its _Const == false. Compare that type
  // with an iterator on a stride view over a simple view that will give
  // us an iterator with its _Const == true. They should *not* be the same.
  static_assert(!std::is_same_v<decltype(ssv_iter), decltype(nsv_iter)>);
  static_assert(!std::is_same_v<NotSimpleStrideViewIterConst, decltype(nsv_iter)>);
  static_assert(std::is_same_v<SimpleStrideViewIterConst, decltype(ssv_iter)>);
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
    static_assert(!std::is_default_constructible<stride_iterator_t>(), "");
  }
  {
    // If the type of the iterator of the range being strided is default
    // constructible, then the stride view's iterator should be default
    // constructible, too!
    constexpr int arr[]     = {1, 2, 3};
    auto stride             = std::ranges::stride_view(arr, 1);
    using stride_iterator_t = decltype(stride.begin());
    static_assert(std::is_default_constructible<stride_iterator_t>(), "");
  }

  return true;
}

int main(int, char**) {
  non_simple_view_iter_ctor_test();
  static_assert(non_simple_view_iter_ctor_test());
  static_assert(iterator_default_constructible());
  return 0;
}
