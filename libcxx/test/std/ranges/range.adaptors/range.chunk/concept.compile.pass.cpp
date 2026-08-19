//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   template <view V>
//     requires input_range<V>
//   class chunk_view;
//
//   template <view V>
//     requires forward_range<V>
//   class chunk_view<_View>;

#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_range.h"

template <>
inline constexpr bool std::ranges::enable_view<InputRangeNotDerivedFrom> = true;

// Test two separate properties:
// 1. Whether the type `chunk_view<View>` can be formed.
// 2. Whether a well-formed `chunk_view<View>` can be constructed.
template <class View>
concept CanFormChunkView = requires { typename std::ranges::chunk_view<View>; };

template <class View>
concept CanConstructChunkView = CanFormChunkView<View> && requires(View view, std::ranges::range_difference_t<View> n) {
  std::ranges::chunk_view<View>(std::move(view), n);
};

// Test constraints when the template argument is not a view
static_assert(!std::ranges::view<test_non_const_range<cpp17_input_iterator>>);
static_assert(std::ranges::input_range<test_non_const_range<cpp17_input_iterator>>);
static_assert(!CanFormChunkView<test_non_const_range<cpp17_input_iterator>>);

// Test constraints when the template argument is not an input_range
static_assert(!std::ranges::input_range<InputRangeNotDerivedFrom>);
static_assert(std::ranges::view<InputRangeNotDerivedFrom>);
static_assert(!CanFormChunkView<InputRangeNotDerivedFrom>);

// Test constraints when the template argument is an input_range and a view
static_assert(std::ranges::input_range<
              std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>);
static_assert(std::ranges::view<
              std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>);
static_assert(CanFormChunkView<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>> &&
              CanConstructChunkView<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>);

// Test constraints when the template argument is a forward_range and a view
static_assert(std::ranges::forward_range<BorrowedView>);
static_assert(std::ranges::view<BorrowedView>);
static_assert(CanFormChunkView<BorrowedView> && CanConstructChunkView<BorrowedView>);

// chunk_view itself models view
static_assert(std::ranges::view<std::ranges::chunk_view<
                  std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>>);
static_assert(std::ranges::view<std::ranges::chunk_view<BorrowedView>>);
