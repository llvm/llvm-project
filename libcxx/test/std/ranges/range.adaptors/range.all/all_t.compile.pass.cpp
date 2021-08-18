//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<viewable_range R>
// using all_t = decltype(views::all(declval<R>()));

#include <ranges>

#include "test_iterators.h"
#include "test_range.h"

struct View : test_range<cpp20_input_iterator>, std::ranges::view_base { };
struct Range : test_range<cpp20_input_iterator> { };
struct BorrowableRange : test_range<forward_iterator> { };
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowableRange> = true;

// When T is a view, returns decay-copy(T)
ASSERT_SAME_TYPE(std::views::all_t<View>, View);
ASSERT_SAME_TYPE(std::views::all_t<View&>, View);
ASSERT_SAME_TYPE(std::views::all_t<View const>, View);
ASSERT_SAME_TYPE(std::views::all_t<View const&>, View);

// Otherwise, when T is a reference to a range, returns ref_view<T>
ASSERT_SAME_TYPE(std::views::all_t<Range&>, std::ranges::ref_view<Range>);
ASSERT_SAME_TYPE(std::views::all_t<Range const&>, std::ranges::ref_view<Range const>);
ASSERT_SAME_TYPE(std::views::all_t<BorrowableRange&>, std::ranges::ref_view<BorrowableRange>);
ASSERT_SAME_TYPE(std::views::all_t<BorrowableRange const&>, std::ranges::ref_view<BorrowableRange const>);

// Otherwise, returns subrange<iterator_t<T>, sentinel_t<R>>
ASSERT_SAME_TYPE(std::views::all_t<BorrowableRange>, std::ranges::subrange<forward_iterator<int*>, sentinel>);
ASSERT_SAME_TYPE(std::views::all_t<BorrowableRange const>, std::ranges::subrange<forward_iterator<int const*>, sentinel>);
