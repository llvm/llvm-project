//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<viewable_range R>
// using all_t = decltype(views::all(declval<R>()));

#include <ranges>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  int *begin() const;
  int *end() const;
};

struct Range {
  int *begin() const;
  int *end() const;
};

struct BorrowableRange {
  int *begin() const;
  int *end() const;
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowableRange> = true;

template <class T>
concept HasAllT = requires {
    typename std::views::all_t<T>;
};

// When T is a view, returns decay-copy(T)
ASSERT_SAME_TYPE(std::views::all_t<View>, View);
ASSERT_SAME_TYPE(std::views::all_t<View&>, View);
ASSERT_SAME_TYPE(std::views::all_t<View&&>, View);
ASSERT_SAME_TYPE(std::views::all_t<const View>, View);
ASSERT_SAME_TYPE(std::views::all_t<const View&>, View);
ASSERT_SAME_TYPE(std::views::all_t<const View&&>, View);

// Otherwise, when T is a reference to a range, returns ref_view<T>
ASSERT_SAME_TYPE(std::views::all_t<Range&>, std::ranges::ref_view<Range>);
ASSERT_SAME_TYPE(std::views::all_t<const Range&>, std::ranges::ref_view<const Range>);
ASSERT_SAME_TYPE(std::views::all_t<BorrowableRange&>, std::ranges::ref_view<BorrowableRange>);
ASSERT_SAME_TYPE(std::views::all_t<const BorrowableRange&>, std::ranges::ref_view<const BorrowableRange>);

// Otherwise, returns owning_view<T>
ASSERT_SAME_TYPE(std::views::all_t<Range>, std::ranges::owning_view<Range>);
ASSERT_SAME_TYPE(std::views::all_t<Range&&>, std::ranges::owning_view<Range>);
static_assert(!HasAllT<const Range>);
static_assert(!HasAllT<const Range&&>);
ASSERT_SAME_TYPE(std::views::all_t<BorrowableRange>, std::ranges::owning_view<BorrowableRange>);
ASSERT_SAME_TYPE(std::views::all_t<BorrowableRange&&>, std::ranges::owning_view<BorrowableRange>);
static_assert(!HasAllT<const BorrowableRange>);
static_assert(!HasAllT<const BorrowableRange&&>);
