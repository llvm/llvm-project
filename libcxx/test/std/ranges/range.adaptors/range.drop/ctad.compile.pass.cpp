//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class R>
//   drop_view(R&&, range_difference_t<R>) -> drop_view<views::all_t<R>>;

#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

struct View : std::ranges::view_base {
  int *begin() const;
  int *end() const;
};

struct Range {
  int *begin() const;
  int *end() const;
};

struct BorrowedRange {
  int *begin() const;
  int *end() const;
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange> = true;

void testCTAD() {
    View v;
    Range r;
    BorrowedRange br;

    static_assert(std::same_as<
        decltype(std::ranges::drop_view(v, 0)),
        std::ranges::drop_view<View>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::drop_view(std::move(v), 0)),
        std::ranges::drop_view<View>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::drop_view(r, 0)),
        std::ranges::drop_view<std::ranges::ref_view<Range>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::drop_view(std::move(r), 0)),
        std::ranges::drop_view<std::ranges::owning_view<Range>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::drop_view(br, 0)),
        std::ranges::drop_view<std::ranges::ref_view<BorrowedRange>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::drop_view(std::move(br), 0)),
        std::ranges::drop_view<std::ranges::owning_view<BorrowedRange>>
    >);
}
