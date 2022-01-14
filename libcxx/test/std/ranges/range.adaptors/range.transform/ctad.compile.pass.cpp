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

// template<class R, class F>
//   transform_view(R&&, F) -> transform_view<views::all_t<R>, F>;

#include <ranges>
#include <cassert>
#include <concepts>

struct PlusOne {
    int operator()(int x) const;
};

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
    PlusOne f;

    static_assert(std::same_as<
        decltype(std::ranges::transform_view(v, f)),
        std::ranges::transform_view<View, PlusOne>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::transform_view(std::move(v), f)),
        std::ranges::transform_view<View, PlusOne>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::transform_view(r, f)),
        std::ranges::transform_view<std::ranges::ref_view<Range>, PlusOne>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::transform_view(std::move(r), f)),
        std::ranges::transform_view<std::ranges::owning_view<Range>, PlusOne>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::transform_view(br, f)),
        std::ranges::transform_view<std::ranges::ref_view<BorrowedRange>, PlusOne>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::transform_view(std::move(br), f)),
        std::ranges::transform_view<std::ranges::owning_view<BorrowedRange>, PlusOne>
    >);
}
