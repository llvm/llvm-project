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

// template<class R>
//   common_view(R&&) -> common_view<views::all_t<R>>;

#include <ranges>
#include <cassert>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  friend int* begin(View&);
  friend int* begin(View const&);
  friend sentinel_wrapper<int*> end(View&);
  friend sentinel_wrapper<int*> end(View const&);
};

struct Range {
  friend int* begin(Range&);
  friend int* begin(Range const&);
  friend sentinel_wrapper<int*> end(Range&);
  friend sentinel_wrapper<int*> end(Range const&);
};

struct BorrowedRange {
  friend int* begin(BorrowedRange&);
  friend int* begin(BorrowedRange const&);
  friend sentinel_wrapper<int*> end(BorrowedRange&);
  friend sentinel_wrapper<int*> end(BorrowedRange const&);
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange> = true;

void testCTAD() {
    View v;
    Range r;
    BorrowedRange br;
    static_assert(std::same_as<
        decltype(std::ranges::common_view(v)),
        std::ranges::common_view<View>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::common_view(r)),
        std::ranges::common_view<std::ranges::ref_view<Range>>
    >);
    // std::ranges::common_view(std::move(r)) invalid. RValue range must be borrowed.
    static_assert(std::same_as<
        decltype(std::ranges::common_view(br)),
        std::ranges::common_view<std::ranges::ref_view<BorrowedRange>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::common_view(std::move(br))),
        std::ranges::common_view<std::ranges::subrange<
          int *, sentinel_wrapper<int *>, std::ranges::subrange_kind::unsized>>
    >);
}
