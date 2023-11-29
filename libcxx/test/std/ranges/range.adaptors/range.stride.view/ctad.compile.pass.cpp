//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <class _Range>
// stride_view(_Range&&, range_difference_t<_Range>) -> stride_view<views::all_t<_Range>>;

#include <concepts>
#include <ranges>

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct Range {
  int* begin() const;
  int* end() const;
};

void testCTAD() {
  View v;
  Range r;

  static_assert(std::same_as< decltype(std::ranges::stride_view(v, 5)), std::ranges::stride_view<View> >);
  static_assert(std::same_as< decltype(std::ranges::stride_view(std::move(v), 5)), std::ranges::stride_view<View> >);
  static_assert(
      std::same_as< decltype(std::ranges::stride_view(r, 5)), std::ranges::stride_view<std::ranges::ref_view<Range>> >);
  static_assert(std::same_as< decltype(std::ranges::stride_view(std::move(r), 5)),
                              std::ranges::stride_view<std::ranges::owning_view<Range>> >);
}
