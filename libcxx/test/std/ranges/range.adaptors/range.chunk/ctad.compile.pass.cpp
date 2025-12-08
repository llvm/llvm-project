//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::chunk

#include <ranges>

struct view : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct range {
  int* begin() const;
  int* end() const;
};

struct borrowed_range {
  int* begin() const;
  int* end() const;
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<borrowed_range> = true;

void testCTAD() {
  view v;
  range r;
  borrowed_range br;

  static_assert(std::same_as< decltype(std::ranges::chunk_view(v, 0)), std::ranges::chunk_view<view> >);
  static_assert(std::same_as< decltype(std::ranges::chunk_view(std::move(v), 0)), std::ranges::chunk_view<view> >);
  static_assert(
      std::same_as< decltype(std::ranges::chunk_view(r, 0)), std::ranges::chunk_view<std::ranges::ref_view<range>> >);
  static_assert(std::same_as< decltype(std::ranges::chunk_view(std::move(r), 0)),
                              std::ranges::chunk_view<std::ranges::owning_view<range>> >);
  static_assert(std::same_as< decltype(std::ranges::chunk_view(br, 0)),
                              std::ranges::chunk_view<std::ranges::ref_view<borrowed_range>> >);
  static_assert(std::same_as< decltype(std::ranges::chunk_view(std::move(br), 0)),
                              std::ranges::chunk_view<std::ranges::owning_view<borrowed_range>> >);
}
