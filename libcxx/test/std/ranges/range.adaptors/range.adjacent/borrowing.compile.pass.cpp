//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class View, size_t N>
// inline constexpr bool enable_borrowed_range<adjacent_view<View, N>> =
//      enable_borrowed_range<View>;

#include <ranges>

struct Borrowed : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<Borrowed> = true;

static_assert(std::ranges::borrowed_range<Borrowed>);

struct NonBorrowed : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};
static_assert(!std::ranges::borrowed_range<NonBorrowed>);

// test borrowed_range
static_assert(std::ranges::borrowed_range<std::ranges::adjacent_view<Borrowed, 1>>);
static_assert(std::ranges::borrowed_range<std::ranges::adjacent_view<Borrowed, 2>>);
static_assert(std::ranges::borrowed_range<std::ranges::adjacent_view<Borrowed, 3>>);
static_assert(!std::ranges::borrowed_range<std::ranges::adjacent_view<NonBorrowed, 1>>);
static_assert(!std::ranges::borrowed_range<std::ranges::adjacent_view<NonBorrowed, 2>>);
static_assert(!std::ranges::borrowed_range<std::ranges::adjacent_view<NonBorrowed, 3>>);
