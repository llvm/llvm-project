//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
//
// template<class T, size_t N>
//   inline constexpr bool enable_borrowed_range<elements_view<T, N>> =
//     enable_borrowed_range<T>;

#include <ranges>
#include <tuple>

struct NonBorrowed : std::ranges::view_base {
  std::tuple<int>* begin();
  std::tuple<int>* end();
};

struct Borrowed : std::ranges::view_base {
  std::tuple<int>* begin();
  std::tuple<int>* end();
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<Borrowed> = true;

static_assert(!std::ranges::borrowed_range<std::ranges::elements_view<NonBorrowed, 0>>);
static_assert(std::ranges::borrowed_range<std::ranges::elements_view<Borrowed, 0>>);
