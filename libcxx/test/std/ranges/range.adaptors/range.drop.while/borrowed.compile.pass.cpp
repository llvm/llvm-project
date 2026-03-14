//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
//
// template<class T, class Pred>
//   inline constexpr bool enable_borrowed_range<drop_while_view<T, Pred>> =
//     enable_borrowed_range<T>;

#include <ranges>

struct NonBorrowed : std::ranges::view_base {
  int* begin();
  int* end();
};

struct Borrowed : std::ranges::view_base {
  int* begin();
  int* end();
};

struct Pred {
  bool operator()(int) const;
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<Borrowed> = true;

static_assert(!std::ranges::borrowed_range<std::ranges::drop_while_view<NonBorrowed, Pred>>);
static_assert(std::ranges::borrowed_range<std::ranges::drop_while_view<Borrowed, Pred>>);
