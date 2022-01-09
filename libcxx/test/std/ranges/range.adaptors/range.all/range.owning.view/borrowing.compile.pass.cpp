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

// template<class T>
//   inline constexpr bool enable_borrowed_range<owning_view<T>> = enable_borrowed_range<T>;

#include <ranges>
#include <cassert>

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

static_assert(!std::ranges::enable_borrowed_range<std::ranges::owning_view<Range>>);
static_assert( std::ranges::enable_borrowed_range<std::ranges::owning_view<BorrowableRange>>);
