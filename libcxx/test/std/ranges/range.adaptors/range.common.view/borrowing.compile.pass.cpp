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
//   inline constexpr bool enable_borrowed_range<common_view<T>> = enable_borrowed_range<T>;

#include <ranges>
#include <cassert>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  friend int* begin(View&);
  friend int* begin(View const&);
  friend sentinel_wrapper<int*> end(View&);
  friend sentinel_wrapper<int*> end(View const&);
};

struct BorrowableView : std::ranges::view_base {
  friend int* begin(BorrowableView&);
  friend int* begin(BorrowableView const&);
  friend sentinel_wrapper<int*> end(BorrowableView&);
  friend sentinel_wrapper<int*> end(BorrowableView const&);
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowableView> = true;

static_assert(!std::ranges::enable_borrowed_range<std::ranges::common_view<View>>);
static_assert( std::ranges::enable_borrowed_range<std::ranges::common_view<BorrowableView>>);
