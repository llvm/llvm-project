//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_SUPPORT_TEST_RANGE_H
#define LIBCXX_TEST_SUPPORT_TEST_RANGE_H

#include <iterator>
#include <ranges>

#include "test_iterators.h"

#if TEST_STD_VER < 17
#error "test/support/test_range.h" can only be included in builds supporting ranges
#endif

struct sentinel {
  bool operator==(std::input_or_output_iterator auto const&) const;
};

template <template <class...> class I>
requires std::input_or_output_iterator<I<int*> >
struct test_range {
  I<int*> begin();
  I<int const*> begin() const;
  sentinel end();
  sentinel end() const;
};

template <template <class...> class I>
requires std::input_or_output_iterator<I<int*> >
struct test_non_const_range {
  I<int*> begin();
  sentinel end();
};

template <template <class...> class I>
requires std::input_or_output_iterator<I<int*> >
struct test_common_range {
  I<int*> begin();
  I<int const*> begin() const;
  I<int*> end();
  I<int const*> end() const;
};

template <template <class...> class I>
requires std::input_or_output_iterator<I<int*> >
struct test_non_const_common_range {
  I<int*> begin();
  I<int*> end();
};

template <template <class...> class I>
requires std::input_or_output_iterator<I<int*> >
struct test_view : std::ranges::view_base {
  I<int*> begin();
  I<int const*> begin() const;
  sentinel end();
  sentinel end() const;
};

struct BorrowedRange {
  int *begin() const;
  int *end() const;
  BorrowedRange(BorrowedRange&&) = delete;
};
template<> inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange> = true;
static_assert(!std::ranges::view<BorrowedRange>);
static_assert(std::ranges::borrowed_range<BorrowedRange>);

using BorrowedView = std::ranges::empty_view<int>;
static_assert(std::ranges::view<BorrowedView>);
static_assert(std::ranges::borrowed_range<BorrowedView>);

using NonBorrowedView = std::ranges::single_view<int>;
static_assert(std::ranges::view<NonBorrowedView>);
static_assert(!std::ranges::borrowed_range<NonBorrowedView>);

#endif // LIBCXX_TEST_SUPPORT_TEST_RANGE_H
