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
// concept borrowed_range;

#include <ranges>

struct NotRange {
  int begin() const;
  int end() const;
};

struct Range {
  int *begin();
  int *end();
};

struct ConstRange {
  int *begin() const;
  int *end() const;
};

struct BorrowedRange {
  int *begin() const;
  int *end() const;
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange> = true;

static_assert(!std::ranges::borrowed_range<NotRange>);
static_assert(!std::ranges::borrowed_range<NotRange&>);
static_assert(!std::ranges::borrowed_range<const NotRange>);
static_assert(!std::ranges::borrowed_range<const NotRange&>);
static_assert(!std::ranges::borrowed_range<NotRange&&>);

static_assert(!std::ranges::borrowed_range<Range>);
static_assert( std::ranges::borrowed_range<Range&>);
static_assert(!std::ranges::borrowed_range<const Range>);
static_assert(!std::ranges::borrowed_range<const Range&>);
static_assert(!std::ranges::borrowed_range<Range&&>);

static_assert(!std::ranges::borrowed_range<ConstRange>);
static_assert( std::ranges::borrowed_range<ConstRange&>);
static_assert(!std::ranges::borrowed_range<const ConstRange>);
static_assert( std::ranges::borrowed_range<const ConstRange&>);
static_assert(!std::ranges::borrowed_range<ConstRange&&>);

static_assert( std::ranges::borrowed_range<BorrowedRange>);
static_assert( std::ranges::borrowed_range<BorrowedRange&>);
static_assert( std::ranges::borrowed_range<const BorrowedRange>);
static_assert( std::ranges::borrowed_range<const BorrowedRange&>);
static_assert( std::ranges::borrowed_range<BorrowedRange&&>);
