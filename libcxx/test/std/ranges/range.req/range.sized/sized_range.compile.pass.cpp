//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class T>
// concept sized_range;

#include <ranges>

#include "test_iterators.h"



static_assert(std::ranges::sized_range<int[5]>);
static_assert(std::ranges::sized_range<int (&)[5]>);
static_assert(!std::ranges::sized_range<int (&)[]>);
static_assert(!std::ranges::sized_range<int[]>);

struct range_has_size {
  bidirectional_iterator<int*> begin();
  bidirectional_iterator<int*> end();
  int size();
};
static_assert(std::ranges::sized_range<range_has_size>);
static_assert(!std::ranges::sized_range<range_has_size const>);

struct range_has_const_size {
  bidirectional_iterator<int*> begin();
  bidirectional_iterator<int*> end();
  int size() const;
};
static_assert(std::ranges::sized_range<range_has_const_size>);
static_assert(!std::ranges::sized_range<range_has_const_size const>);

struct const_range_has_size {
  bidirectional_iterator<int*> begin() const;
  bidirectional_iterator<int*> end() const;
  int size();
};
static_assert(std::ranges::sized_range<const_range_has_size>);
static_assert(std::ranges::range<const_range_has_size const>);
static_assert(!std::ranges::sized_range<const_range_has_size const>);

struct const_range_has_const_size {
  bidirectional_iterator<int*> begin() const;
  bidirectional_iterator<int*> end() const;
  int size() const;
};
static_assert(std::ranges::sized_range<const_range_has_const_size>);
static_assert(std::ranges::sized_range<const_range_has_const_size const>);

struct sized_sentinel_range_has_size {
  int* begin();
  int* end();
};
static_assert(std::ranges::sized_range<sized_sentinel_range_has_size>);
static_assert(!std::ranges::sized_range<sized_sentinel_range_has_size const>);

struct const_sized_sentinel_range_has_size {
  int* begin() const;
  int* end() const;
};
static_assert(std::ranges::sized_range<const_sized_sentinel_range_has_size>);
static_assert(std::ranges::sized_range<const_sized_sentinel_range_has_size const>);

struct non_range_has_size {
  int size() const;
};
static_assert(requires(non_range_has_size const x) { std::ranges::size(x); });
static_assert(!std::ranges::sized_range<non_range_has_size>);
static_assert(!std::ranges::sized_range<non_range_has_size const>);
