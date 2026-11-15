//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<class T>
// concept approximately_sized_range;

#include <ranges>

#include "test_iterators.h"

static_assert(std::ranges::approximately_sized_range<int[5]>);
static_assert(std::ranges::approximately_sized_range<int (&)[5]>);
static_assert(!std::ranges::approximately_sized_range<int (&)[]>);
static_assert(!std::ranges::approximately_sized_range<int[]>);

struct range_has_reserve_hint {
  bidirectional_iterator<int*> begin();
  bidirectional_iterator<int*> end();
  int reserve_hint();
};
static_assert(std::ranges::approximately_sized_range<range_has_reserve_hint>);
static_assert(!std::ranges::approximately_sized_range<const range_has_reserve_hint>);
static_assert(!std::ranges::sized_range<range_has_reserve_hint>);
static_assert(!std::ranges::sized_range<const range_has_reserve_hint>);

struct range_has_const_reserve_hint {
  bidirectional_iterator<int*> begin();
  bidirectional_iterator<int*> end();
  int reserve_hint() const;
};
static_assert(std::ranges::approximately_sized_range<range_has_const_reserve_hint>);
static_assert(!std::ranges::approximately_sized_range<const range_has_const_reserve_hint>);
static_assert(!std::ranges::sized_range<range_has_const_reserve_hint>);
static_assert(!std::ranges::sized_range<const range_has_const_reserve_hint>);

struct const_range_has_reserve_hint {
  bidirectional_iterator<int*> begin() const;
  bidirectional_iterator<int*> end() const;
  int reserve_hint();
};
static_assert(std::ranges::approximately_sized_range<const_range_has_reserve_hint>);
static_assert(std::ranges::range<const const_range_has_reserve_hint>);
static_assert(!std::ranges::approximately_sized_range<const const_range_has_reserve_hint>);
static_assert(!std::ranges::sized_range<const_range_has_reserve_hint>);
static_assert(!std::ranges::sized_range<const const_range_has_reserve_hint>);

struct const_range_has_const_reserve_hint {
  bidirectional_iterator<int*> begin() const;
  bidirectional_iterator<int*> end() const;
  int reserve_hint() const;
};
static_assert(std::ranges::approximately_sized_range<const_range_has_const_reserve_hint>);
static_assert(std::ranges::approximately_sized_range<const const_range_has_const_reserve_hint>);
static_assert(!std::ranges::sized_range<const_range_has_const_reserve_hint>);
static_assert(!std::ranges::sized_range<const const_range_has_const_reserve_hint>);

struct sized_sentinel_range_is_approximately_sized {
  int* begin();
  int* end();
};
static_assert(std::ranges::approximately_sized_range<sized_sentinel_range_is_approximately_sized>);
static_assert(!std::ranges::approximately_sized_range<const sized_sentinel_range_is_approximately_sized>);

struct const_sized_sentinel_range_is_approximately_sized {
  int* begin() const;
  int* end() const;
};
static_assert(std::ranges::approximately_sized_range<const_sized_sentinel_range_is_approximately_sized>);
static_assert(std::ranges::approximately_sized_range<const const_sized_sentinel_range_is_approximately_sized>);

struct non_range_has_reserve_hint {
  int reserve_hint() const;
};
static_assert(requires(const non_range_has_reserve_hint x) { std::ranges::reserve_hint(x); });
static_assert(!std::ranges::approximately_sized_range<non_range_has_reserve_hint>);
static_assert(!std::ranges::approximately_sized_range<const non_range_has_reserve_hint>);

struct range_has_unqualified_reserve_hint {
  bidirectional_iterator<int*> begin() const;
  bidirectional_iterator<int*> end() const;

  friend int reserve_hint(range_has_unqualified_reserve_hint&);
};
static_assert(std::ranges::approximately_sized_range<range_has_unqualified_reserve_hint>);
static_assert(!std::ranges::approximately_sized_range<const range_has_unqualified_reserve_hint>);
static_assert(!std::ranges::sized_range<range_has_unqualified_reserve_hint>);
static_assert(!std::ranges::sized_range<const range_has_unqualified_reserve_hint>);

struct range_has_const_unqualified_reserve_hint {
  bidirectional_iterator<int*> begin() const;
  bidirectional_iterator<int*> end() const;

  friend int reserve_hint(const range_has_const_unqualified_reserve_hint&);
};
static_assert(std::ranges::approximately_sized_range<range_has_const_unqualified_reserve_hint>);
static_assert(std::ranges::approximately_sized_range<const range_has_const_unqualified_reserve_hint>);
static_assert(!std::ranges::sized_range<range_has_const_unqualified_reserve_hint>);
static_assert(!std::ranges::sized_range<const range_has_const_unqualified_reserve_hint>);

struct range_has_bool_reserve_hint {
  bidirectional_iterator<int*> begin() const;
  bidirectional_iterator<int*> end() const;

  bool reserve_hint();
};
static_assert(!std::ranges::approximately_sized_range<range_has_bool_reserve_hint>);
static_assert(!std::ranges::approximately_sized_range<const range_has_bool_reserve_hint>);
