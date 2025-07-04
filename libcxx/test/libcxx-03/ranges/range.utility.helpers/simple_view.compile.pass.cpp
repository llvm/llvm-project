//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ranges>

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"

struct SimpleView : std::ranges::view_base {
  int *begin() const;
  int *end() const;
};

struct WrongConstView : std::ranges::view_base {
  int *begin();
  const int *begin() const;
  int *end();
  const int *end() const;
};

struct NoConstView : std::ranges::view_base {
  int *begin();
  int *end();
};

struct DifferentSentinel : std::ranges::view_base {
  int *begin() const;
  sentinel_wrapper<int*> end() const;
};

struct WrongConstSentinel : std::ranges::view_base {
  int *begin() const;
  sentinel_wrapper<int*> end();
  sentinel_wrapper<const int*> end() const;
};

static_assert( std::ranges::__simple_view<SimpleView>);
static_assert(!std::ranges::__simple_view<WrongConstView>);
static_assert(!std::ranges::__simple_view<NoConstView>);
static_assert( std::ranges::__simple_view<DifferentSentinel>);
static_assert(!std::ranges::__simple_view<WrongConstSentinel>);

// To make sure __simple_view and the test version of the concept stay in sync.
static_assert(simple_view<SimpleView>);
static_assert(!simple_view<WrongConstView>);
static_assert(!simple_view<NoConstView>);
static_assert(simple_view<DifferentSentinel>);
static_assert(!simple_view<WrongConstSentinel>);
