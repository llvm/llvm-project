//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ranges>

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

#include <ranges>

#include "test_macros.h"
#include "test_iterators.h"

struct SimpleView : std::ranges::view_base {
  friend int* begin(SimpleView&);
  friend int* begin(SimpleView const&);
  friend int* end(SimpleView&);
  friend int* end(SimpleView const&);
};

struct WrongConstView : std::ranges::view_base {
  friend       int* begin(WrongConstView&);
  friend const int* begin(WrongConstView const&);
  friend       int* end(WrongConstView&);
  friend const int* end(WrongConstView const&);
};

struct NoConstView : std::ranges::view_base {
  friend int* begin(NoConstView&);
  friend int* end(NoConstView&);
};

struct DifferentSentinel : std::ranges::view_base {
  friend int* begin(DifferentSentinel&);
  friend int* begin(DifferentSentinel const&);
  friend sentinel_wrapper<int*> end(DifferentSentinel&);
  friend sentinel_wrapper<int*> end(DifferentSentinel const&);
};

static_assert( std::ranges::__simple_view<SimpleView>);
static_assert(!std::ranges::__simple_view<WrongConstView>);
static_assert(!std::ranges::__simple_view<NoConstView>);
static_assert(!std::ranges::__simple_view<DifferentSentinel>);
