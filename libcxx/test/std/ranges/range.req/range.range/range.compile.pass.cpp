//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// template<class T>
// concept range;

#include <ranges>

#include <vector>

#include "test_range.h"

namespace stdr = std::ranges;

static_assert(stdr::range<test_range<cpp20_input_iterator> >);

struct incompatible_iterators {
  int* begin();
  long* end();
};
static_assert(!stdr::range<incompatible_iterators>);

struct int_begin_int_end {
  int begin();
  int end();
};
static_assert(!stdr::range<int_begin_int_end>);

struct iterator_begin_int_end {
  int* begin();
  int end();
};
static_assert(!stdr::range<iterator_begin_int_end>);

struct int_begin_iterator_end {
  int begin();
  int* end();
};
static_assert(!stdr::range<int_begin_iterator_end>);
