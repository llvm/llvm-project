//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// concept range;

#include <ranges>

#include <vector>

#include "test_range.h"



static_assert(std::ranges::range<test_range<cpp20_input_iterator> >);

struct incompatible_iterators {
  int* begin();
  long* end();
};
static_assert(!std::ranges::range<incompatible_iterators>);

struct int_begin_int_end {
  int begin();
  int end();
};
static_assert(!std::ranges::range<int_begin_int_end>);

struct iterator_begin_int_end {
  int* begin();
  int end();
};
static_assert(!std::ranges::range<iterator_begin_int_end>);

struct int_begin_iterator_end {
  int begin();
  int* end();
};
static_assert(!std::ranges::range<int_begin_iterator_end>);

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::ranges::range<Holder<Incomplete>*>);
