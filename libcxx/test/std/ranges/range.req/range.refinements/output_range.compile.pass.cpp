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

// template<class R, class T>
// concept output_range;

#include <ranges>

#include <iterator>
#include "test_iterators.h"
#include "test_range.h"

struct T { };

// Satisfied when it's a range and has the right iterator
struct GoodRange {
    output_iterator<T*> begin();
    sentinel end();
};
static_assert(std::ranges::range<GoodRange>);
static_assert(std::output_iterator<std::ranges::iterator_t<GoodRange>, T>);
static_assert(std::ranges::output_range<GoodRange, T>);

// Not satisfied when it's not a range
struct NotRange {
    output_iterator<T*> begin();
};
static_assert(!std::ranges::range<NotRange>);
static_assert( std::output_iterator<std::ranges::iterator_t<NotRange>, T>);
static_assert(!std::ranges::output_range<NotRange, T>);

// Not satisfied when the iterator is not an output_iterator
struct RangeWithBadIterator {
    cpp17_input_iterator<T const*> begin();
    sentinel end();
};
static_assert( std::ranges::range<RangeWithBadIterator>);
static_assert(!std::output_iterator<std::ranges::iterator_t<RangeWithBadIterator>, T>);
static_assert(!std::ranges::output_range<RangeWithBadIterator, T>);
