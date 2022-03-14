//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<range R>
// using range_difference_t = iter_difference_t<iterator_t<R>>;

// template<range R>
// using range_value_t = iter_value_t<iterator_t<R>>;

// template<range R>
// using range_reference_t = iter_reference_t<iterator_t<R>>;

// template<range R>
// using range_rvalue_reference_t = iter_rvalue_reference_t<iterator_t<R>>;

#include <ranges>

#include <concepts>

#include "test_iterators.h"
#include "test_range.h"



static_assert(std::same_as<std::ranges::range_difference_t<test_range<cpp20_input_iterator> >, std::iter_difference_t<int*> >);
static_assert(std::same_as<std::ranges::range_value_t<test_range<cpp20_input_iterator> >, std::iter_value_t<int*> >);
static_assert(std::same_as<std::ranges::range_reference_t<test_range<cpp20_input_iterator> >, std::iter_reference_t<int*> >);
static_assert(std::same_as<std::ranges::range_rvalue_reference_t<test_range<cpp20_input_iterator> >, std::iter_rvalue_reference_t<int*> >);
