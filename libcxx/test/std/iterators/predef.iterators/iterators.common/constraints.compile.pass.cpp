//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_or_output_iterator I, sentinel_for<I> S>
//   requires (!same_as<I, S> && copyable<I>)

#include <iterator>

#include "test_iterators.h"

template<class I, class S>
concept ValidCommonIterator = requires {
  typename std::common_iterator<I, S>;
};

static_assert( ValidCommonIterator<int*, const int*>);
static_assert(!ValidCommonIterator<int, int>); // !input_or_output_iterator<I>
static_assert(!ValidCommonIterator<int*, float*>); // !sentinel_for<S, I>
static_assert(!ValidCommonIterator<int*, int*>); // !same_as<I, S>
static_assert(!ValidCommonIterator<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>); // !copyable<I>
