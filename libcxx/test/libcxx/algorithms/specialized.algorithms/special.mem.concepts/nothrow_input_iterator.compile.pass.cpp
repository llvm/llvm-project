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

// template<class I>
// concept __nothrow_input_iterator;

#include <memory>

#include "test_iterators.h"

struct InputProxyIterator {
  using value_type = int;
  using difference_type = int;
  InputProxyIterator& operator++();
  InputProxyIterator operator++(int);

  int operator*() const;
};

static_assert(std::ranges::__nothrow_input_iterator<cpp20_input_iterator<int*>>);
static_assert(!std::ranges::__nothrow_input_iterator<cpp17_output_iterator<int*>>);
static_assert(std::input_iterator<InputProxyIterator>);
static_assert(!std::ranges::__nothrow_input_iterator<InputProxyIterator>);
