//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class R>
// concept __nothrow_input_range;

#include <memory>

#include "test_iterators.h"
#include "test_range.h"

// Has to be a template to work with `test_range`.
template <typename>
struct InputProxyIterator {
  using value_type = int;
  using difference_type = int;
  InputProxyIterator& operator++();
  InputProxyIterator operator++(int);

  int operator*() const;
};

static_assert(std::ranges::__nothrow_input_range<test_range<cpp20_input_iterator>>);
static_assert(std::ranges::input_range<test_range<InputProxyIterator>>);
static_assert(!std::ranges::__nothrow_input_range<test_range<InputProxyIterator>>);
