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

// template<class R>
// concept __nothrow_forward_range;

#include <memory>

#include "test_iterators.h"
#include "test_range.h"

// Has to be a template to work with `test_range`.
template <typename>
struct ForwardProxyIterator {
  using value_type = int;
  using difference_type = int;
  ForwardProxyIterator& operator++();
  ForwardProxyIterator operator++(int);
  bool operator==(const ForwardProxyIterator&) const;

  int operator*() const;
};

static_assert(std::ranges::__nothrow_forward_range<test_range<forward_iterator>>);
static_assert(!std::ranges::__nothrow_forward_range<test_range<cpp20_input_iterator>>);
static_assert(std::ranges::forward_range<test_range<ForwardProxyIterator>>);
static_assert(!std::ranges::__nothrow_forward_range<test_range<ForwardProxyIterator>>);

constexpr bool forward_subsumes_input(std::ranges::__nothrow_forward_range auto&&) {
  return true;
}
constexpr bool forward_subsumes_input(std::ranges::__nothrow_input_range auto&&);

static_assert(forward_subsumes_input("foo"));
