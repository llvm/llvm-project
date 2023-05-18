//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class I>
// concept __nothrow_forward_iterator;

#include <memory>

#include "test_iterators.h"

static_assert(std::ranges::__nothrow_forward_iterator<forward_iterator<int*>>);
static_assert(std::forward_iterator<ForwardProxyIterator<int*>>);
static_assert(!std::ranges::__nothrow_forward_iterator<ForwardProxyIterator<int*>>);

constexpr bool forward_subsumes_input(std::ranges::__nothrow_forward_iterator auto) {
  return true;
}
constexpr bool forward_subsumes_input(std::ranges::__nothrow_input_iterator auto);

static_assert(forward_subsumes_input("foo"));
