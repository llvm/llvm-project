//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// inplace_vector<T,N>::iterator
// inplace_vector<T,N>::const_iterator

#include <inplace_vector>
#include <iterator>

template <class C>
constexpr bool test() {
  static_assert(std::contiguous_iterator<typename C::iterator>);
  static_assert(std::contiguous_iterator<typename C::const_iterator>);
  static_assert(std::sentinel_for<typename C::iterator, typename C::iterator>);
  static_assert(std::sentinel_for<typename C::const_iterator, typename C::iterator>);
  static_assert(std::sentinel_for<typename C::iterator, typename C::const_iterator>);
  static_assert(std::sentinel_for<typename C::const_iterator, typename C::const_iterator>);

  return true;
}

static_assert(test<std::inplace_vector<int, 0> >());
static_assert(test<std::inplace_vector<int, 8> >());
