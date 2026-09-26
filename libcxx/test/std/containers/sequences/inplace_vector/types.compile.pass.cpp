//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class T, size_t N> class inplace_vector;

#include <concepts>
#include <cstddef>
#include <inplace_vector>
#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "../../Copyable.h"

template <class T, std::size_t _Cap>
void test() {
  using C = std::inplace_vector<T, _Cap>;

  static_assert(std::is_same_v<typename C::value_type, T>);
  static_assert(std::is_same_v<typename C::reference, T&>);
  static_assert(std::is_same_v<typename C::const_reference, const T&>);
  static_assert(std::is_same_v<typename C::pointer, T*>);
  static_assert(std::is_same_v<typename C::const_pointer, const T*>);
  static_assert(std::is_same_v<typename C::size_type, std::size_t>);
  static_assert(std::is_same_v<typename C::difference_type, std::ptrdiff_t>);
  static_assert(std::is_same_v<typename C::reverse_iterator, std::reverse_iterator<typename C::iterator> >);
  static_assert(std::is_same_v<typename C::const_reverse_iterator, std::reverse_iterator<typename C::const_iterator> >);

  static_assert(std::contiguous_iterator<typename C::iterator>);
  static_assert(std::contiguous_iterator<typename C::const_iterator>);
  static_assert(std::same_as<typename std::iterator_traits<typename C::iterator>::iterator_category,
                             std::random_access_iterator_tag>);
  static_assert(std::same_as<typename std::iterator_traits<typename C::const_iterator>::iterator_category,
                             std::random_access_iterator_tag>);
}

int main(int, char**) {
  test<int, 0>();
  test<int, 42>();
  test<int*, 42>();
  test<Copyable, 42>();

  return 0;
}
