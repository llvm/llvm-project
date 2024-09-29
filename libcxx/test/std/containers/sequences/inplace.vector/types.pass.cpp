//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// Test nested types:

// template <class T, size_t N>
// class inplace_vector
// {
// public:
//     using value_type             = T;
//     using pointer                = T*;
//     using const_pointer          = const T*;
//     using reference              = value_type&;
//     using const_reference        = const value_type&;
//     using size_type              = size_t;
//     using difference_type        = ptrdiff_t;
//     using iterator               = implementation-defined;
//     using const_iterator         = implementation-defined;
//     using reverse_iterator       = std::reverse_iterator<iterator>;
//     using const_reverse_iterator = std::reverse_iterator<const_iterator>;
// };

#include <inplace_vector>
#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "../../Copyable.h"

template <class T, size_t N>
void test() {
  using C = std::inplace_vector<T, N>;
  static_assert(std::is_same_v<typename C::value_type, T>);
  static_assert(std::is_same_v<typename C::size_type, std::size_t>);
  static_assert(std::is_same_v<typename C::difference_type, std::ptrdiff_t>);
  static_assert(std::is_same_v<typename C::reference, T&>);
  static_assert(std::is_same_v<typename C::const_reference, const T&>);
  static_assert(std::is_same_v<typename C::pointer, T*>);
  static_assert(std::is_same_v<typename C::const_pointer, const T*>);
  static_assert(std::is_same_v< typename std::iterator_traits<typename C::iterator>::iterator_category,
                                std::random_access_iterator_tag>);
  static_assert(std::is_same_v< typename std::iterator_traits<typename C::const_iterator>::iterator_category,
                                std::random_access_iterator_tag>);
  static_assert(std::contiguous_iterator<typename C::iterator>);
  static_assert(std::contiguous_iterator<typename C::const_iterator>);
  static_assert(std::is_same_v<typename C::reverse_iterator, std::reverse_iterator<typename C::iterator>>);
  static_assert(std::is_same_v<typename C::const_reverse_iterator, std::reverse_iterator<typename C::const_iterator>>);
}

template <class T>
void test() {
  test<T, 0>();
  test<T, 1>();
  test<T, 10>();
  test<T, 100>();
}

int main(int, char**) {
  test<int>();
  test<int*>();
  test<Copyable>();
}
