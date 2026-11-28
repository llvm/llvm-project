//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// Member types of hive.

#include <hive>
#include <iterator>
#include <memory>
#include <type_traits>

#include "min_allocator.h"
#include "test_macros.h"

template <class T, class Alloc>
void test() {
  using H      = std::hive<T, Alloc>;
  using Traits = std::allocator_traits<Alloc>;

  static_assert(std::is_same_v<typename H::value_type, T>);
  static_assert(std::is_same_v<typename H::allocator_type, Alloc>);
  static_assert(std::is_same_v<typename H::pointer, typename Traits::pointer>);
  static_assert(std::is_same_v<typename H::const_pointer, typename Traits::const_pointer>);
  static_assert(std::is_same_v<typename H::reference, T&>);
  static_assert(std::is_same_v<typename H::const_reference, const T&>);
  static_assert(std::is_unsigned_v<typename H::size_type>);
  static_assert(std::is_signed_v<typename H::difference_type>);
  static_assert(std::is_same_v<typename H::difference_type,
                               typename std::iterator_traits<typename H::iterator>::difference_type>);
  static_assert(std::is_same_v<typename H::difference_type,
                               typename std::iterator_traits<typename H::const_iterator>::difference_type>);
  static_assert(std::is_same_v<typename H::reverse_iterator, std::reverse_iterator<typename H::iterator>>);
  static_assert(std::is_same_v<typename H::const_reverse_iterator, std::reverse_iterator<typename H::const_iterator>>);
}

template void test<int, std::allocator<int>>();
template void test<double, std::allocator<double>>();
template void test<int, min_allocator<int>>();
