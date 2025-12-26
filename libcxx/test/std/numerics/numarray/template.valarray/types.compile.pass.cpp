//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T>
// class valarray
// {
// public:
//     typedef T value_type;
//     typedef unspecified iterator;
//     typedef unspecified const_iterator;
//     ...

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <valarray>
#include <iterator>
#include <type_traits>

#include "test_macros.h"

template <class T>
void test() {
  using It  = typename std::valarray<T>::iterator;
  using CIt = typename std::valarray<T>::const_iterator;

  ASSERT_SAME_TYPE(typename std::valarray<T>::value_type, T);
  static_assert(
      std::is_base_of<std::random_access_iterator_tag, typename std::iterator_traits<It>::iterator_category>::value,
      "");
  static_assert(
      std::is_base_of<std::random_access_iterator_tag, typename std::iterator_traits<CIt>::iterator_category>::value,
      "");

  ASSERT_SAME_TYPE(decltype(*It()), T&);
  ASSERT_SAME_TYPE(decltype(*CIt()), const T&);

#if TEST_STD_VER >= 20
  static_assert(std::contiguous_iterator<It>);
  static_assert(std::contiguous_iterator<CIt>);
#endif
}

void test() {
  test<int>();
  test<double>();
}
