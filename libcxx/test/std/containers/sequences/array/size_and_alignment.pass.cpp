//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <class T, size_t N >
// struct array

// Test the size and alignment matches that of an array of a given type.

#include <array>
#include <iterator>
#include <type_traits>
#include <cstddef>

#include "test_macros.h"


template <class T, size_t Size>
struct MyArray {
  T elems[Size];
};

template <class T, size_t Size>
void test() {
  typedef T CArrayT[Size == 0 ? 1 : Size];
  typedef std::array<T, Size> ArrayT;
  typedef MyArray<T, Size == 0 ? 1 : Size> MyArrayT;
  static_assert(sizeof(ArrayT) == sizeof(CArrayT), "");
  static_assert(sizeof(ArrayT) == sizeof(MyArrayT), "");
  static_assert(TEST_ALIGNOF(ArrayT) == TEST_ALIGNOF(MyArrayT), "");
}

template <class T>
void test_type() {
  test<T, 1>();
  test<T, 42>();
  test<T, 0>();
}

#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
struct TEST_ALIGNAS(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) TestType1 {

};

struct TEST_ALIGNAS(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2) TestType2 {
  char data[1000];
};
#else
struct TEST_ALIGNAS(TEST_ALIGNOF(std::max_align_t) * 2) TestType1 {

};

struct TEST_ALIGNAS(TEST_ALIGNOF(std::max_align_t) * 2) TestType2 {
  char data[1000];
};
#endif

int main(int, char**) {
  test_type<char>();
  test_type<int>();
  test_type<double>();
  test_type<long double>();

#if TEST_STD_VER >= 11
  test_type<std::max_align_t>();
#endif
  test_type<TestType1>();
  test_type<TestType2>();

  return 0;
}
