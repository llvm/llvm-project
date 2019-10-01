// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_ALIGNOF acts the same as the C++11 keyword `alignof`, and
// not as the GNU extension `__alignof`. The former returns the minimal required
// alignment for a type, whereas the latter returns the preferred alignment.
//
// See llvm.org/PR39713

#include <type_traits>
#include "test_macros.h"

template <class T>
void test() {
  static_assert(_LIBCPP_ALIGNOF(T) == std::alignment_of<T>::value, "");
  static_assert(_LIBCPP_ALIGNOF(T) == TEST_ALIGNOF(T), "");
#if TEST_STD_VER >= 11
  static_assert(_LIBCPP_ALIGNOF(T) == alignof(T), "");
#endif
#ifdef TEST_COMPILER_CLANG
  static_assert(_LIBCPP_ALIGNOF(T) == _Alignof(T), "");
#endif
}

int main(int, char**) {
  test<int>();
  test<long long>();
  test<double>();
  test<long double>();
  return 0;
}
