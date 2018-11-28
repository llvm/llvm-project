// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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

int main() {
  test<int>();
  test<long long>();
  test<double>();
  test<long double>();
}
