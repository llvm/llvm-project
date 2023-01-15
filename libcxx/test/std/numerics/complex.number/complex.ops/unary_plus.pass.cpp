//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
//   complex<T>
//   operator+(const complex<T>&); // constexpr in C++20

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX20
bool
test()
{
  assert(+std::complex<T>(1.5, -2.5) == std::complex<T>(1.5, -2.5));
  assert(+std::complex<T>(-1.5, 2.5) == std::complex<T>(-1.5, 2.5));
  return true;
}

int main(int, char**)
{
  test<float>();
  test<double>();
  test<long double>();

#if TEST_STD_VER > 17
  static_assert(test<float>());
  static_assert(test<double>());
  static_assert(test<long double>());
#endif

  return 0;
}
