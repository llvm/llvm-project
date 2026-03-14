//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator+=(const T& rhs); // constexpr in C++20

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX20
bool
test()
{
    std::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    c += 1.5;
    assert(c.real() == 1.5);
    assert(c.imag() == 0);
    c += 1.5;
    assert(c.real() == 3);
    assert(c.imag() == 0);
    c += -1.5;
    assert(c.real() == 1.5);
    assert(c.imag() == 0);
    return true;
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

#if TEST_STD_VER >= 20
    static_assert(test<float>());
    static_assert(test<double>());
    static_assert(test<long double>());
#endif

  return 0;
}
