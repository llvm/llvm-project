//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// void real(T val); // constexpr in C++20
// void imag(T val); // constexpr in C++20

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX20
void
test_constexpr()
{
#if TEST_STD_VER > 11
    constexpr std::complex<T> c1;
    static_assert(c1.real() == 0, "");
    static_assert(c1.imag() == 0, "");
    constexpr std::complex<T> c2(3);
    static_assert(c2.real() == 3, "");
    static_assert(c2.imag() == 0, "");
    constexpr std::complex<T> c3(3, 4);
    static_assert(c3.real() == 3, "");
    static_assert(c3.imag() == 4, "");
#endif
}

template <class T>
TEST_CONSTEXPR_CXX20
bool
test()
{
    std::complex<T> c;
    assert(c.real() == 0);
    assert(c.imag() == 0);
    c.real(3.5);
    assert(c.real() == 3.5);
    assert(c.imag() == 0);
    c.imag(4.5);
    assert(c.real() == 3.5);
    assert(c.imag() == 4.5);
    c.real(-4.5);
    assert(c.real() == -4.5);
    assert(c.imag() == 4.5);
    c.imag(-5.5);
    assert(c.real() == -4.5);
    assert(c.imag() == -5.5);

    test_constexpr<T>();
    return true;
}

int main(int, char**) {
    test<float>();
    test<double>();
    test<long double>();
    test_constexpr<int>();

#if TEST_STD_VER >= 20
    static_assert(test<float>());
    static_assert(test<double>());
    static_assert(test<long double>());
#endif

  return 0;
}
