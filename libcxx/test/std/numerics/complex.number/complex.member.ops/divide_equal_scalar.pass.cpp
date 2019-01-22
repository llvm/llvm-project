//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator/=(const T& rhs);

#include <complex>
#include <cassert>

template <class T>
void
test()
{
    std::complex<T> c(1);
    assert(c.real() == 1);
    assert(c.imag() == 0);
    c /= 0.5;
    assert(c.real() == 2);
    assert(c.imag() == 0);
    c /= 0.5;
    assert(c.real() == 4);
    assert(c.imag() == 0);
    c /= -0.5;
    assert(c.real() == -8);
    assert(c.imag() == 0);
    c.imag(2);
    c /= 0.5;
    assert(c.real() == -16);
    assert(c.imag() == 4);
}

int main()
{
    test<float>();
    test<double>();
    test<long double>();
}
