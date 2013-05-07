//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator*=(const complex& rhs);

#include <complex>
#include <cassert>

template <class T>
void
test()
{
    std::complex<T> c(1);
    const std::complex<T> c2(1.5, 2.5);
    assert(c.real() == 1);
    assert(c.imag() == 0);
    c *= c2;
    assert(c.real() == 1.5);
    assert(c.imag() == 2.5);
    c *= c2;
    assert(c.real() == -4);
    assert(c.imag() == 7.5);
}

int main()
{
    test<float>();
    test<double>();
    test<long double>();
}
