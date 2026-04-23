//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// complex& operator/=(const complex& rhs); // constexpr in C++20

#include <cassert>
#include <complex>

#include "floating_pointer_helpers.h"
#include "test_macros.h"

template <class T>
TEST_CONSTEXPR_CXX20
bool
test()
{
    std::complex<T> c(-4, 7.5);
    const std::complex<T> c2(1.5, 2.5);
    assert(is_close(c.real(), T(-4)));
    assert(is_close(c.imag(), T(7.5)));
    c /= c2;
    assert(is_close(c.real(), T(1.5)));
    assert(is_close(c.imag(), T(2.5)));
    c /= c2;
    assert(is_close(c.real(), T(1)));
    assert(is_close(c.imag(), T(0)));

    std::complex<T> c3;

    c3 = c;
    std::complex<int> ic (1,1);
    c3 /= ic;
    assert(is_close(c3.real(), T(0.5)));
    assert(is_close(c3.imag(), T(-0.5)));

    c3 = c;
    std::complex<float> fc (1,1);
    c3 /= fc;
    assert(is_close(c3.real(), T(0.5)));
    assert(is_close(c3.imag(), T(-0.5)));
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
