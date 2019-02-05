//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<> class complex<double>
// {
// public:
//     explicit constexpr complex(const complex<long double>&);
// };

#include <complex>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    const std::complex<long double> cd(2.5, 3.5);
    std::complex<double> cf(cd);
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());
    }
#if TEST_STD_VER >= 11
    {
    constexpr std::complex<long double> cd(2.5, 3.5);
    constexpr std::complex<double> cf(cd);
    static_assert(cf.real() == cd.real(), "");
    static_assert(cf.imag() == cd.imag(), "");
    }
#endif

  return 0;
}
