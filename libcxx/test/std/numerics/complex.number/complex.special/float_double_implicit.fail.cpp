//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<> class complex<float>
// {
// public:
//     explicit constexpr complex(const complex<double>&);
// };

#include <complex>
#include <cassert>

int main()
{
    const std::complex<double> cd(2.5, 3.5);
    std::complex<float> cf = cd;
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());
}
