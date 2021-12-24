//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-AIX-FIXME

// <complex>

// template<class T>
//   complex<T>
//   pow(const complex<T>& x, const T& y);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& a, const T& b, std::complex<T> x)
{
    std::complex<T> c = pow(a, b);
    is_about(real(c), real(x));
    is_about(imag(c), imag(x));
}

template <class T>
void
test()
{
    test(std::complex<T>(2, 3), T(2), std::complex<T>(-5, 12));
}

void test_edges()
{
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            std::complex<double> r = pow(testcases[i], real(testcases[j]));
            std::complex<double> z = exp(std::complex<double>(real(testcases[j])) * log(testcases[i]));
            if (std::isnan(real(r)))
                assert(std::isnan(real(z)));
            else
            {
                assert(real(r) == real(z));
            }
            if (std::isnan(imag(r)))
                assert(std::isnan(imag(z)));
            else
            {
                assert(imag(r) == imag(z));
            }
        }
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();
    test_edges();

  return 0;
}
