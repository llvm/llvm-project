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
//   operator/(const complex<T>& lhs, const T& rhs);

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test(const std::complex<T>& lhs, const T& rhs, std::complex<T> x)
{
    assert(lhs / rhs == x);
}

template <class T>
void
test()
{
    std::complex<T> lhs(-4.0, 7.5);
    T rhs(2);
    std::complex<T>   x(-2, 3.75);
    test(lhs, rhs, x);
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

  return 0;
}
