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
//   conj(const complex<T>& x);

#include <complex>
#include <cassert>

template <class T>
void
test(const std::complex<T>& z, std::complex<T> x)
{
    assert(conj(z) == x);
}

template <class T>
void
test()
{
    test(std::complex<T>(1, 2), std::complex<T>(1, -2));
    test(std::complex<T>(-1, 2), std::complex<T>(-1, -2));
    test(std::complex<T>(1, -2), std::complex<T>(1, 2));
    test(std::complex<T>(-1, -2), std::complex<T>(-1, 2));
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

  return 0;
}
