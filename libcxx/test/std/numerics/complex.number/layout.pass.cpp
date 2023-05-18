//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

#include <complex>
#include <cassert>

#include "test_macros.h"

template <class T>
void
test()
{
    std::complex<T> z;
    T* a = (T*)&z;
    assert(0 == z.real());
    assert(0 == z.imag());
    assert(a[0] == z.real());
    assert(a[1] == z.imag());
    a[0] = 5;
    a[1] = 6;
    assert(a[0] == z.real());
    assert(a[1] == z.imag());
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

  return 0;
}
