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
//   sin(const complex<T>& x);

#include <complex>
#include <cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
void
test(const std::complex<T>& c, std::complex<T> x)
{
    assert(sin(c) == x);
}

template <class T>
void
test()
{
    test(std::complex<T>(0, 0), std::complex<T>(0, 0));
}

template <class T>
void test_edges() {
  const unsigned N = sizeof(testcases<T>) / sizeof(testcases<T>[0]);
  for (unsigned i = 0; i < N; ++i) {
    std::complex<T> r = sin(testcases<T>[i]);
    std::complex<T> t1(-imag(testcases<T>[i]), real(testcases<T>[i]));
    std::complex<T> t2 = sinh(t1);
    std::complex<T> z(imag(t2), -real(t2));
    if (std::isnan(real(r)))
      assert(std::isnan(real(z)));
    else {
      assert(real(r) == real(z));
      assert(std::signbit(real(r)) == std::signbit(real(z)));
    }
    if (std::isnan(imag(r)))
      assert(std::isnan(imag(z)));
    else {
      assert(imag(r) == imag(z));
      assert(std::signbit(imag(r)) == std::signbit(imag(z)));
    }
  }
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();
    test_edges<float>();
    test_edges<double>();
    test_edges<long double>();

    return 0;
}
