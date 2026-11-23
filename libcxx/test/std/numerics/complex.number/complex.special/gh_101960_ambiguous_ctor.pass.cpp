//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// Regression test for https://llvm.org/PR101960 where we used to trigger an ambiguous constructor.

#include <complex>
#include <cassert>

struct NastyConvertible {
  template <class T>
  operator T() const {
    return T(0);
  }
};

template <class T>
void test() {
  NastyConvertible nasty;
  std::complex<T> x(nasty, nasty);
  assert(x.real() == T(0));
  assert(x.imag() == T(0));
}

int main(int, char**) {
  test<float>();
  test<double>();
  test<long double>();

  return 0;
}
