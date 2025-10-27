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
//   exp(const complex<T>& x);
//
// Tests for libc++-specific overflow handling behavior in complex exponential.
// These tests validate implementation-specific handling of edge cases where
// exp(real_part) overflows but the result should still be well-defined.

#include <complex>
#include <cassert>
#include <cmath>

#include "test_macros.h"

template <class T>
void test_overflow_case() {
  typedef std::complex<T> C;

  // In this case, the overflow of exp(real_part) is compensated when
  // sin(imag_part) is close to zero, resulting in a finite imaginary part.
  C z(T(90.0238094), T(5.900613e-39));
  C result = std::exp(z);

  assert(std::isinf(result.real()));
  assert(result.real() > 0);

  assert(std::isfinite(result.imag()));
  assert(std::abs(result.imag() - T(7.3746)) < T(1.0));
}

int main(int, char**) {
  test_overflow_case<float>();
  return 0;
}
