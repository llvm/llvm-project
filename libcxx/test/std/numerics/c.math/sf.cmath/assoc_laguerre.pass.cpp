//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Missing powl
// XFAIL: LLVM-LIBC-FIXME

// REQUIRES: std-at-least-c++17

// <cmath>
//
// [sf.cmath.assoc.laguerre], associated Laguerre polynomials
// floating-point-type assoc_laguerre(unsigned n, unsigned m, floating-point-type x);
// float               assoc_laguerref(unsigned n, unsigned m, float x);
// long double         assoc_laguerrel(unsigned n, unsigned m, long double x);

#include <cmath>

int main(int, char**) {
  std::assoc_laguerref(0, 0, 0.0f);

  return 0;
}
