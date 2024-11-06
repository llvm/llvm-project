//===-- lib/Evaluate/wrappers.c -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef _AIX
#include <complex.h>

void csqrtf_wrapper(const float x[2], float res[2]) {
  float complex c = x[0] + I * x[1];
  float complex r = csqrtf(c);
  res[0] = crealf(r);
  res[1] = cimagf(r);
}

void csqrt_wrapper(const double x[2], double res[2]) {
  double complex c = x[0] + I * x[1];
  double complex r = csqrt(c);
  res[0] = creal(r);
  res[1] = cimag(r);
}
#endif
