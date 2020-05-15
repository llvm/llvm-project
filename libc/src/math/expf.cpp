//===-- Single-precision e^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exp_utils.h"
#include "math_utils.h"

#include "include/math.h"
#include "src/__support/common.h"

#include <stdint.h>

#define InvLn2N exp2f_data.invln2_scaled
#define T exp2f_data.tab
#define C exp2f_data.poly_scaled
#define SHIFT exp2f_data.shift

namespace __llvm_libc {

float LLVM_LIBC_ENTRYPOINT(expf)(float x) {
  uint32_t abstop;
  uint64_t ki, t;
  // double_t for better performance on targets with FLT_EVAL_METHOD == 2.
  double_t kd, xd, z, r, r2, y, s;

  xd = static_cast<double_t>(x);
  abstop = top12_bits(x) & 0x7ff;
  if (unlikely(abstop >= top12_bits(88.0f))) {
    // |x| >= 88 or x is nan.
    if (as_uint32_bits(x) == as_uint32_bits(-INFINITY))
      return 0.0f;
    if (abstop >= top12_bits(INFINITY))
      return x + x;
    if (x > as_float(0x42b17217)) // x > log(0x1p128) ~= 88.72
      return overflow<float>(0);
    if (x < as_float(0xc2cff1b4)) // x < log(0x1p-150) ~= -103.97
      return underflow<float>(0);
    if (x < as_float(0xc2ce8ecf)) // x < log(0x1p-149) ~= -103.28
      return may_underflow<float>(0);
  }

  // x*N/Ln2 = k + r with r in [-1/2, 1/2] and int k.
  z = InvLn2N * xd;

  // Round and convert z to int, the result is in [-150*N, 128*N] and
  // ideally nearest int is used, otherwise the magnitude of r can be
  // bigger which gives larger approximation error.
  kd = static_cast<double>(z + SHIFT);
  ki = as_uint64_bits(kd);
  kd -= SHIFT;
  r = z - kd;

  // exp(x) = 2^(k/N) * 2^(r/N) ~= s *(C0*r^3 + C1*r^2 + C2*r + 1)
  t = T[ki % N];
  t += ki << (52 - EXP2F_TABLE_BITS);
  s = as_double(t);
  z = C[0] * r + C[1];
  r2 = r * r;
  y = C[2] * r + 1;
  y = z * r2 + y;
  y = y * s;
  return static_cast<float>(y);
}

} // namespace __llvm_libc
