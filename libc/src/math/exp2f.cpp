//===-- Single-precision 2^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "exp_utils.h"
#include "math_utils.h"

#include "src/__support/common.h"
#include <math.h>

#include <stdint.h>

#define T exp2f_data.tab
#define C exp2f_data.poly
#define SHIFT exp2f_data.shift_scaled

namespace __llvm_libc {

float LLVM_LIBC_ENTRYPOINT(exp2f)(float x) {
  uint32_t abstop;
  uint64_t ki, t;
  // double_t for better performance on targets with FLT_EVAL_METHOD==2.
  double_t kd, xd, z, r, r2, y, s;

  xd = static_cast<double_t>(x);
  abstop = top12_bits(x) & 0x7ff;
  if (unlikely(abstop >= top12_bits(128.0f))) {
    // |x| >= 128 or x is nan.
    if (as_uint32_bits(x) == as_uint32_bits(-INFINITY))
      return 0.0f;
    if (abstop >= top12_bits(INFINITY))
      return x + x;
    if (x > 0.0f)
      return overflow<float>(0);
    if (x <= -150.0f)
      return underflow<float>(0);
    if (x < -149.0f)
      return may_underflow<float>(0);
  }

  // x = k/N + r with r in [-1/(2N), 1/(2N)] and int k.
  kd = static_cast<double>(xd + SHIFT);
  ki = as_uint64_bits(kd);
  kd -= SHIFT; // k/N for int k.
  r = xd - kd;

  // exp2(x) = 2^(k/N) * 2^r ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
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
