//===-- Single-precision sin function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math_utils.h"
#include "sincosf_utils.h"

#include "include/math.h"
#include "src/__support/common.h"

#include <stdint.h>

namespace __llvm_libc {

// Fast sinf implementation. Worst-case ULP is 0.5607, maximum relative
// error is 0.5303 * 2^-23. A single-step range reduction is used for
// small values. Large inputs have their range reduced using fast integer
// arithmetic.
float LLVM_LIBC_ENTRYPOINT(sinf)(float y) {
  double x = y;
  double s;
  int n;
  const sincos_t *p = &__sincosf_table[0];

  if (abstop12(y) < abstop12(pio4)) {
    s = x * x;

    if (unlikely(abstop12(y) < abstop12(as_float(0x39800000)))) {
      if (unlikely(abstop12(y) < abstop12(as_float(0x800000))))
        // Force underflow for tiny y.
        force_eval<float>(s);
      return y;
    }

    return sinf_poly(x, s, p, 0);
  } else if (likely(abstop12(y) < abstop12(120.0f))) {
    x = reduce_fast(x, p, &n);

    // Setup the signs for sin and cos.
    s = p->sign[n & 3];

    if (n & 2)
      p = &__sincosf_table[1];

    return sinf_poly(x * s, x * x, p, n);
  } else if (abstop12(y) < abstop12(INFINITY)) {
    uint32_t xi = as_uint32_bits(y);
    int sign = xi >> 31;

    x = reduce_large(xi, &n);

    // Setup signs for sin and cos - include original sign.
    s = p->sign[(n + sign) & 3];

    if ((n + sign) & 2)
      p = &__sincosf_table[1];

    return sinf_poly(x * s, x * x, p, n);
  }

  return invalid(y);
}

} // namespace __llvm_libc
