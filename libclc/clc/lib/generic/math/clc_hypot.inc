//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Returns sqrt(x*x + y*y) with no overflow or underflow unless the result
// warrants it

#if __CLC_FPSIZE == 32
_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_GENTYPE __clc_hypot(__CLC_GENTYPE x,
                                                            __CLC_GENTYPE y) {
  __CLC_GENTYPE a = __clc_fabs(x);
  __CLC_GENTYPE b = __clc_fabs(y);
  __CLC_GENTYPE t = __clc_fmax(a, b);
  __CLC_INTN e = __clc_frexp_exp(t);

  a = __clc_ldexp(a, -e);
  b = __clc_ldexp(b, -e);
  __CLC_GENTYPE ret = __clc_ldexp(__clc_sqrt_fast(__clc_mad(a, a, b * b)), e);

  return __clc_isinf(t) ? __CLC_GENTYPE_INF : ret;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_GENTYPE __clc_hypot(__CLC_GENTYPE x,
                                                            __CLC_GENTYPE y) {
  __CLC_GENTYPE a = __clc_fabs(x);
  __CLC_GENTYPE b = __clc_fabs(y);
  __CLC_GENTYPE t = __clc_fmax(a, b);
  __CLC_INTN e = __clc_frexp_exp(t);

  a = __clc_ldexp(a, -e);
  b = __clc_ldexp(b, -e);
  __CLC_GENTYPE ret = __clc_ldexp(__clc_sqrt_fast(__clc_mad(a, a, b * b)), e);

  ret = __clc_isunordered(x, y) ? __CLC_GENTYPE_NAN : ret;
  return __clc_isinf(x) || __clc_isinf(y) ? __CLC_GENTYPE_INF : ret;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_GENTYPE __clc_hypot(__CLC_GENTYPE x,
                                                            __CLC_GENTYPE y) {
  __CLC_FLOATN fx = __CLC_CONVERT_FLOATN(x);
  __CLC_FLOATN fy = __CLC_CONVERT_FLOATN(y);
  __CLC_FLOATN d2 = __clc_mad(fx, fx, fy * fy);

  __CLC_GENTYPE ret = __CLC_CONVERT_HALFN(__clc_sqrt_fast(d2));

  return __clc_isinf(x) || __clc_isinf(y) ? __CLC_GENTYPE_INF : ret;
}

#endif
