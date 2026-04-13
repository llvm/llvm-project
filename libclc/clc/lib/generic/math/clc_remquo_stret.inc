//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

#if __CLC_FPSIZE == 32 || __CLC_FPSIZE == 64
#define __CLC_REMQUO_EVAL_TYPE __CLC_GENTYPE
#define __CLC_CONVERT_REMQUO_EVAL_TYPE __CLC_CONVERT_GENTYPE
#define __CLC_S_EVAL_TYPE __CLC_S_GENTYPE
#define __CLC_CONVERT_S_EVAL_TYPE __CLC_CONVERT_S_GENTYPE
#elif __CLC_FPSIZE == 16
#define __CLC_REMQUO_EVAL_TYPE __CLC_FLOATN
#define __CLC_CONVERT_REMQUO_EVAL_TYPE __CLC_CONVERT_FLOATN
#define __CLC_S_EVAL_TYPE __CLC_INTN
#define __CLC_CONVERT_S_EVAL_TYPE __CLC_CONVERT_INTN
#endif

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_REMQUO_RET_GENTYPE
__clc_remquo_stret(__CLC_GENTYPE x, __CLC_GENTYPE y) {
  // How many bits of the quotient per iteration

#if __CLC_FPSIZE == 32
  const __CLC_INTN bits = 12;
  const __CLC_GENTYPE max_exp = 0x1.0p+127f;
#elif __CLC_FPSIZE == 64
  const __CLC_INTN bits = 26;
  const __CLC_GENTYPE max_exp = 0x1.0p+1023;
#elif __CLC_FPSIZE == 16
  const __CLC_INTN bits = 11;
  const __CLC_GENTYPE max_exp = 0x1.0p+15h;
#endif

  // Track low 7 bits of the integral quotient.
  __CLC_INTN q7;

  __CLC_REMQUO_EVAL_TYPE ax = __CLC_CONVERT_REMQUO_EVAL_TYPE(__clc_fabs(x));
  __CLC_REMQUO_EVAL_TYPE ay = __CLC_CONVERT_REMQUO_EVAL_TYPE(__clc_fabs(y));

  __CLC_GENTYPE ret;

  if (ax > ay) {
    __CLC_INTN ex, ey;

    __CLC_REMQUO_EVAL_TYPE mx = __clc_frexp(ax, &ex);
    --ex;

    __CLC_REMQUO_EVAL_TYPE my = __clc_frexp(ay, &ey);
    --ey;

    ax = __clc_ldexp(mx, bits);
    ay = __clc_ldexp(my, 1);

    __CLC_INTN nb = ex - ey;
    __CLC_REMQUO_EVAL_TYPE ayinv = __clc_recip_fast(ay);

    __CLC_INTN qacc = 0;

    while (nb > bits) {
      __CLC_REMQUO_EVAL_TYPE q = __clc_rint(ax * ayinv);

#if __CLC_FPSIZE == 16
      ax = __clc_mad(-q, ay, ax);
#else
      ax = __clc_fma(-q, ay, ax);
#endif
      __CLC_S_GENTYPE clt = ax < (__CLC_REMQUO_EVAL_TYPE)0.0;
      __CLC_REMQUO_EVAL_TYPE axp = ax + ay;
      ax = clt ? axp : ax;
      ax = __clc_ldexp(ax, bits);

      __CLC_INTN iq = __CLC_CONVERT_INTN(q);
      iq -= __CLC_CONVERT_INTN(clt) ? 1 : 0;
      qacc = (qacc << bits) | iq;

      nb -= bits;
    }

    ax = __clc_ldexp(ax, nb - bits + 1);

    __CLC_INTN is_odd;

    // Final iteration
    {
      __CLC_REMQUO_EVAL_TYPE q = __clc_rint(ax * ayinv);
#if __CLC_FPSIZE == 16
      ax = __clc_mad(-q, ay, ax);
#else
      ax = __clc_fma(-q, ay, ax);
#endif

      __CLC_S_GENTYPE clt = ax < (__CLC_REMQUO_EVAL_TYPE)0.0;
      __CLC_REMQUO_EVAL_TYPE axp = ax + ay;
      ax = clt ? axp : ax;
      __CLC_INTN iq = __CLC_CONVERT_INTN(q);
      iq -= __CLC_CONVERT_INTN(clt) ? 1 : 0;

      qacc = (qacc << (nb + 1)) | iq;
      is_odd = (iq & 1) != 0;
    }

    // Adjust ax so that it is the range (-y/2, y/2]
    // We need to choose the even integer when x/y is midway between two
    // integers
    __CLC_S_EVAL_TYPE aq = ((__CLC_REMQUO_EVAL_TYPE)2.0 * ax > ay) |
                           (__CLC_CONVERT_S_EVAL_TYPE(is_odd) &
                            ((__CLC_REMQUO_EVAL_TYPE)2.0 * ax == ay));
    ax = ax - (aq ? ay : (__CLC_REMQUO_EVAL_TYPE)0.0);

    ax = __clc_ldexp(ax, ey);
    qacc += aq ? 1 : 0;

    __CLC_S_GENTYPE qneg = __clc_signbit(x) ^ __clc_signbit(y) ? -1 : 0;
    q7 = ((qacc & 0x7f) ^ qneg) - qneg;

    ret = __clc_signbit(x) ? -ax : ax;
  } else {
    __CLC_S_EVAL_TYPE c = (ax > (__CLC_REMQUO_EVAL_TYPE)0.5 * ay);
    if (__CLC_FPSIZE != 16)
      c |= (ay < max_exp && (__CLC_REMQUO_EVAL_TYPE)2.0 * ax > ay);

    __CLC_CHARN qsgn = __CLC_CONVERT_CHARN(__clc_signbit(x) == __clc_signbit(y))
                           ? (__CLC_CHARN)1
                           : (__CLC_CHARN)-1;

    __CLC_GENTYPE t = __clc_mad(y, -__CLC_CONVERT_GENTYPE(qsgn), x);
    ret = c ? t : __clc_flush_if_daz(x);
    q7 = c ? qsgn : 0;

    __CLC_GENTYPE zero = __clc_copysign(__CLC_FP_LIT(0.0), x);
    ret = ax == ay ? zero : ret;
    q7 = ax == ay ? qsgn : q7;
  }

  ret = y == __CLC_FP_LIT(0.0) ? __CLC_GENTYPE_NAN : ret;
  q7 = y == __CLC_FP_LIT(0.0) ? 0 : q7;

  __CLC_S_GENTYPE finite = !__clc_isnan(y) && __clc_isfinite(x);

  // A defined 0 result for quo with a nan result is an additional OpenCL
  // requirement beyond standard C.
  __CLC_REMQUO_RET_GENTYPE result;
  result.quo = finite ? q7 : 0;
  result.rem = finite ? ret : __CLC_GENTYPE_NAN;

  return result;
}

#undef __CLC_REMQUO_EVAL_TYPE
#undef __CLC_CONVERT_REMQUO_EVAL_TYPE
#undef __CLC_S_EVAL_TYPE
#undef __CLC_CONVERT_S_EVAL_TYPE

#endif // __CLC_SCALAR
