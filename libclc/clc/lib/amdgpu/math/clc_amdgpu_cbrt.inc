//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_SCALAR

#define __CLC_ONE_THIRD_F32 (__CLC_FLOATN)0x1.555556p-2f

static _CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_GENTYPE
__clc_fixup_cbrt_sign(__CLC_GENTYPE magnitude, __CLC_GENTYPE x) {
  // Is normal or subnormal.
  __CLC_S_GENTYPE is_zero = x == __CLC_FP_LIT(0.0);
  __CLC_S_GENTYPE is_inf = __clc_isinf(x);
  magnitude = is_zero || is_inf ? x : magnitude;
  return __clc_copysign(magnitude, x);
}

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_cbrt(__CLC_FLOATN x) {
  x = __clc_flush_if_daz(x);

  __CLC_FLOATN ax = __clc_fabs(x);
  __CLC_INTN denorm_or_zero = ax < FLT_MIN;

  if (!__clc_denormals_are_zero_fp32()) {
    ax = denorm_or_zero ? __clc_ldexp(ax, 24) : ax;
  }

  __CLC_GENTYPE z = __clc_exp2_fast(__CLC_ONE_THIRD_F32 * __clc_log2_fast(ax));
  z = __clc_mad(__clc_mad(__clc_recip_fast(z * z), -ax, z),
                -__CLC_ONE_THIRD_F32, z);

  if (!__clc_denormals_are_zero_fp32()) {
    z = denorm_or_zero ? __clc_ldexp(z, -8) : z;
  }

  return __clc_fixup_cbrt_sign(z, x);
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_cbrt(__CLC_DOUBLEN x) {
  __CLC_GENTYPE a = __clc_fabs(x);
  __CLC_INTN e3 = __clc_frexp_exp(a);
  __CLC_INTN e = __CLC_CONVERT_INTN(
      __clc_rint(__CLC_ONE_THIRD_F32 * __CLC_CONVERT_FLOATN(e3)));
  a = __clc_ldexp(a, -3 * e);

  __CLC_GENTYPE c = __CLC_CONVERT_DOUBLEN(__clc_exp2_fast(
      __CLC_ONE_THIRD_F32 * __clc_log2_fast(__CLC_CONVERT_FLOATN(a))));
  __CLC_GENTYPE c2 = c * c;
  c = __clc_mad(
      c, __clc_div_fast(__clc_mad(-c, c2, a), __clc_mad(c + c, c2, a)), c);

  c = __clc_ldexp(c, e);

  return __clc_fixup_cbrt_sign(c, x);
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_cbrt(__CLC_HALFN x) {
  __CLC_HALFN absx = __clc_fabs(x);
  __CLC_FLOATN absx_float = __CLC_CONVERT_FLOATN(absx);
  __CLC_FLOATN result_float =
      __clc_exp2_fast(__CLC_ONE_THIRD_F32 * __clc_log2_fast(absx_float));
  __CLC_HALFN ret = __CLC_CONVERT_HALFN(result_float);
  return __clc_fixup_cbrt_sign(ret, x);
}

#endif

#endif // __CLC_SCALAR
