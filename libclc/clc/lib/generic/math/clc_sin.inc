//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN __clc_sin(__CLC_FLOATN x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN, __clc_isinf(x));

  __CLC_FLOATN absx = __clc_fabs(x);

  __CLC_FLOATN r0;
  __CLC_INTN regn = __clc_argReductionS(&r0, absx);
  __CLC_SINCOS_RET_GENTYPE eval = __clc_sincos_reduced_eval(r0);

  __CLC_FLOATN s = (regn & 1) != 0 ? eval.cos : eval.sin;

  return __CLC_AS_FLOATN(__CLC_AS_UINTN(s) ^ (regn > 1 ? SIGNBIT_SP32 : 0) ^
                         (__CLC_AS_UINTN(x) ^ __CLC_AS_UINTN(absx)));
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sin(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_sin(__CLC_CONVERT_FLOATN(x)));
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sin(__CLC_GENTYPE x) {
  x = __clc_select(x, __CLC_GENTYPE_NAN,
                   __CLC_CONVERT_S_GENTYPE(__clc_isinf(x)));

  __CLC_GENTYPE absx = __clc_fabs(x);

  __CLC_BIT_INTN is_medium = absx < 0x1.0p+47;

  __CLC_INTN regn_m, regn_l;
  __CLC_GENTYPE r_m, r_l, rr_m, rr_l;

  __clc_remainder_piby2_medium(absx, &r_m, &rr_m, &regn_m);
  __clc_remainder_piby2_large(absx, &r_l, &rr_l, &regn_l);

  __CLC_GENTYPE r = is_medium ? r_m : r_l;
  __CLC_GENTYPE rr = is_medium ? rr_m : rr_l;
  __CLC_INTN regn = __CLC_CONVERT_INTN(is_medium) ? regn_m : regn_l;

  __CLC_GENTYPE sinval, cosval;
  __clc_sincos_piby4(r, rr, &sinval, &cosval);

  __CLC_LONGN s =
      __CLC_AS_LONGN(__CLC_CONVERT_BIT_INTN((regn & 1) != 0) ? cosval : sinval);

  s ^= (__CLC_CONVERT_BIT_INTN(regn > 1) << 63) ^
       (__CLC_CONVERT_BIT_INTN(x < 0.0) << 63);

  return __CLC_AS_GENTYPE(s);
}

#endif
