//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN __clc_cos(__CLC_FLOATN x) {
  __CLC_FLOATN absx = __clc_fabs(x);

  __CLC_FLOATN r0, r1;
  __CLC_INTN regn = __clc_argReductionS(&r0, &r1, absx);

  __CLC_FLOATN ss = -__clc_sinf_piby4(r0, r1);
  __CLC_FLOATN cc = __clc_cosf_piby4(r0, r1);

  __CLC_FLOATN c = (regn & 1) != 0 ? ss : cc;
  c = __CLC_AS_FLOATN(__CLC_AS_INTN(c) ^ ((regn > 1) << 31));

  c = __clc_select(c, __CLC_GENTYPE_NAN, __clc_isnan(x) || __clc_isinf(x));

  return c;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_cos(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_cos(__CLC_CONVERT_FLOATN(x)));
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_cos(__CLC_GENTYPE x) {
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
  sinval = -sinval;

  __CLC_LONGN c =
      __CLC_AS_LONGN(__CLC_CONVERT_BIT_INTN((regn & 1) != 0) ? sinval : cosval);
  c ^= __CLC_CONVERT_BIT_INTN(regn > 1) << 63;

  return __clc_isnan(absx) | __clc_isinf(absx) ? __CLC_GENTYPE_NAN
                                               : __CLC_AS_GENTYPE(c);
}

#endif
