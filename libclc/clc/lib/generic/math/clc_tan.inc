//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_tan(__CLC_GENTYPE x) {
  __CLC_GENTYPE absx = __clc_fabs(x);
  __CLC_UINTN x_signbit = __CLC_AS_UINTN(x) & SIGNBIT_SP32;

  __CLC_GENTYPE r0, r1;
  __CLC_INTN regn = __clc_argReductionS(&r0, &r1, absx);

  __CLC_GENTYPE t = __clc_tanf_piby4(r0 + r1, regn);
  t = __CLC_AS_GENTYPE(__CLC_AS_UINTN(t) ^ x_signbit);

  t = __clc_select(t, __CLC_GENTYPE_NAN, __clc_isnan(x) || __clc_isinf(x));
  // Take care of subnormals
  t = (x == 0.0f) ? x : t;
  return t;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_tan(__CLC_GENTYPE x) {
  __CLC_GENTYPE y = __clc_fabs(x);

  __CLC_BIT_INTN is_medium = y < 0x1.0p+30;

  __CLC_INTN regn_m, regn_l;
  __CLC_GENTYPE r_m, r_l, rr_m, rr_l;

  __clc_remainder_piby2_medium(y, &r_m, &rr_m, &regn_m);
  __clc_remainder_piby2_large(y, &r_l, &rr_l, &regn_l);

  __CLC_GENTYPE r = is_medium ? r_m : r_l;
  __CLC_GENTYPE rr = is_medium ? rr_m : rr_l;
  __CLC_INTN regn = __CLC_CONVERT_INTN(is_medium) ? regn_m : regn_l;

  __CLC_GENTYPE lead, tail;
  __clc_tan_piby4(r, rr, &lead, &tail);

  __CLC_LONGN t =
      __CLC_AS_LONGN(__CLC_CONVERT_BIT_INTN((regn & 1) != 0) ? tail : lead);
  t ^= __CLC_CONVERT_BIT_INTN(x < 0.0) << 63;

  return __clc_isnan(x) || __clc_isinf(x) ? __CLC_GENTYPE_NAN
                                          : __CLC_AS_GENTYPE(t);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_tan(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_tan(__CLC_CONVERT_FLOATN(x)));
}

#endif
