//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN cos(__CLC_FLOATN x) {
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

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE cos(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(cos(__CLC_CONVERT_FLOATN(x)));
}

#endif
