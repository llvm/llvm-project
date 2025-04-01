//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN sin(__CLC_FLOATN x) {
  __CLC_FLOATN absx = __clc_fabs(x);

  __CLC_FLOATN r0, r1;
  __CLC_INTN regn = __clc_argReductionS(&r0, &r1, absx);

  __CLC_FLOATN ss = __clc_sinf_piby4(r0, r1);
  __CLC_FLOATN cc = __clc_cosf_piby4(r0, r1);

  __CLC_FLOATN s = (regn & 1) != 0 ? cc : ss;
  s = __CLC_AS_FLOATN(__CLC_AS_INTN(s) ^ ((regn > 1) << 31) ^
                      (__CLC_AS_INTN(x) ^ __CLC_AS_INTN(absx)));

  s = __clc_select(s, __CLC_GENTYPE_NAN, __clc_isnan(x) || __clc_isinf(x));

  // Subnormals
  s = x == 0.0f ? x : s;

  return s;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE sin(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(sin(__CLC_CONVERT_FLOATN(x)));
}

#endif
