//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN cos(__CLC_FLOATN x) {
  __CLC_INTN ix = __CLC_AS_INTN(x);
  __CLC_INTN ax = ix & 0x7fffffff;
  __CLC_FLOATN dx = __CLC_AS_FLOATN(ax);

  __CLC_FLOATN r0, r1;
  __CLC_INTN regn = __clc_argReductionS(&r0, &r1, dx);

  __CLC_FLOATN ss = -__clc_sinf_piby4(r0, r1);
  __CLC_FLOATN cc = __clc_cosf_piby4(r0, r1);

  __CLC_FLOATN c = (regn & 1) != 0 ? ss : cc;
  c = __CLC_AS_FLOATN(__CLC_AS_INTN(c) ^ ((regn > 1) << 31));

  c = ax >= PINFBITPATT_SP32 ? __CLC_AS_FLOATN((__CLC_UINTN)QNANBITPATT_SP32)
                             : c;

  return c;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE cos(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(cos(__CLC_CONVERT_FLOATN(x)));
}

#endif
