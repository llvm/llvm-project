//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_DEF _CLC_OVERLOAD __CLC_INTN
__clc_piArgReductionS(private __CLC_GENTYPE *reduced, __CLC_GENTYPE x) {
  __CLC_GENTYPE unused;
  __CLC_GENTYPE t =
      __CLC_FP_LIT(2.0) * __clc_fract(__CLC_FP_LIT(0.5) * x, &unused);
  x = x > __CLC_FP_LIT(1.0) ? t : x;
  t = __clc_rint(__CLC_FP_LIT(2.0) * x);

  *reduced = __clc_mad(t, __CLC_FP_LIT(-0.5), x);
  return __CLC_CONVERT_INTN(t) & 0x3;
}
