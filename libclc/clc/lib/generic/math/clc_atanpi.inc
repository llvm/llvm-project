//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanpi(__CLC_GENTYPE x) {
  __CLC_GENTYPE v = __clc_fabs(x);
  __CLC_INTN g = v > 1.0f;

  __CLC_GENTYPE vi = __clc_recip_fast(v);
  v = g ? vi : v;

  __CLC_GENTYPE a = __clc_atanpi_reduced(v);

  __CLC_GENTYPE y = 0.5f - a;
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanpi(__CLC_GENTYPE x) {
  __CLC_GENTYPE v = __clc_fabs(x);
  __CLC_LONGN g = v > 1.0;

  v = g ? (1.0 / v) : v;

  __CLC_GENTYPE a = __clc_atanpi_reduced(v);
  __CLC_GENTYPE y = 0.5 - a;
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanpi(__CLC_GENTYPE x) {
  __CLC_GENTYPE v = __clc_fabs(x);
  __CLC_SHORTN g = v > 1.0h;

  __CLC_GENTYPE vi = __clc_recip_fast(v);
  v = g ? vi : v;

  __CLC_GENTYPE a = __clc_atanpi_reduced(v);

  __CLC_GENTYPE y = 0.5h - a;
  a = g ? y : a;

  return __clc_copysign(a, x);
}

#endif
