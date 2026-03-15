//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sinpi(__CLC_GENTYPE x) {
  __CLC_INTN ix = __CLC_AS_INTN(x);
  __CLC_INTN xsgn = ix & (__CLC_INTN)0x80000000;
  ix ^= xsgn;
  __CLC_GENTYPE absx = __clc_fabs(x);
  __CLC_INTN iax = __CLC_CONVERT_INTN(absx);
  __CLC_GENTYPE r = absx - __CLC_CONVERT_GENTYPE(iax);
  __CLC_INTN xodd =
      xsgn ^ ((iax & 0x1) != 0 ? (__CLC_INTN)0x80000000 : (__CLC_INTN)0);

  // Initialize with return for +-Inf and NaN
  __CLC_INTN ir = QNANBITPATT_SP32;

  // 2^23 <= |x| < Inf, the result is always integer
  ir = ix < PINFBITPATT_SP32 ? xsgn : ir;

  // 0x1.0p-7 <= |x| < 2^23, result depends on which 0.25 interval

  // r < 1.0
  __CLC_GENTYPE a = 1.0f - r;
  __CLC_INTN e = 0;

  // r <= 0.75
  __CLC_INTN c = r <= 0.75f;
  a = c ? r - 0.5f : a;
  e = c ? 1 : e;

  // r < 0.5
  c = r < 0.5f;
  a = c ? 0.5f - r : a;

  // 0 < r <= 0.25
  c = r <= 0.25f;
  a = c ? r : a;
  e = c ? 0 : e;

  __CLC_GENTYPE sinval, cosval;
  __clc_sincos_piby4(a * M_PI_F, &sinval, &cosval);
  __CLC_INTN jr = xodd ^ __CLC_AS_INTN(e != 0 ? cosval : sinval);

  ir = ix < 0x4b000000 ? jr : ir;

  return __CLC_AS_GENTYPE(ir);
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sinpi(__CLC_GENTYPE x) {
  __CLC_LONGN ix = __CLC_AS_LONGN(x);
  __CLC_LONGN xsgn = ix & (__CLC_LONGN)0x8000000000000000L;
  ix ^= xsgn;
  __CLC_GENTYPE absx = __clc_fabs(x);
  __CLC_LONGN iax = __CLC_CONVERT_LONGN(absx);
  __CLC_GENTYPE r = absx - __CLC_CONVERT_GENTYPE(iax);
  __CLC_LONGN xodd =
      xsgn ^
      ((iax & 0x1L) != 0 ? (__CLC_LONGN)0x8000000000000000L : (__CLC_LONGN)0L);

  // Initialize with return for +-Inf and NaN
  __CLC_LONGN ir = QNANBITPATT_DP64;

  // 2^23 <= |x| < Inf, the result is always integer
  ir = ix < PINFBITPATT_DP64 ? xsgn : ir;

  // 0x1.0p-7 <= |x| < 2^23, result depends on which 0.25 interval

  // r < 1.0
  __CLC_GENTYPE a = 1.0 - r;
  __CLC_LONGN e = 0;

  //  r <= 0.75
  __CLC_LONGN c = r <= 0.75;
  __CLC_GENTYPE t = r - 0.5;
  a = c ? t : a;
  e = c ? 1 : e;

  // r < 0.5
  c = r < 0.5;
  t = 0.5 - r;
  a = c ? t : a;

  // r <= 0.25
  c = r <= 0.25;
  a = c ? r : a;
  e = c ? 0 : e;

  __CLC_GENTYPE api = a * M_PI;

  __CLC_GENTYPE sinval, cosval;
  __clc_sincos_piby4(api, 0.0, &sinval, &cosval);
  __CLC_LONGN jr = xodd ^ __CLC_AS_LONGN(e != 0 ? cosval : sinval);

  ir = absx < 0x1.0p+52 ? jr : ir;

  return __CLC_AS_GENTYPE(ir);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sinpi(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_sinpi(__CLC_CONVERT_FLOATN(x)));
}

#endif
