//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_logb(__CLC_GENTYPE x) {
  __CLC_INTN ax = __CLC_AS_INTN(x) & EXSIGNBIT_SP32;
  __CLC_GENTYPE s = __CLC_CONVERT_GENTYPE(LOG_MAGIC_NUM_SP32 - __clc_clz(ax));
  __CLC_GENTYPE r =
      __CLC_CONVERT_GENTYPE((ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32);
  r = ax >= PINFBITPATT_SP32 ? __CLC_AS_GENTYPE(ax) : r;
  r = ax < 0x00800000 ? s : r;
  r = ax == 0 ? __CLC_AS_GENTYPE((__CLC_INTN)NINFBITPATT_SP32) : r;
  return r;
}

#endif

#if __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_logb(__CLC_GENTYPE x) {
  __CLC_LONGN ax = __CLC_AS_LONGN(x) & EXSIGNBIT_DP64;
  __CLC_GENTYPE s = __CLC_CONVERT_GENTYPE(LOG_MAGIC_NUM_DP64 - __clc_clz(ax));
  __CLC_GENTYPE r =
      __CLC_CONVERT_GENTYPE((ax >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64);
  r = ax >= PINFBITPATT_DP64 ? __CLC_AS_GENTYPE(ax) : r;
  r = ax < 0x0010000000000000L ? s : r;
  r = ax == 0L ? __CLC_AS_GENTYPE((__CLC_LONGN)NINFBITPATT_DP64) : r;
  return r;
}

#endif

#if __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_logb(__CLC_GENTYPE x) {
  __CLC_SHORTN ax = __CLC_AS_SHORTN(x) & (__CLC_SHORTN)EXSIGNBIT_FP16;
  __CLC_GENTYPE s = __CLC_CONVERT_GENTYPE((__CLC_SHORTN)LOG_MAGIC_NUM_FP16 -
                                          (__CLC_SHORTN)__clc_clz(ax));
  __CLC_GENTYPE r = __CLC_CONVERT_GENTYPE(
      (ax >> (__CLC_SHORTN)EXPSHIFTBITS_FP16) - (__CLC_SHORTN)EXPBIAS_FP16);
  r = ax >= (__CLC_SHORTN)PINFBITPATT_FP16 ? __CLC_AS_GENTYPE(ax) : r;
  r = ax < (__CLC_SHORTN)0x0400 ? s : r;
  r = ax == (__CLC_SHORTN)0 ? __CLC_AS_GENTYPE((__CLC_SHORTN)NINFBITPATT_FP16)
                            : r;
  return r;
}

#endif
