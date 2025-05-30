//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_INTN __clc_ilogb(__CLC_GENTYPE x) {
  __CLC_UINTN ux = __CLC_AS_UINTN(x);
  __CLC_UINTN ax = ux & EXSIGNBIT_SP32;
  __CLC_INTN rs = (__CLC_INTN)LOG_MAGIC_NUM_SP32 -
                  __CLC_AS_INTN(__clc_clz(ux & MANTBITS_SP32));
  __CLC_INTN r = __CLC_AS_INTN(ax >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
  r = ax < 0x00800000U ? rs : r;
  r = ax == 0 ? FP_ILOGB0 : r;

  // We could merge those 2 tests and have:
  //
  //    r = ax >= EXPBITS_SP32 ? 0x7fffffff : r
  //
  // since FP_ILOGBNAN is set to INT_MAX, but it's clearer this way and
  // FP_ILOGBNAN can change without requiring changes to __clc_ilogb() code.
  r = ax > EXPBITS_SP32 ? FP_ILOGBNAN : r;
  r = ax == EXPBITS_SP32 ? 0x7fffffff : r;
  return r;
}

#endif

#if __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_INTN __clc_ilogb(__CLC_GENTYPE x) {
  __CLC_ULONGN ux = __CLC_AS_ULONGN(x);
  __CLC_ULONGN ax = ux & ~SIGNBIT_DP64;
  __CLC_INTN rs = (__CLC_INTN)LOG_MAGIC_NUM_DP64 -
                  __CLC_CONVERT_INTN(__clc_clz(ax & MANTBITS_DP64));
  __CLC_INTN r = __CLC_CONVERT_INTN(ax >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64;
  r = __CLC_CONVERT_INTN(ax < 0x0010000000000000UL) ? rs : r;
  r = __CLC_CONVERT_INTN(ax == 0UL) ? (__CLC_INTN)FP_ILOGB0 : r;

  // We could merge those 2 tests and have:
  //
  //    r = ax >= 0x7ff0000000000000UL ? 0x7fffffff : r
  //
  // since FP_ILOGBNAN is set to INT_MAX, but it's clearer this way and
  // FP_ILOGBNAN can change without requiring changes to __clc_ilogb() code.
  r = __CLC_CONVERT_INTN(ax > 0x7ff0000000000000UL) ? FP_ILOGBNAN : r;
  r = __CLC_CONVERT_INTN(ax == 0x7ff0000000000000UL) ? 0x7fffffff : r;
  return r;
}

#endif

#if __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_INTN __clc_ilogb(__CLC_GENTYPE x) {
  __CLC_USHORTN ux = __CLC_AS_USHORTN(x);
  __CLC_USHORTN ax = ux & (__CLC_USHORTN)EXSIGNBIT_FP16;
  __CLC_USHORTN mantx = ux & (__CLC_USHORTN)MANTBITS_FP16;
  __CLC_INTN rs =
      (__CLC_INTN)LOG_MAGIC_NUM_FP16 - __CLC_CONVERT_INTN(__clc_clz(mantx));
  __CLC_INTN r =
      __CLC_CONVERT_INTN(ax >> (__CLC_USHORTN)EXPSHIFTBITS_FP16) - EXPBIAS_FP16;
  r = __CLC_CONVERT_INTN(ax < (__CLC_USHORTN)0x0400U) ? rs : r;
  r = __CLC_CONVERT_INTN(ax == (__CLC_USHORTN)0) ? (__CLC_INTN)FP_ILOGB0 : r;

  // We could merge those 2 tests and have:
  //
  //    r = ax >= EXPBITS_FP16 ? 0x7fffffff : r
  //
  // since FP_ILOGBNAN is set to INT_MAX, but it's clearer this way and
  // FP_ILOGBNAN can change without requiring changes to __clc_ilogb() code.
  r = __CLC_CONVERT_INTN(ax > (__CLC_USHORTN)EXPBITS_FP16) ? FP_ILOGBNAN : r;
  r = __CLC_CONVERT_INTN(ax == (__CLC_USHORTN)EXPBITS_FP16) ? 0x7fffffff : r;
  return r;
}

#endif
