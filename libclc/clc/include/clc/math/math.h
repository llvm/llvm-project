//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MATH_MATH_H__
#define __CLC_MATH_MATH_H__

#include <clc/clc_as_type.h>
#include <clc/clcfunc.h>
#include <clc/math/clc_subnormal_config.h>

#define SNAN 0x001
#define QNAN 0x002
#define NINF 0x004
#define NNOR 0x008
#define NSUB 0x010
#define NZER 0x020
#define PZER 0x040
#define PSUB 0x080
#define PNOR 0x100
#define PINF 0x200

#if (defined __AMDGCN__ || defined __R600__) && !defined __HAS_FMAF__
#define __CLC_HAVE_HW_FMA32() (0)
#elif defined(CLC_SPIRV)
bool __attribute__((noinline)) __clc_runtime_has_hw_fma32(void);
#define __CLC_HAVE_HW_FMA32() __clc_runtime_has_hw_fma32()
#else
#define __CLC_HAVE_HW_FMA32() (1)
#endif

#define HAVE_BITALIGN() (0)
#define HAVE_FAST_FMA32() (0)

#define MATH_DIVIDE(X, Y) ((X) / (Y))
#define MATH_RECIP(X) (1.0f / (X))
#define MATH_SQRT(X) sqrt(X)

#define SIGNBIT_SP32 0x80000000
#define EXSIGNBIT_SP32 0x7fffffff
#define EXPBITS_SP32 0x7f800000
#define MANTBITS_SP32 0x007fffff
#define ONEEXPBITS_SP32 0x3f800000
#define TWOEXPBITS_SP32 0x40000000
#define HALFEXPBITS_SP32 0x3f000000
#define IMPBIT_SP32 0x00800000
#define QNANBITPATT_SP32 0x7fc00000
#define INDEFBITPATT_SP32 0xffc00000
#define PINFBITPATT_SP32 0x7f800000
#define NINFBITPATT_SP32 0xff800000
#define NUMEXPBITS_SP32 8
#define EXPBIAS_SP32 127
#define EXPSHIFTBITS_SP32 23
#define BIASEDEMIN_SP32 1
#define EMIN_SP32 -126
#define BIASEDEMAX_SP32 254
#define EMAX_SP32 127
#define LAMBDA_SP32 1.0e30
#define MANTLENGTH_SP32 24
#define BASEDIGITS_SP32 7

#define LOG_MAGIC_NUM_SP32 (1 + NUMEXPBITS_SP32 - EXPBIAS_SP32)

_CLC_OVERLOAD _CLC_INLINE float __clc_flush_denormal_if_not_supported(float x) {
  int ix = __clc_as_int(x);
  if (!__clc_fp32_subnormals_supported() && ((ix & EXPBITS_SP32) == 0) &&
      ((ix & MANTBITS_SP32) != 0)) {
    ix &= SIGNBIT_SP32;
    x = __clc_as_float(ix);
  }
  return x;
}

#ifdef cl_khr_fp64

#define SIGNBIT_DP64 0x8000000000000000L
#define EXSIGNBIT_DP64 0x7fffffffffffffffL
#define EXPBITS_DP64 0x7ff0000000000000L
#define MANTBITS_DP64 0x000fffffffffffffL
#define ONEEXPBITS_DP64 0x3ff0000000000000L
#define TWOEXPBITS_DP64 0x4000000000000000L
#define HALFEXPBITS_DP64 0x3fe0000000000000L
#define IMPBIT_DP64 0x0010000000000000L
#define QNANBITPATT_DP64 0x7ff8000000000000L
#define INDEFBITPATT_DP64 0xfff8000000000000L
#define PINFBITPATT_DP64 0x7ff0000000000000L
#define NINFBITPATT_DP64 0xfff0000000000000L
#define NUMEXPBITS_DP64 11
#define EXPBIAS_DP64 1023
#define EXPSHIFTBITS_DP64 52
#define BIASEDEMIN_DP64 1
#define EMIN_DP64 -1022
#define BIASEDEMAX_DP64 2046 /* 0x7fe */
#define EMAX_DP64 1023       /* 0x3ff */
#define LAMBDA_DP64 1.0e300
#define MANTLENGTH_DP64 53
#define BASEDIGITS_DP64 15

#define LOG_MAGIC_NUM_DP64 (1 + NUMEXPBITS_DP64 - EXPBIAS_DP64)

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#define SIGNBIT_FP16 0x8000
#define EXSIGNBIT_FP16 0x7fff
#define EXPBITS_FP16 0x7c00
#define MANTBITS_FP16 0x03ff
#define PINFBITPATT_FP16 0x7c00
#define NINFBITPATT_FP16 0xfc00
#define NUMEXPBITS_FP16 5
#define EXPBIAS_FP16 15
#define EXPSHIFTBITS_FP16 10

#define LOG_MAGIC_NUM_FP16 (1 + NUMEXPBITS_FP16 - EXPBIAS_FP16)

#endif // cl_khr_fp16

#endif // __CLC_MATH_MATH_H__
