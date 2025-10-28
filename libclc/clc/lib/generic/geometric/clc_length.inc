//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_VECSIZE_OR_1 == 1

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_length(__CLC_GENTYPE p) {
  return __clc_fabs(p);
}

#elif (__CLC_VECSIZE_OR_1 == 2 || __CLC_VECSIZE_OR_1 == 3 ||                   \
       __CLC_VECSIZE_OR_1 == 4)

#if __CLC_FPSIZE == 32
#define __CLC_GENTYPE_MIN FLT_MIN
#define __CLC_GENTYPE_MIN_SCALE_UP 0x1.0p+86F
#define __CLC_GENTYPE_MIN_SCALE_DOWN 0x1.0p-86F
#define __CLC_GENTYPE_INF_SCALE_UP 0x1.0p+65F
#define __CLC_GENTYPE_INF_SCALE_DOWN 0x1.0p-65F
#elif __CLC_FPSIZE == 64
#define __CLC_GENTYPE_MIN DBL_MIN
#define __CLC_GENTYPE_MIN_SCALE_UP 0x1.0p+563
#define __CLC_GENTYPE_MIN_SCALE_DOWN 0x1.0p-563
#define __CLC_GENTYPE_INF_SCALE_UP 0x1.0p+513
#define __CLC_GENTYPE_INF_SCALE_DOWN 0x1.0p-513
#elif __CLC_FPSIZE == 16
#define __CLC_GENTYPE_MIN HALF_MIN
#define __CLC_GENTYPE_MIN_SCALE_UP 0x1.0p+12H
#define __CLC_GENTYPE_MIN_SCALE_DOWN 0x1.0p-12H
#define __CLC_GENTYPE_INF_SCALE_UP 0x1.0p+7H
#define __CLC_GENTYPE_INF_SCALE_DOWN 0x1.0p-7H
#else
#error "Unsupported FP size"
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_SCALAR_GENTYPE __clc_length(__CLC_GENTYPE p) {
  __CLC_SCALAR_GENTYPE l2 = __clc_dot(p, p);

  if (l2 < __CLC_GENTYPE_MIN) {
    p *= __CLC_GENTYPE_MIN_SCALE_UP;
    return __clc_sqrt(__clc_dot(p, p)) * __CLC_GENTYPE_MIN_SCALE_DOWN;
  }

  if (l2 == INFINITY) {
    p *= __CLC_GENTYPE_INF_SCALE_DOWN;
    return __clc_sqrt(__clc_dot(p, p)) * __CLC_GENTYPE_INF_SCALE_UP;
  }

  return __clc_sqrt(l2);
}

#undef __CLC_GENTYPE_INF_SCALE_DOWN
#undef __CLC_GENTYPE_INF_SCALE_UP
#undef __CLC_GENTYPE_MIN_SCALE_DOWN
#undef __CLC_GENTYPE_MIN_SCALE_UP
#undef __CLC_GENTYPE_MIN

#endif
