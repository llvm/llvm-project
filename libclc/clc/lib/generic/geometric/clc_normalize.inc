//===----------------------------------------------------------------------===//
//

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if (__CLC_VECSIZE_OR_1 == 1 || __CLC_VECSIZE_OR_1 == 2 ||                     \
     __CLC_VECSIZE_OR_1 == 3 || __CLC_VECSIZE_OR_1 == 4)

// Until we have a native FP16 implementation, go via FP32
#if __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_normalize(__CLC_GENTYPE p) {
  return __CLC_CONVERT_GENTYPE(__clc_normalize(__CLC_CONVERT_FLOATN(p)));
}

// Scalar normalize
#elif defined(__CLC_SCALAR)

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_normalize(__CLC_GENTYPE p) {
  return __clc_sign(p);
}

// Vector normalize
#else

#if __CLC_FPSIZE == 32

#define MIN_VAL FLT_MIN
#define MAX_SQRT 0x1.0p+86F
#if __CLC_VECSIZE_OR_1 == 2
#define MIN_SQRT 0x1.0p-65F
#else
#define MIN_SQRT 0x1.0p-66F
#endif

#elif __CLC_FPSIZE == 64

#define MIN_VAL DBL_MIN
#define MAX_SQRT 0x1.0p+563
#if __CLC_VECSIZE_OR_1 == 2
#define MIN_SQRT 0x1.0p-513
#else
#define MIN_SQRT 0x1.0p-514
#endif

#else
#error "Invalid FP size"
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_normalize(__CLC_GENTYPE p) {
  if (__clc_all(p == __CLC_FP_LIT(0.0))) {
    return p;
  }

  __CLC_SCALAR_GENTYPE l2 = __clc_dot(p, p);

  if (l2 < MIN_VAL) {
    p *= MAX_SQRT;
    l2 = __clc_dot(p, p);
  } else if (l2 == INFINITY) {
    p *= MIN_SQRT;
    l2 = __clc_dot(p, p);
    if (l2 == INFINITY) {
      p = __clc_copysign(__clc_select((__CLC_GENTYPE)__CLC_FP_LIT(0.0),
                                      (__CLC_GENTYPE)__CLC_FP_LIT(1.0),
                                      __clc_isinf(p)),
                         p);
      l2 = __clc_dot(p, p);
    }
  }
  return p * __clc_rsqrt(l2);
}

#undef MIN_VAL
#undef MIN_SQRT
#undef MAX_SQRT

#endif

#endif
