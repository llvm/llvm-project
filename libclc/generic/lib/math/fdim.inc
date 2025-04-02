//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32
#ifdef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE fdim(__CLC_GENTYPE x, __CLC_GENTYPE y) {
    if (__builtin_isnan(x) || __builtin_isnan(y))
        return as_float(QNANBITPATT_SP32);
    return fmax(x - y, 0.0f);
}
#define __CLC_FDIM_VEC(width) \
_CLC_OVERLOAD _CLC_DEF float##width fdim(float##width x, float##width y) { \
    /* Determine if x or y is NaN. */ \
    /* Vector true is -1, i.e. all-bits-set, and NaN==NaN is false. */ \
    /* If either is NaN, then ~((x==x) & (y==y)) will be 0 (e.g. ~(-1)), as will n. */ \
    int##width n = ~((x == x) & (y == y)) & QNANBITPATT_SP32; \
    /* Calculate x-y if x>y, otherwise positive 0, again taking */ \
    /* advantage of vector true being all-bits-set. */ \
    int##width r = (x > y) & as_int##width(x - y); \
    return as_float##width(n | r); \
}
__CLC_FDIM_VEC(2)
__CLC_FDIM_VEC(3)
__CLC_FDIM_VEC(4)
__CLC_FDIM_VEC(8)
__CLC_FDIM_VEC(16)
#undef __CLC_FDIM_VEC
#endif
#endif

#if __CLC_FPSIZE == 64
#ifdef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE fdim(__CLC_GENTYPE x, private __CLC_GENTYPE y) {
    long n = -(isnan(x) | isnan(y)) & QNANBITPATT_DP64;
    long r = -(x > y) & as_long(x - y);
    return as_double(n | r);
}
#define __CLC_FDIM_VEC(width) \
_CLC_OVERLOAD _CLC_DEF double##width fdim(double##width x, double##width y) { \
    /* See comment in float implementation for explanation. */ \
    long##width n = ~((x == x) & (y == y)) & QNANBITPATT_DP64; \
    long##width r = (x > y) & as_long##width(x - y); \
    return as_double##width(n | r); \
}
__CLC_FDIM_VEC(2)
__CLC_FDIM_VEC(3)
__CLC_FDIM_VEC(4)
__CLC_FDIM_VEC(8)
__CLC_FDIM_VEC(16)
#undef __CLC_FDIM_VEC
#endif
#endif

#if __CLC_FPSIZE == 16
#ifdef __CLC_SCALAR
#define QNANBITPATT_FP16 ((short)0x7e00)
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE fdim(__CLC_GENTYPE x,
                                          private __CLC_GENTYPE y) {
  short n = -(isnan(x) | isnan(y)) & QNANBITPATT_FP16;
  short r = -(x > y) & as_short(x - y);
  return as_half((short)(n | r));
}
#define __CLC_FDIM_VEC(width)                                                  \
  _CLC_OVERLOAD _CLC_DEF half##width fdim(half##width x, half##width y) {      \
    /* See comment in float implementation for explanation. */                 \
    short##width n = ~((x == x) & (y == y)) & QNANBITPATT_FP16;                \
    short##width r = (x > y) & as_short##width(x - y);                         \
    return as_half##width(n | r);                                              \
  }
__CLC_FDIM_VEC(2)
__CLC_FDIM_VEC(3)
__CLC_FDIM_VEC(4)
__CLC_FDIM_VEC(8)
__CLC_FDIM_VEC(16)
#undef __CLC_FDIM_VEC
#endif
#endif
