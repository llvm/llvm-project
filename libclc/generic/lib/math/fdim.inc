/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 * Copyright (c) 2016 Aaron Watry
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
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
