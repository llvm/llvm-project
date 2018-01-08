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
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE frexp(float x, private int *ep) {
    int i = as_int(x);
    int ai = i & 0x7fffffff;
    int d = ai > 0 & ai < 0x00800000;
    // scale subnormal by 2^26 without multiplying
    float s = as_float(ai | 0x0d800000) - 0x1.0p-100F;
    ai = d ? as_int(s) : ai;
    int e = (ai >> 23) - 126 - (d ? 26 : 0);
    int t = ai == 0 | e == 129;
    i = (i & 0x80000000) | 0x3f000000 | (ai & 0x007fffff);
    *ep = t ? 0 : e;
    return t ? x : as_float(i);
}
#define __CLC_FREXP_VEC(width) \
_CLC_OVERLOAD _CLC_DEF float##width frexp(float##width x, private int##width *ep) { \
    int##width i = as_int##width(x); \
    int##width ai = i & 0x7fffffff; \
    int##width d = ai > 0 & ai < 0x00800000; \
    /* scale subnormal by 2^26 without multiplying */ \
    float##width s = as_float##width(ai | 0x0d800000) - 0x1.0p-100F; \
    ai = bitselect(ai, as_int##width(s), d); \
    int##width e = (ai >> 23) - 126 - bitselect((int##width)0, (int##width)26, d); \
    int##width t = ai == (int##width)0 | e == (int##width)129; \
    i = (i & (int##width)0x80000000) | (int##width)0x3f000000 | (ai & 0x007fffff); \
    *ep = bitselect(e, (int##width)0, t); \
    return bitselect(as_float##width(i), x, as_float##width(t)); \
}
__CLC_FREXP_VEC(2)
__CLC_FREXP_VEC(3)
__CLC_FREXP_VEC(4)
__CLC_FREXP_VEC(8)
__CLC_FREXP_VEC(16)
#undef __CLC_FREXP_VEC
#endif
#endif

#if __CLC_FPSIZE == 64
#ifdef __CLC_SCALAR
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE frexp(__CLC_GENTYPE x, private __CLC_INTN *ep) {
    long i = as_long(x);
    long ai = i & 0x7fffffffffffffffL;
    int d = ai > 0 & ai < 0x0010000000000000L;
    // scale subnormal by 2^54 without multiplying
    double s = as_double(ai | 0x0370000000000000L) - 0x1.0p-968;
    ai = d ? as_long(s) : ai;
    int e = (int)(ai >> 52) - 1022 - (d ? 54 : 0);
    int t = ai == 0 | e == 1025;
    i = (i & 0x8000000000000000L) | 0x3fe0000000000000L | (ai & 0x000fffffffffffffL);
    *ep = t ? 0 : e;
    return t ? x : as_double(i);
}
#define __CLC_FREXP_VEC(width) \
_CLC_OVERLOAD _CLC_DEF double##width frexp(double##width x, private int##width *ep) { \
    long##width i = as_long##width(x); \
    long##width ai = i & 0x7fffffffffffffffL; \
    long##width d = ai > 0 & ai < 0x0010000000000000L; \
    /* scale subnormal by 2^54 without multiplying */ \
    double##width s = as_double##width(ai | 0x0370000000000000L) - 0x1.0p-968; \
    ai = bitselect(ai, as_long##width(s), d); \
    int##width e = convert_int##width(ai >> 52) - 1022 - bitselect((int##width)0, (int##width)54, convert_int##width(d)); \
    int##width t = convert_int##width(ai == (long##width)0) | (e == (int##width)129); \
    i = (i & (long##width)0x8000000000000000L) | (long##width)0x3fe0000000000000L | (ai & 0x000fffffffffffffL); \
    *ep = bitselect(e, (int##width)0, t); \
    return bitselect(as_double##width(i), x, as_double##width(convert_long##width(t))); \
}
__CLC_FREXP_VEC(2)
__CLC_FREXP_VEC(3)
__CLC_FREXP_VEC(4)
__CLC_FREXP_VEC(8)
__CLC_FREXP_VEC(16)
#undef __CLC_FREXP_VEC
#endif
#endif

#define __CLC_FREXP_DEF(addrspace) \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE frexp(__CLC_GENTYPE x, addrspace __CLC_INTN *iptr) { \
    __CLC_INTN private_iptr; \
    __CLC_GENTYPE ret = frexp(x, &private_iptr); \
    *iptr = private_iptr; \
    return ret; \
}

__CLC_FREXP_DEF(local);
__CLC_FREXP_DEF(global);

#undef __CLC_FREXP_DEF
