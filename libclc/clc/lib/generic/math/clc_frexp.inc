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

#include <clc/clcmacro.h>
#include <clc/utils.h>

#define __CLC_AS_GENTYPE __CLC_XCONCAT(__clc_as_, __CLC_GENTYPE)
#define __CLC_AS_INTN __CLC_XCONCAT(__clc_as_, __CLC_INTN)

#if __CLC_FPSIZE == 32
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  __CLC_INTN i = __CLC_AS_INTN(x);
  __CLC_INTN ai = i & 0x7fffffff;
  __CLC_INTN d = ai > 0 & ai < 0x00800000;
  /* scale subnormal by 2^26 without multiplying */
  __CLC_GENTYPE s = __CLC_AS_GENTYPE(ai | 0x0d800000) - 0x1.0p-100f;
  ai = __clc_select(ai, __CLC_AS_INTN(s), d);
  __CLC_INTN e =
      (ai >> 23) - 126 - __clc_select((__CLC_INTN)0, (__CLC_INTN)26, d);
  __CLC_INTN t = ai == (__CLC_INTN)0 | e == (__CLC_INTN)129;
  i = (i & (__CLC_INTN)0x80000000) | (__CLC_INTN)0x3f000000 | (ai & 0x007fffff);
  *ep = __clc_select(e, (__CLC_INTN)0, t);
  return __clc_select(__CLC_AS_GENTYPE(i), x, t);
}
#endif

#if __CLC_FPSIZE == 16
#ifdef __CLC_SCALAR
#define __CLC_CONVERT_HALFN __clc_convert_half
#define __CLC_CONVERT_FLOATN __clc_convert_float
#else
#define __CLC_CONVERT_HALFN __CLC_XCONCAT(__clc_convert_half, __CLC_VECSIZE)
#define __CLC_CONVERT_FLOATN __CLC_XCONCAT(__clc_convert_float, __CLC_VECSIZE)
#endif
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  return __CLC_CONVERT_HALFN(__clc_frexp(__CLC_CONVERT_FLOATN(x), ep));
}
#undef __CLC_CONVERT_FLOATN
#undef __CLC_CONVERT_HALFN
#endif

#if __CLC_FPSIZE == 64
#ifdef __CLC_SCALAR
#define __CLC_AS_LONGN __clc_as_long
#define __CLC_LONGN long
#define __CLC_CONVERT_INTN __clc_convert_int
#else
#define __CLC_AS_LONGN __CLC_XCONCAT(__clc_as_long, __CLC_VECSIZE)
#define __CLC_LONGN __CLC_XCONCAT(long, __CLC_VECSIZE)
#define __CLC_CONVERT_INTN __CLC_XCONCAT(__clc_convert_int, __CLC_VECSIZE)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  __CLC_LONGN i = __CLC_AS_LONGN(x);
  __CLC_LONGN ai = i & 0x7fffffffffffffffL;
  __CLC_LONGN d = ai > 0 & ai < 0x0010000000000000L;
  // scale subnormal by 2^54 without multiplying
  __CLC_GENTYPE s = __CLC_AS_GENTYPE(ai | 0x0370000000000000L) - 0x1.0p-968;
  ai = __clc_select(ai, __CLC_AS_LONGN(s), d);
  __CLC_LONGN e = (ai >> 52) - (__CLC_LONGN)1022 -
                  __clc_select((__CLC_LONGN)0, (__CLC_LONGN)54, d);
  __CLC_LONGN t = ai == 0 | e == 1025;
  i = (i & (__CLC_LONGN)0x8000000000000000L) |
      (__CLC_LONGN)0x3fe0000000000000L |
      (ai & (__CLC_LONGN)0x000fffffffffffffL);
  *ep = __CLC_CONVERT_INTN(__clc_select(e, 0L, t));
  return __clc_select(__CLC_AS_GENTYPE(i), x, t);
}

#undef __CLC_AS_LONGN
#undef __CLC_LONGN
#undef __CLC_CONVERT_INTN
#endif

#undef __CLC_AS_GENTYPE
#undef __CLC_AS_INTN
