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

#if __CLC_FPSIZE == 32
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  __CLC_INTN i = __CLC_AS_INTN(x);
  __CLC_INTN ai = i & EXSIGNBIT_SP32;

  // Scale subnormal by 2^26 without multiplying
  __CLC_INTN is_subnormal = ai > 0 && ai < 0x00800000;
  __CLC_GENTYPE s = __CLC_AS_GENTYPE(ai | 0x0d800000) - 0x1.0p-100f;
  ai = __clc_select(ai, __CLC_AS_INTN(s), is_subnormal);
  __CLC_INTN e = (ai >> EXPSHIFTBITS_SP32) - 126 -
                 __clc_select((__CLC_INTN)0, (__CLC_INTN)26, is_subnormal);

  i = (i & (__CLC_INTN)SIGNBIT_SP32) | (__CLC_INTN)HALFEXPBITS_SP32 |
      (ai & (__CLC_INTN)MANTBITS_SP32);

  __CLC_INTN is_inf_nan_or_zero =
      x == __CLC_FP_LIT(0.0) || __clc_isinf(x) || __clc_isnan(x);
  *ep = __clc_select(e, (__CLC_INTN)0, is_inf_nan_or_zero);
  return __clc_select(__CLC_AS_GENTYPE(i), x, is_inf_nan_or_zero);
}
#endif

#if __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  return __CLC_CONVERT_HALFN(__clc_frexp(__CLC_CONVERT_FLOATN(x), ep));
}

#endif

#if __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE
__clc_frexp(__CLC_GENTYPE x, __CLC_ADDRESS_SPACE __CLC_INTN *ep) {
  __CLC_LONGN i = __CLC_AS_LONGN(x);
  __CLC_LONGN ai = i & EXSIGNBIT_DP64;

  // Scale subnormal by 2^54 without multiplying
  __CLC_LONGN is_subnormal = ai > 0 && ai < 0x0010000000000000L;
  __CLC_GENTYPE s = __CLC_AS_GENTYPE(ai | 0x0370000000000000L) - 0x1.0p-968;
  ai = __clc_select(ai, __CLC_AS_LONGN(s), is_subnormal);
  __CLC_LONGN e = (ai >> EXPSHIFTBITS_DP64) - (__CLC_LONGN)1022 -
                  __clc_select((__CLC_LONGN)0, (__CLC_LONGN)54, is_subnormal);

  i = (i & (__CLC_LONGN)SIGNBIT_DP64) | (__CLC_LONGN)HALFEXPBITS_DP64 |
      (ai & (__CLC_LONGN)MANTBITS_DP64);

  __CLC_LONGN is_inf_nan_or_zero =
      x == __CLC_FP_LIT(0.0) || __clc_isinf(x) || __clc_isnan(x);
  *ep = __CLC_CONVERT_INTN(__clc_select(e, 0L, is_inf_nan_or_zero));
  return __clc_select(__CLC_AS_GENTYPE(i), x, is_inf_nan_or_zero);
}

#endif
