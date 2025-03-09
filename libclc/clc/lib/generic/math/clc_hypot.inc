/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
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

// Returns sqrt(x*x + y*y) with no overflow or underflow unless the result
// warrants it

#if __CLC_FPSIZE == 32
_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_hypot(__CLC_GENTYPE x,
                                                 __CLC_GENTYPE y) {
  __CLC_UINTN ux = __CLC_AS_UINTN(x);
  __CLC_UINTN aux = ux & EXSIGNBIT_SP32;
  __CLC_UINTN uy = __CLC_AS_UINTN(y);
  __CLC_UINTN auy = uy & EXSIGNBIT_SP32;
  __CLC_INTN c = aux > auy;
  ux = c ? aux : auy;
  uy = c ? auy : aux;

  __CLC_INTN xexp = __clc_clamp(
      __CLC_AS_INTN(ux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32, -126, 126);
  __CLC_GENTYPE fx_exp =
      __CLC_AS_GENTYPE((xexp + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);
  __CLC_GENTYPE fi_exp =
      __CLC_AS_GENTYPE((-xexp + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);
  __CLC_GENTYPE fx = __CLC_AS_GENTYPE(ux) * fi_exp;
  __CLC_GENTYPE fy = __CLC_AS_GENTYPE(uy) * fi_exp;

  __CLC_GENTYPE retval = __clc_sqrt(__clc_mad(fx, fx, fy * fy)) * fx_exp;

  retval = (ux > PINFBITPATT_SP32 || uy == 0) ? __CLC_AS_GENTYPE(ux) : retval;
  retval = (ux == PINFBITPATT_SP32 || uy == PINFBITPATT_SP32)
               ? __CLC_AS_GENTYPE((__CLC_UINTN)PINFBITPATT_SP32)
               : retval;
  return retval;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_hypot(__CLC_GENTYPE x,
                                                 __CLC_GENTYPE y) {
  __CLC_ULONGN ux = __CLC_AS_ULONGN(x) & ~SIGNBIT_DP64;
  __CLC_INTN xexp = __CLC_CONVERT_INTN(ux >> EXPSHIFTBITS_DP64);
  x = __CLC_AS_GENTYPE(ux);

  __CLC_ULONGN uy = __CLC_AS_ULONGN(y) & ~SIGNBIT_DP64;
  __CLC_INTN yexp = __CLC_CONVERT_INTN(uy >> EXPSHIFTBITS_DP64);
  y = __CLC_AS_GENTYPE(uy);

  __CLC_LONGN c = __CLC_CONVERT_LONGN(xexp > EXPBIAS_DP64 + 500 ||
                                      yexp > EXPBIAS_DP64 + 500);
  __CLC_GENTYPE preadjust = c ? 0x1.0p-600 : 1.0;
  __CLC_GENTYPE postadjust = c ? 0x1.0p+600 : 1.0;

  c = __CLC_CONVERT_LONGN(xexp < EXPBIAS_DP64 - 500 ||
                          yexp < EXPBIAS_DP64 - 500);
  preadjust = c ? 0x1.0p+600 : preadjust;
  postadjust = c ? 0x1.0p-600 : postadjust;

  __CLC_GENTYPE ax = x * preadjust;
  __CLC_GENTYPE ay = y * preadjust;

  // The post adjust may overflow, but this can't be avoided in any case
  __CLC_GENTYPE r = __clc_sqrt(__clc_fma(ax, ax, ay * ay)) * postadjust;

  // If the difference in exponents between x and y is large
  __CLC_GENTYPE s = x + y;
  c = __CLC_CONVERT_LONGN(__clc_abs(xexp - yexp) > MANTLENGTH_DP64 + 1);
  r = c ? s : r;

  // Check for NaN
  c = __clc_isnan(x) || __clc_isnan(y);
  r = c ? __CLC_AS_GENTYPE((__CLC_ULONGN)QNANBITPATT_DP64) : r;

  // If either is Inf, we must return Inf
  c = x == __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64) ||
      y == __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64);
  r = c ? __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64) : r;

  return r;
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_hypot(__CLC_GENTYPE x,
                                                 __CLC_GENTYPE y) {
  return __CLC_CONVERT_GENTYPE(
      __clc_hypot(__CLC_CONVERT_FLOATN(x), __CLC_CONVERT_FLOATN(y)));
}

#endif
