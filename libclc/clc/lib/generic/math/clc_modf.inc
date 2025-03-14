/*
 * Copyright (c) 2015 Advanced Micro Devices, Inc.
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
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_modf(__CLC_GENTYPE x,
                                                private __CLC_GENTYPE *iptr) {
  *iptr = __clc_trunc(x);
  return __clc_copysign(__clc_isinf(x) ? __CLC_FP_LIT(0.0) : x - *iptr, x);
}

#define CLC_MODF_DEF(addrspace)                                                \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_modf(                             \
      __CLC_GENTYPE x, addrspace __CLC_GENTYPE *iptr) {                        \
    __CLC_GENTYPE private_iptr;                                                \
    __CLC_GENTYPE ret = __clc_modf(x, &private_iptr);                          \
    *iptr = private_iptr;                                                      \
    return ret;                                                                \
  }

CLC_MODF_DEF(local);
CLC_MODF_DEF(global);

#undef CLC_MODF_DEF
