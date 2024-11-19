/*
 * Copyright (c) 2014,2015 Advanced Micro Devices, Inc.
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

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/relational/clc_bitselect.h>

#define __CLC_BODY <bitselect.inc>
#include <clc/integer/gentype.inc>
#undef __CLC_BODY

#define FLOAT_BITSELECT(f_type, i_type, width)                                 \
  _CLC_OVERLOAD _CLC_DEF f_type##width bitselect(                              \
      f_type##width x, f_type##width y, f_type##width z) {                     \
    return __clc_bitselect(x, y, z);                                           \
  }

FLOAT_BITSELECT(float, uint, )
FLOAT_BITSELECT(float, uint, 2)
FLOAT_BITSELECT(float, uint, 3)
FLOAT_BITSELECT(float, uint, 4)
FLOAT_BITSELECT(float, uint, 8)
FLOAT_BITSELECT(float, uint, 16)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

FLOAT_BITSELECT(double, ulong, )
FLOAT_BITSELECT(double, ulong, 2)
FLOAT_BITSELECT(double, ulong, 3)
FLOAT_BITSELECT(double, ulong, 4)
FLOAT_BITSELECT(double, ulong, 8)
FLOAT_BITSELECT(double, ulong, 16)

#endif
