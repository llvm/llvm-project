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

#include <clc/clc.h>

#include "../../../generic/lib/clcmacro.h"

#ifdef __HAS_LDEXPF__
#define BUILTINF __builtin_amdgcn_ldexpf
#else
#include "math/clc_ldexp.h"
#define BUILTINF __clc_ldexp
#endif

// This defines all the ldexp(floatN, intN) variants.
_CLC_DEFINE_BINARY_BUILTIN(float, ldexp, BUILTINF, float, int);

#ifdef cl_khr_fp64
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    // This defines all the ldexp(doubleN, intN) variants.
  _CLC_DEFINE_BINARY_BUILTIN(double, ldexp, __builtin_amdgcn_ldexp, double, int);
#endif

// This defines all the ldexp(GENTYPE, int);
#define __CLC_BODY <../../../generic/lib/math/ldexp.inc>
#include <clc/math/gentype.inc>

#undef BUILTINF
