//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_ldexp.h>

#ifdef __HAS_LDEXPF__
// This defines all the ldexp(floatN, intN) variants.
_CLC_DEFINE_BINARY_BUILTIN(float, __clc_ldexp, __builtin_amdgcn_ldexpf, float, int);
#endif

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// This defines all the ldexp(doubleN, intN) variants.
_CLC_DEFINE_BINARY_BUILTIN(double, __clc_ldexp, __builtin_amdgcn_ldexp, double,
                           int);
#endif
