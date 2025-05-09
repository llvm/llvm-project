//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_ldexp.h>

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(float, ldexp, __clc_ldexp, float, int)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(double, ldexp, __clc_ldexp, double, int)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(half, ldexp, __clc_ldexp, half, int)

#endif

// This defines all the ldexp(GENTYPE, int) variants
#define __CLC_BODY <ldexp.inc>
#include <clc/math/gentype.inc>
