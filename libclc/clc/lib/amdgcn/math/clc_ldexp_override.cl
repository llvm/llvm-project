//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/math/clc_ldexp.h>

#define __CLC_FUNCTION __clc_ldexp
#define __CLC_ARG2_TYPE int
#define __CLC_MIN_VECSIZE 1

#ifdef __HAS_LDEXPF__
// This defines all the ldexp(floatN, intN) variants.
#define __CLC_FLOAT_ONLY
#define __CLC_IMPL_FUNCTION __builtin_amdgcn_ldexpf
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_IMPL_FUNCTION
#endif

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// This defines all the ldexp(doubleN, intN) variants.
#define __CLC_DOUBLE_ONLY
#define __CLC_IMPL_FUNCTION __builtin_amdgcn_ldexp
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_IMPL_FUNCTION
#endif
