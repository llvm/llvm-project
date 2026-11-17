//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/float/definitions.h"
#include "clc/internal/clc.h"
#include "clc/math/clc_fabs.h"
#include "clc/relational/relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, int, __clc_isinf,
                             (__clc_fabs(x) == (float)INFINITY), float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_isinf(double) returns an int, but the vector
// versions return long.
_CLC_DEFINE_RELATIONAL_UNARY(int, long, __clc_isinf,
                             (__clc_fabs(x) == (double)INFINITY), double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_isinf(half) returns an int, but the vector
// versions return short.
_CLC_DEFINE_RELATIONAL_UNARY(int, short, __clc_isinf,
                             (__clc_fabs(x) == (half)INFINITY), half)

#endif
