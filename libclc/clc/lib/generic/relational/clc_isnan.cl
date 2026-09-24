//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/internal/clc.h"
#include "clc/relational/relational.h"

_CLC_DEFINE_RELATIONAL_UNARY(int, int, __clc_isnan, (x != x), float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_isnan(double) returns an int, but the vector
// versions return a long.
_CLC_DEFINE_RELATIONAL_UNARY(int, long, __clc_isnan, (x != x), double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_isnan(half) returns an int, but the vector
// versions return a short.
_CLC_DEFINE_RELATIONAL_UNARY(int, short, __clc_isnan, (x != x), half)

#endif

