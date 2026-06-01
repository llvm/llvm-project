//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/clc_convert.h"
#include "clc/integer/clc_clz.h"
#include "clc/internal/clc.h"
#include "clc/math/clc_floor.h"
#include "clc/math/clc_fma.h"
#include "clc/math/clc_frexp.h"
#include "clc/math/clc_ldexp.h"
#include "clc/math/clc_mad.h"
#include "clc/math/clc_native_divide.h"
#include "clc/math/clc_rint.h"
#include "clc/math/clc_sincos_helpers.h"
#include "clc/math/clc_trunc.h"
#include "clc/math/math.h"
#include "clc/relational/clc_isinf.h"
#include "clc/relational/clc_isnan.h"

#ifndef __opencl_c_int64
#include "clc/integer/clc_mul_hi.h"
#define __CLC_FULL_MUL(A, B, HI, LO)                                           \
  LO = A * B;                                                                  \
  HI = __clc_mul_hi(A, B)

#define __CLC_FULL_MAD(A, B, C, HI, LO)                                        \
  LO = ((A) * (B) + (C));                                                      \
  HI = __clc_mul_hi(A, B);                                                     \
  HI += LO < C ? 1U : 0U;
#endif

#define bitalign(hi, lo, shift) __builtin_elementwise_fshr(hi, lo, shift)

#define __CLC_FLOAT_ONLY
#define __CLC_BODY "clc_sincos_helpers.inc"

#include "clc/math/gentype.inc"

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "clc/math/clc_ep.h"
#include "clc/math/clc_fract.h"
#include "clc/math/clc_get_twobypi_bits.h"
#include "clc/math/tables.h"

#define __CLC_DOUBLE_ONLY
#define __CLC_BODY "clc_sincos_helpers_fp64.inc"

#include "clc/math/gentype.inc"

#endif

#ifdef cl_khr_fp16
#define __CLC_HALF_ONLY
#define __CLC_BODY "clc_sincos_helpers_fp16.inc"
#include "clc/math/gentype.inc"
#endif
