//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_ldexp.h>

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifdef __AMDGCN__
#define __clc_builtin_rsq __builtin_amdgcn_rsq
#else
#define __clc_builtin_rsq __builtin_r600_recipsqrt_ieee
#endif

_CLC_OVERLOAD _CLC_DEF double __clc_sqrt(double x) {
  uint vcc = x < 0x1p-767;
  uint exp0 = vcc ? 0x100 : 0;
  unsigned exp1 = vcc ? 0xffffff80 : 0;

  double v01 = __clc_ldexp(x, exp0);
  double v23 = __clc_builtin_rsq(v01);
  double v45 = v01 * v23;
  v23 = v23 * 0.5;

  double v67 = __clc_fma(-v23, v45, 0.5);
  v45 = __clc_fma(v45, v67, v45);
  double v89 = __clc_fma(-v45, v45, v01);
  v23 = __clc_fma(v23, v67, v23);
  v45 = __clc_fma(v89, v23, v45);
  v67 = __clc_fma(-v45, v45, v01);
  v23 = __clc_fma(v67, v23, v45);

  v23 = __clc_ldexp(v23, exp1);
  return (x == __builtin_inf() || (x == 0.0)) ? v01 : v23;
}

#define __CLC_DOUBLE_ONLY
#define __CLC_FUNCTION __clc_sqrt
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>

#endif
