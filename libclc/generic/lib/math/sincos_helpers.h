//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcfunc.h>
#include <clc/clctypes.h>

_CLC_DECL float __clc_sinf_piby4(float x, float y);
_CLC_DECL float __clc_cosf_piby4(float x, float y);
_CLC_DECL float __clc_tanf_piby4(float x, int y);
_CLC_DECL int __clc_argReductionS(private float *r, private float *rr, float x);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DECL void __clc_remainder_piby2_medium(double x, private double *r,
                                            private double *rr,
                                            private int *regn);
_CLC_DECL void __clc_remainder_piby2_large(double x, private double *r,
                                           private double *rr,
                                           private int *regn);
_CLC_DECL double2 __clc_sincos_piby4(double x, double xx);

#endif
