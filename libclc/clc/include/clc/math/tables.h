//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_MATH_TABLES_H__
#define __CLC_MATH_TABLES_H__

#include <clc/clctypes.h>

#define TABLE_SPACE __constant

#define TABLE_MANGLE(NAME) __clc_##NAME

#define DECLARE_TABLE(TYPE, NAME, LENGTH) TABLE_SPACE TYPE NAME[LENGTH]

#define TABLE_FUNCTION(TYPE, TABLE, NAME)                                      \
  TYPE TABLE_MANGLE(NAME)(size_t idx) { return TABLE[idx]; }

#define TABLE_FUNCTION_DECL(TYPE, NAME) TYPE TABLE_MANGLE(NAME)(size_t idx);

#define USE_TABLE(NAME, IDX) TABLE_MANGLE(NAME)(IDX)

TABLE_FUNCTION_DECL(float2, loge_tbl);
TABLE_FUNCTION_DECL(float, log_inv_tbl);
TABLE_FUNCTION_DECL(float2, log_inv_tbl_ep);
TABLE_FUNCTION_DECL(float2, log2_tbl);
TABLE_FUNCTION_DECL(float2, log10_tbl);
TABLE_FUNCTION_DECL(uint4, pibits_tbl);
TABLE_FUNCTION_DECL(float2, sinhcosh_tbl);
TABLE_FUNCTION_DECL(float2, cbrt_tbl);
TABLE_FUNCTION_DECL(float, exp_tbl);
TABLE_FUNCTION_DECL(float2, exp_tbl_ep);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

TABLE_FUNCTION_DECL(double2, ln_tbl);
TABLE_FUNCTION_DECL(double2, atan_jby256_tbl);
TABLE_FUNCTION_DECL(double2, two_to_jby64_ep_tbl);
TABLE_FUNCTION_DECL(double2, sinh_tbl);
TABLE_FUNCTION_DECL(double2, cosh_tbl);
TABLE_FUNCTION_DECL(double, cbrt_inv_tbl);
TABLE_FUNCTION_DECL(double2, cbrt_dbl_tbl);
TABLE_FUNCTION_DECL(double2, cbrt_rem_tbl);
TABLE_FUNCTION_DECL(double2, powlog_tbl);
TABLE_FUNCTION_DECL(double2, log_f_inv_tbl);

#endif // cl_khr_fp64

#endif // __CLC_MATH_TABLES_H__
