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

#define CLC_TABLE_FUNCTION(TYPE, TABLE, NAME)                                  \
  _CLC_DEF _CLC_OVERLOAD TYPE TABLE_MANGLE(NAME)(int idx) {                    \
    return TABLE[idx];                                                         \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##2 TABLE_MANGLE(NAME)(int##2 idx) {              \
    return (TYPE##2){TABLE[idx.s0], TABLE[idx.s1]};                            \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##3 TABLE_MANGLE(NAME)(int##3 idx) {              \
    return (TYPE##3){TABLE[idx.s0], TABLE[idx.s1], TABLE[idx.s2]};             \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##4 TABLE_MANGLE(NAME)(int##4 idx) {              \
    return (TYPE##4){TABLE[idx.s0], TABLE[idx.s1], TABLE[idx.s2],              \
                     TABLE[idx.s3]};                                           \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##8 TABLE_MANGLE(NAME)(int##8 idx) {              \
    return (TYPE##8){TABLE[idx.s0], TABLE[idx.s1], TABLE[idx.s2],              \
                     TABLE[idx.s3], TABLE[idx.s4], TABLE[idx.s5],              \
                     TABLE[idx.s6], TABLE[idx.s7]};                            \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##16 TABLE_MANGLE(NAME)(int##16 idx) {            \
    return (TYPE##16){                                                         \
        TABLE[idx.s0], TABLE[idx.s1], TABLE[idx.s2], TABLE[idx.s3],            \
        TABLE[idx.s4], TABLE[idx.s5], TABLE[idx.s6], TABLE[idx.s7],            \
        TABLE[idx.s8], TABLE[idx.s9], TABLE[idx.sA], TABLE[idx.sB],            \
        TABLE[idx.sC], TABLE[idx.sD], TABLE[idx.sE], TABLE[idx.sF]};           \
  }

#define TABLE_FUNCTION_DECL(TYPE, NAME) TYPE TABLE_MANGLE(NAME)(size_t idx);

#define CLC_TABLE_FUNCTION_DECL(TYPE, NAME)                                    \
  _CLC_DECL _CLC_OVERLOAD TYPE TABLE_MANGLE(NAME)(int idx);                    \
  _CLC_DECL _CLC_OVERLOAD TYPE##2 TABLE_MANGLE(NAME)(int##2 idx);              \
  _CLC_DECL _CLC_OVERLOAD TYPE##3 TABLE_MANGLE(NAME)(int##3 idx);              \
  _CLC_DECL _CLC_OVERLOAD TYPE##4 TABLE_MANGLE(NAME)(int##4 idx);              \
  _CLC_DECL _CLC_OVERLOAD TYPE##8 TABLE_MANGLE(NAME)(int##8 idx);              \
  _CLC_DECL _CLC_OVERLOAD TYPE##16 TABLE_MANGLE(NAME)(int##16 idx);

#define USE_TABLE(NAME, IDX) TABLE_MANGLE(NAME)(IDX)

TABLE_FUNCTION_DECL(float2, log2_tbl);
TABLE_FUNCTION_DECL(float2, log10_tbl);
TABLE_FUNCTION_DECL(uint4, pibits_tbl);

CLC_TABLE_FUNCTION_DECL(float, log_inv_tbl_ep_head);
CLC_TABLE_FUNCTION_DECL(float, log_inv_tbl_ep_tail);
CLC_TABLE_FUNCTION_DECL(float, loge_tbl_lo);
CLC_TABLE_FUNCTION_DECL(float, loge_tbl_hi);
CLC_TABLE_FUNCTION_DECL(float, log_inv_tbl);
CLC_TABLE_FUNCTION_DECL(float, exp_tbl);
CLC_TABLE_FUNCTION_DECL(float, exp_tbl_ep_head);
CLC_TABLE_FUNCTION_DECL(float, exp_tbl_ep_tail);
CLC_TABLE_FUNCTION_DECL(float, cbrt_tbl_head);
CLC_TABLE_FUNCTION_DECL(float, cbrt_tbl_tail);
CLC_TABLE_FUNCTION_DECL(float, sinhcosh_tbl_head);
CLC_TABLE_FUNCTION_DECL(float, sinhcosh_tbl_tail);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

CLC_TABLE_FUNCTION_DECL(double, ln_tbl_lo);
CLC_TABLE_FUNCTION_DECL(double, ln_tbl_hi);
CLC_TABLE_FUNCTION_DECL(double, atan_jby256_tbl_head);
CLC_TABLE_FUNCTION_DECL(double, atan_jby256_tbl_tail);
CLC_TABLE_FUNCTION_DECL(double, two_to_jby64_ep_tbl_head);
CLC_TABLE_FUNCTION_DECL(double, two_to_jby64_ep_tbl_tail);
CLC_TABLE_FUNCTION_DECL(double, sinh_tbl_head);
CLC_TABLE_FUNCTION_DECL(double, sinh_tbl_tail);
CLC_TABLE_FUNCTION_DECL(double, cosh_tbl_head);
CLC_TABLE_FUNCTION_DECL(double, cosh_tbl_tail);
CLC_TABLE_FUNCTION_DECL(double, cbrt_inv_tbl);
CLC_TABLE_FUNCTION_DECL(double, cbrt_dbl_tbl_head);
CLC_TABLE_FUNCTION_DECL(double, cbrt_dbl_tbl_tail);
CLC_TABLE_FUNCTION_DECL(double, cbrt_rem_tbl_head);
CLC_TABLE_FUNCTION_DECL(double, cbrt_rem_tbl_tail);
CLC_TABLE_FUNCTION_DECL(double, powlog_tbl_head);
CLC_TABLE_FUNCTION_DECL(double, powlog_tbl_tail);
CLC_TABLE_FUNCTION_DECL(double, log_f_inv_tbl_head);
CLC_TABLE_FUNCTION_DECL(double, log_f_inv_tbl_tail);

#endif // cl_khr_fp64

#endif // __CLC_MATH_TABLES_H__
