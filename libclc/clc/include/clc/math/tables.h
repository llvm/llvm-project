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

#define __CLC_TABLE_SPACE __constant

#define __CLC_TABLE_MANGLE(NAME) __clc_##NAME

#define __CLC_DECLARE_TABLE(TYPE, NAME, LENGTH)                                \
  __CLC_TABLE_SPACE TYPE NAME[LENGTH]

#define __CLC_TABLE_FUNCTION(TYPE, TABLE, NAME)                                \
  TYPE __CLC_TABLE_MANGLE(NAME)(size_t idx) { return TABLE[idx]; }

#define __CLC_TABLE_FUNCTION_VEC(TYPE, TABLE, NAME)                            \
  _CLC_DEF _CLC_OVERLOAD TYPE __CLC_TABLE_MANGLE(NAME)(int idx) {              \
    return TABLE[idx];                                                         \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##2 __CLC_TABLE_MANGLE(NAME)(int##2 idx) {        \
    return (TYPE##2){TABLE[idx.s0], TABLE[idx.s1]};                            \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##3 __CLC_TABLE_MANGLE(NAME)(int##3 idx) {        \
    return (TYPE##3){TABLE[idx.s0], TABLE[idx.s1], TABLE[idx.s2]};             \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##4 __CLC_TABLE_MANGLE(NAME)(int##4 idx) {        \
    return (TYPE##4){TABLE[idx.s0], TABLE[idx.s1], TABLE[idx.s2],              \
                     TABLE[idx.s3]};                                           \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##8 __CLC_TABLE_MANGLE(NAME)(int##8 idx) {        \
    return (TYPE##8){TABLE[idx.s0], TABLE[idx.s1], TABLE[idx.s2],              \
                     TABLE[idx.s3], TABLE[idx.s4], TABLE[idx.s5],              \
                     TABLE[idx.s6], TABLE[idx.s7]};                            \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD TYPE##16 __CLC_TABLE_MANGLE(NAME)(int##16 idx) {      \
    return (TYPE##16){                                                         \
        TABLE[idx.s0], TABLE[idx.s1], TABLE[idx.s2], TABLE[idx.s3],            \
        TABLE[idx.s4], TABLE[idx.s5], TABLE[idx.s6], TABLE[idx.s7],            \
        TABLE[idx.s8], TABLE[idx.s9], TABLE[idx.sA], TABLE[idx.sB],            \
        TABLE[idx.sC], TABLE[idx.sD], TABLE[idx.sE], TABLE[idx.sF]};           \
  }

#define __CLC_TABLE_FUNCTION_DECL(TYPE, NAME)                                  \
  TYPE __CLC_TABLE_MANGLE(NAME)(size_t idx);

#define __CLC_TABLE_FUNCTION_DECL_VEC(TYPE, NAME)                              \
  _CLC_DECL _CLC_OVERLOAD TYPE __CLC_TABLE_MANGLE(NAME)(int idx);              \
  _CLC_DECL _CLC_OVERLOAD TYPE##2 __CLC_TABLE_MANGLE(NAME)(int##2 idx);        \
  _CLC_DECL _CLC_OVERLOAD TYPE##3 __CLC_TABLE_MANGLE(NAME)(int##3 idx);        \
  _CLC_DECL _CLC_OVERLOAD TYPE##4 __CLC_TABLE_MANGLE(NAME)(int##4 idx);        \
  _CLC_DECL _CLC_OVERLOAD TYPE##8 __CLC_TABLE_MANGLE(NAME)(int##8 idx);        \
  _CLC_DECL _CLC_OVERLOAD TYPE##16 __CLC_TABLE_MANGLE(NAME)(int##16 idx);

#define __CLC_USE_TABLE(NAME, IDX) __CLC_TABLE_MANGLE(NAME)(IDX)

__CLC_TABLE_FUNCTION_DECL(float2, log2_tbl);
__CLC_TABLE_FUNCTION_DECL(float2, log10_tbl);

__CLC_TABLE_FUNCTION_DECL_VEC(float, log_inv_tbl_ep_head);
__CLC_TABLE_FUNCTION_DECL_VEC(float, log_inv_tbl_ep_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(float, loge_tbl_lo);
__CLC_TABLE_FUNCTION_DECL_VEC(float, loge_tbl_hi);
__CLC_TABLE_FUNCTION_DECL_VEC(float, log_inv_tbl);
__CLC_TABLE_FUNCTION_DECL_VEC(float, exp_tbl);
__CLC_TABLE_FUNCTION_DECL_VEC(float, exp_tbl_ep_head);
__CLC_TABLE_FUNCTION_DECL_VEC(float, exp_tbl_ep_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(float, cbrt_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(float, cbrt_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(float, sinhcosh_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(float, sinhcosh_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(ulong, pibits_tbl);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__CLC_TABLE_FUNCTION_DECL_VEC(double, ln_tbl_lo);
__CLC_TABLE_FUNCTION_DECL_VEC(double, ln_tbl_hi);
__CLC_TABLE_FUNCTION_DECL_VEC(double, atan_jby256_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(double, atan_jby256_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(double, two_to_jby64_ep_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(double, two_to_jby64_ep_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(double, sinh_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(double, sinh_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(double, cosh_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(double, cosh_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(double, cbrt_inv_tbl);
__CLC_TABLE_FUNCTION_DECL_VEC(double, cbrt_dbl_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(double, cbrt_dbl_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(double, cbrt_rem_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(double, cbrt_rem_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(double, powlog_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(double, powlog_tbl_tail);
__CLC_TABLE_FUNCTION_DECL_VEC(double, log_f_inv_tbl_head);
__CLC_TABLE_FUNCTION_DECL_VEC(double, log_f_inv_tbl_tail);

#endif // cl_khr_fp64

#endif // __CLC_MATH_TABLES_H__
