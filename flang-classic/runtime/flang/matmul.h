/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/** \file
 * \brief Matrix multiplication routines
 */

void f90_mm_cplx16_str1_mxv_(__CPLX16_T *, __CPLX16_T *, __CPLX16_T *,
                             __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_cplx16_str1_vxm_(__CPLX16_T *, __CPLX16_T *, __CPLX16_T *,
                             __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_cplx16_str1_(__CPLX16_T *, __CPLX16_T *, __CPLX16_T *, __INT_T *,
                         __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                         __INT_T *);
void f90_mm_cplx16_str1_mxv_t_(__CPLX16_T *, __CPLX16_T *, __CPLX16_T *,
                               __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_cplx16_str1_mxv_i8_(__CPLX16_T *, __CPLX16_T *, __CPLX16_T *,
                                __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_cplx16_str1_vxm_i8_(__CPLX16_T *, __CPLX16_T *, __CPLX16_T *,
                                __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_cplx16_str1_i8_(__CPLX16_T *, __CPLX16_T *, __CPLX16_T *, __INT_T *,
                            __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                            __INT_T *, __INT_T *);
void f90_mm_cplx16_str1_mxv_t_i8_(__CPLX16_T *, __CPLX16_T *, __CPLX16_T *,
                                  __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_cplx8_str1_mxv_(__CPLX8_T *, __CPLX8_T *, __CPLX8_T *, __INT_T *,
                            __INT_T *, __INT_T *, __INT_T *);
void f90_mm_cplx8_str1_vxm_(__CPLX8_T *, __CPLX8_T *, __CPLX8_T *, __INT_T *,
                            __INT_T *, __INT_T *, __INT_T *);
void f90_mm_cplx8_str1_(__CPLX8_T *, __CPLX8_T *, __CPLX8_T *, __INT_T *,
                        __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                        __INT_T *);
void f90_mm_cplx8_str1_mxv_t_(__CPLX8_T *, __CPLX8_T *, __CPLX8_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);

void f90_mm_cplx8_str1_mxv_i8_(__CPLX8_T *, __CPLX8_T *, __CPLX8_T *, __INT_T *,
                               __INT_T *, __INT_T *, __INT_T *);
void f90_mm_cplx8_str1_vxm_i8_(__CPLX8_T *, __CPLX8_T *, __CPLX8_T *, __INT_T *,
                               __INT_T *, __INT_T *, __INT_T *);
void f90_mm_cplx8_str1_i8_(__CPLX8_T *, __CPLX8_T *, __CPLX8_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                           __INT_T *, __INT_T *);
void f90_mm_cplx8_str1_mxv_t_i8_(__CPLX8_T *, __CPLX8_T *, __CPLX8_T *,
                                 __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_int1_str1_mxv_(__INT1_T *, __INT1_T *, __INT1_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int1_str1_vxm_(__INT1_T *, __INT1_T *, __INT1_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int1_str1_(__INT1_T *, __INT1_T *, __INT1_T *, __INT_T *, __INT_T *,
                       __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_int1_str1_mxv_i8_(__INT1_T *, __INT1_T *, __INT1_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int1_str1_vxm_i8_(__INT1_T *, __INT1_T *, __INT1_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int1_str1_i8_(__INT1_T *, __INT1_T *, __INT1_T *, __INT_T *,
                          __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                          __INT_T *);

void f90_mm_int2_str1_mxv_(__INT2_T *, __INT2_T *, __INT2_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int2_str1_vxm_(__INT2_T *, __INT2_T *, __INT2_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int2_str1_(__INT2_T *, __INT2_T *, __INT2_T *, __INT_T *, __INT_T *,
                       __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_int2_str1_mxv_i8_(__INT2_T *, __INT2_T *, __INT2_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int2_str1_vxm_i8_(__INT2_T *, __INT2_T *, __INT2_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int2_str1_i8_(__INT2_T *, __INT2_T *, __INT2_T *, __INT_T *,
                          __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                          __INT_T *);

void f90_mm_int4_str1_mxv_(__INT4_T *, __INT4_T *, __INT4_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int4_str1_vxm_(__INT4_T *, __INT4_T *, __INT4_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int4_str1_(__INT4_T *, __INT4_T *, __INT4_T *, __INT_T *, __INT_T *,
                       __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_int4_str1_mxv_i8_(__INT4_T *, __INT4_T *, __INT4_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int4_str1_vxm_i8_(__INT4_T *, __INT4_T *, __INT4_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int4_str1_i8_(__INT4_T *, __INT4_T *, __INT4_T *, __INT_T *,
                          __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                          __INT_T *);

void f90_mm_int8_str1_mxv_(__INT8_T *, __INT8_T *, __INT8_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int8_str1_vxm_(__INT8_T *, __INT8_T *, __INT8_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int8_str1_(__INT8_T *, __INT8_T *, __INT8_T *, __INT_T *, __INT_T *,
                       __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_int8_str1_mxv_i8_(__INT8_T *, __INT8_T *, __INT8_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int8_str1_vxm_i8_(__INT8_T *, __INT8_T *, __INT8_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);
void f90_mm_int8_str1_i8_(__INT8_T *, __INT8_T *, __INT8_T *, __INT_T *,
                          __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                          __INT_T *);

void f90_mm_real16_str1_mxv_(__REAL16_T *, __REAL16_T *, __REAL16_T *,
                             __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real16_str1_vxm_(__REAL16_T *, __REAL16_T *, __REAL16_T *,
                             __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real16_str1_(__REAL16_T *, __REAL16_T *, __REAL16_T *, __INT_T *,
                         __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                         __INT_T *);
void f90_mm_real16_str1_mxv_t_(__REAL16_T *, __REAL16_T *, __REAL16_T *,
                               __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real16_str1_mxv_i8_(__REAL16_T *, __REAL16_T *, __REAL16_T *,
                                __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real16_str1_vxm_i8_(__REAL16_T *, __REAL16_T *, __REAL16_T *,
                                __INT_T *, __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real16_str1_i8_(__REAL16_T *, __REAL16_T *, __REAL16_T *, __INT_T *,
                            __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                            __INT_T *, __INT_T *);
void f90_mm_real16_str1_mxv_t_i8_(__REAL16_T *, __REAL16_T *, __REAL16_T *,
                                  __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_real4_str1_mxv_(__REAL4_T *, __REAL4_T *, __REAL4_T *, __INT_T *,
                            __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real4_str1_vxm_(__REAL4_T *, __REAL4_T *, __REAL4_T *, __INT_T *,
                            __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real4_str1_(__REAL4_T *, __REAL4_T *, __REAL4_T *, __INT_T *,
                        __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                        __INT_T *);
void f90_mm_real4_str1_mxv_t_(__REAL4_T *, __REAL4_T *, __REAL4_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);

void f90_mm_real4_str1_mxv_i8_(__REAL4_T *, __REAL4_T *, __REAL4_T *, __INT_T *,
                               __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real4_str1_vxm_i8_(__REAL4_T *, __REAL4_T *, __REAL4_T *, __INT_T *,
                               __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real4_str1_i8_(__REAL4_T *, __REAL4_T *, __REAL4_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                           __INT_T *, __INT_T *);
void f90_mm_real4_str1_mxv_t_i8_(__REAL4_T *, __REAL4_T *, __REAL4_T *,
                                 __INT_T *, __INT_T *, __INT_T *, __INT_T *);

void f90_mm_real8_str1_mxv_(__REAL8_T *, __REAL8_T *, __REAL8_T *, __INT_T *,
                            __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real8_str1_vxm_(__REAL8_T *, __REAL8_T *, __REAL8_T *, __INT_T *,
                            __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real8_str1_(__REAL8_T *, __REAL8_T *, __REAL8_T *, __INT_T *,
                        __INT_T *, __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                        __INT_T *);
void f90_mm_real8_str1_mxv_t_(__REAL8_T *, __REAL8_T *, __REAL8_T *, __INT_T *,
                              __INT_T *, __INT_T *, __INT_T *);

void f90_mm_real8_str1_mxv_i8_(__REAL8_T *, __REAL8_T *, __REAL8_T *, __INT_T *,
                               __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real8_str1_vxm_i8_(__REAL8_T *, __REAL8_T *, __REAL8_T *, __INT_T *,
                               __INT_T *, __INT_T *, __INT_T *);
void f90_mm_real8_str1_i8_(__REAL8_T *, __REAL8_T *, __REAL8_T *, __INT_T *,
                           __INT_T *, __INT_T *, __INT_T *, __INT_T *,
                           __INT_T *, __INT_T *);
void f90_mm_real8_str1_mxv_t_i8_(__REAL8_T *, __REAL8_T *, __REAL8_T *,
                                 __INT_T *, __INT_T *, __INT_T *, __INT_T *);
