/* file: libm_error_codes.h */


/*
// Copyright (c) 2000 - 2004, Intel Corporation
// All rights reserved.
//
// Contributed 2000 by the Intel Numerics Group, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// * The name of Intel Corporation may not be used to endorse or promote
// products derived from this software without specific prior written
// permission.

//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of this code, and requests that all
// problem reports or change requests be submitted to it directly at
// http://www.intel.com/software/products/opensource/libraries/num.htm.
//

// Abstract:
// ========================================================================
// This file contains the interface to the Intel exception dispatcher.
//
//
// History:
// ========================================================================
// 12/15/2004 Initial version - extracted from libm_support.h
//
*/

#if !defined(__LIBM_ERROR_CODES_H__)
#define __LIBM_ERROR_CODES_H__

typedef enum
{
  logl_zero=0,   logl_negative,                  /*  0,  1 */
  log_zero,      log_negative,                   /*  2,  3 */
  logf_zero,     logf_negative,                  /*  4,  5 */
  log10l_zero,   log10l_negative,                /*  6,  7 */
  log10_zero,    log10_negative,                 /*  8,  9 */
  log10f_zero,   log10f_negative,                /* 10, 11 */
  expl_overflow, expl_underflow,                 /* 12, 13 */
  exp_overflow,  exp_underflow,                  /* 14, 15 */
  expf_overflow, expf_underflow,                 /* 16, 17 */
  powl_overflow, powl_underflow,                 /* 18, 19 */
  powl_zero_to_zero,                             /* 20     */
  powl_zero_to_negative,                         /* 21     */
  powl_neg_to_non_integer,                       /* 22     */
  powl_nan_to_zero,                              /* 23     */
  pow_overflow,  pow_underflow,                  /* 24, 25 */
  pow_zero_to_zero,                              /* 26     */
  pow_zero_to_negative,                          /* 27     */
  pow_neg_to_non_integer,                        /* 28     */
  pow_nan_to_zero,                               /* 29     */
  powf_overflow, powf_underflow,                 /* 30, 31 */
  powf_zero_to_zero,                             /* 32     */
  powf_zero_to_negative,                         /* 33     */
  powf_neg_to_non_integer,                       /* 34     */
  powf_nan_to_zero,                              /* 35     */
  atan2l_zero,                                   /* 36     */
  atan2_zero,                                    /* 37     */
  atan2f_zero,                                   /* 38     */
  expm1l_overflow,                               /* 39     */
  expm1l_underflow,                              /* 40     */
  expm1_overflow,                                /* 41     */
  expm1_underflow,                               /* 42     */
  expm1f_overflow,                               /* 43     */
  expm1f_underflow,                              /* 44     */
  hypotl_overflow,                               /* 45     */
  hypot_overflow,                                /* 46     */
  hypotf_overflow,                               /* 47     */
  sqrtl_negative,                                /* 48     */
  sqrt_negative,                                 /* 49     */
  sqrtf_negative,                                /* 50     */
  scalbl_overflow, scalbl_underflow,             /* 51, 52  */
  scalb_overflow,  scalb_underflow,              /* 53, 54  */
  scalbf_overflow, scalbf_underflow,             /* 55, 56  */
  acosl_gt_one, acos_gt_one, acosf_gt_one,       /* 57, 58, 59 */
  asinl_gt_one, asin_gt_one, asinf_gt_one,       /* 60, 61, 62 */
  coshl_overflow, cosh_overflow, coshf_overflow, /* 63, 64, 65 */
  y0l_zero, y0l_negative,y0l_gt_loss,            /* 66, 67, 68 */
  y0_zero, y0_negative,y0_gt_loss,               /* 69, 70, 71 */
  y0f_zero, y0f_negative,y0f_gt_loss,            /* 72, 73, 74 */
  y1l_zero, y1l_negative,y1l_gt_loss,            /* 75, 76, 77 */
  y1_zero, y1_negative,y1_gt_loss,               /* 78, 79, 80 */
  y1f_zero, y1f_negative,y1f_gt_loss,            /* 81, 82, 83 */
  ynl_zero, ynl_negative,ynl_gt_loss,            /* 84, 85, 86 */
  yn_zero, yn_negative,yn_gt_loss,               /* 87, 88, 89 */
  ynf_zero, ynf_negative,ynf_gt_loss,            /* 90, 91, 92 */
  j0l_gt_loss,                                   /* 93 */
  j0_gt_loss,                                    /* 94 */
  j0f_gt_loss,                                   /* 95 */
  j1l_gt_loss,                                   /* 96 */
  j1_gt_loss,                                    /* 97 */
  j1f_gt_loss,                                   /* 98 */
  jnl_gt_loss,                                   /* 99 */
  jn_gt_loss,                                    /* 100 */
  jnf_gt_loss,                                   /* 101 */
  lgammal_overflow, lgammal_negative,lgammal_reserve, /* 102, 103, 104 */
  lgamma_overflow, lgamma_negative,lgamma_reserve,    /* 105, 106, 107 */
  lgammaf_overflow, lgammaf_negative, lgammaf_reserve,/* 108, 109, 110 */
  gammal_overflow,gammal_negative, gammal_reserve,    /* 111, 112, 113 */
  gamma_overflow, gamma_negative, gamma_reserve,      /* 114, 115, 116 */
  gammaf_overflow,gammaf_negative,gammaf_reserve,     /* 117, 118, 119 */
  fmodl_by_zero,                                 /* 120 */
  fmod_by_zero,                                  /* 121 */
  fmodf_by_zero,                                 /* 122 */
  remainderl_by_zero,                            /* 123 */
  remainder_by_zero,                             /* 124 */
  remainderf_by_zero,                            /* 125 */
  sinhl_overflow, sinh_overflow, sinhf_overflow, /* 126, 127, 128 */
  atanhl_gt_one, atanhl_eq_one,                  /* 129, 130 */
  atanh_gt_one, atanh_eq_one,                    /* 131, 132 */
  atanhf_gt_one, atanhf_eq_one,                  /* 133, 134 */
  acoshl_lt_one,                                 /* 135 */
  acosh_lt_one,                                  /* 136 */
  acoshf_lt_one,                                 /* 137 */
  log1pl_zero,   log1pl_negative,                /* 138, 139 */
  log1p_zero,    log1p_negative,                 /* 140, 141 */
  log1pf_zero,   log1pf_negative,                /* 142, 143 */
  ldexpl_overflow,   ldexpl_underflow,           /* 144, 145 */
  ldexp_overflow,    ldexp_underflow,            /* 146, 147 */
  ldexpf_overflow,   ldexpf_underflow,           /* 148, 149 */
  logbl_zero,   logb_zero, logbf_zero,           /* 150, 151, 152 */
  nextafterl_overflow,   nextafter_overflow,
  nextafterf_overflow,                           /* 153, 154, 155 */
  ilogbl_zero,  ilogb_zero, ilogbf_zero,         /* 156, 157, 158 */
  exp2l_overflow, exp2l_underflow,               /* 159, 160 */
  exp2_overflow,  exp2_underflow,                /* 161, 162 */
  exp2f_overflow, exp2f_underflow,               /* 163, 164 */
  exp10l_overflow, exp10_overflow,
  exp10f_overflow,                               /* 165, 166, 167 */
  log2l_zero,    log2l_negative,                 /* 168, 169 */
  log2_zero,     log2_negative,                  /* 170, 171 */
  log2f_zero,    log2f_negative,                 /* 172, 173 */
  scalbnl_overflow, scalbnl_underflow,           /* 174, 175 */
  scalbn_overflow,  scalbn_underflow,            /* 176, 177 */
  scalbnf_overflow, scalbnf_underflow,           /* 178, 179 */
  remquol_by_zero,                               /* 180 */
  remquo_by_zero,                                /* 181 */
  remquof_by_zero,                               /* 182 */
  lrintl_large, lrint_large, lrintf_large,       /* 183, 184, 185 */
  llrintl_large, llrint_large, llrintf_large,    /* 186, 187, 188 */
  lroundl_large, lround_large, lroundf_large,    /* 189, 190, 191 */
  llroundl_large, llround_large, llroundf_large, /* 192, 193, 194 */
  fdiml_overflow, fdim_overflow, fdimf_overflow, /* 195, 196, 197 */
  nexttowardl_overflow,   nexttoward_overflow,
  nexttowardf_overflow,                          /* 198, 199, 200 */
  scalblnl_overflow, scalblnl_underflow,         /* 201, 202 */
  scalbln_overflow,  scalbln_underflow,          /* 203, 204 */
  scalblnf_overflow, scalblnf_underflow,         /* 205, 206 */
  erfcl_underflow, erfc_underflow, erfcf_underflow, /* 207, 208, 209 */
  acosdl_gt_one, acosd_gt_one, acosdf_gt_one,    /* 210, 211, 212 */
  asindl_gt_one, asind_gt_one, asindf_gt_one,    /* 213, 214, 215 */
  atan2dl_zero, atan2d_zero, atan2df_zero,       /* 216, 217, 218 */
  tandl_overflow, tand_overflow, tandf_overflow, /* 219, 220, 221 */
  cotdl_overflow, cotd_overflow, cotdf_overflow, /* 222, 223, 224 */
  cotl_overflow, cot_overflow, cotf_overflow,    /* 225, 226, 227 */
  sinhcoshl_overflow, sinhcosh_overflow, sinhcoshf_overflow, /* 228, 229, 230 */
  annuityl_by_zero, annuity_by_zero, annuityf_by_zero, /* 231, 232, 233 */
  annuityl_less_m1, annuity_less_m1, annuityf_less_m1, /* 234, 235, 236 */
  annuityl_overflow, annuity_overflow, annuityf_overflow, /* 237, 238, 239 */
  annuityl_underflow, annuity_underflow, annuityf_underflow, /* 240, 241, 242 */
  compoundl_by_zero, compound_by_zero, compoundf_by_zero, /* 243, 244, 245 */
  compoundl_less_m1, compound_less_m1, compoundf_less_m1, /* 246, 247, 248 */
  compoundl_overflow, compound_overflow, compoundf_overflow, /* 249, 250, 251 */
  compoundl_underflow, compound_underflow, compoundf_underflow, /* 252, 253, 254 */
  tgammal_overflow, tgammal_negative, tgammal_reserve, /* 255, 256, 257 */
  tgamma_overflow, tgamma_negative, tgamma_reserve, /* 258, 259, 260 */
  tgammaf_overflow, tgammaf_negative, tgammaf_reserve, /* 261, 262, 263 */
  exp10l_underflow, exp10_underflow, exp10f_underflow, /* 264, 265, 266 */
  nextafterl_underflow, nextafter_underflow,
  nextafterf_underflow,                                /* 267, 268, 269 */
  nexttowardl_underflow, nexttoward_underflow,
  nexttowardf_underflow                                /* 270, 271, 272 */
} error_types;

#define LIBM_ERROR __libm_error_support

extern void LIBM_ERROR(void*,void*,void*,error_types);
#ifdef _LIBC
libc_hidden_proto(LIBM_ERROR)
#endif

#define LIBM_ERROR1(x,r,e)	LIBM_ERROR(&(x), (void *)0, &(r), e)
#define LIBM_ERROR2(x,y,r,e)	LIBM_ERROR(&(x), &(y), &(r), e)

#endif // !defined(__LIBM_ERROR_CODES_H__)
