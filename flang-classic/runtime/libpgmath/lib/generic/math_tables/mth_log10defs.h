/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/******************************************************************************
 *                                                                            *
 * Background:                                                                *
 * The POWERPC ABI does not provide for tail calls. Thus, the math dispatch   *
 * table processing incurs overhead with the saving and restoration of GPR 2  *
 * that can severely affect application performance.  For POWERPC, we use an  *
 * optimized assembly dispatch set of routines that make tail calls to all of *
 * the routines defined in the math dispatch configuration files but do not   *
 * saveand /restore GPR 2.                                                    *
 *                                                                            *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! *
 *                                                                            *
 * If any entry (routine <FUNC>) in any of the dispatch tables is not present *
 * in i.e. not  satisfied by, libpgmath, in order to properly preserve/restore*
 * GRP 2 when calling routine <FUNC>, the actual function must first be       *
 * encapsulated in a routine present in libpgmath.                            *
 *                                                                            *
 * No doubt there are pathological cases that will show this engineering      *
 * choice to be wrong, but current performance testing shows otherwise.       *
 *                                                                            *
 *****************************************************************************/

MTHINTRIN(log10, ss   , any        , __mth_i_log10         , __mth_i_log10         , __mth_i_log10         ,__math_dispatch_error)
MTHINTRIN(log10, ds   , any        , __mth_i_dlog10        , __mth_i_dlog10        , __mth_i_dlog10        ,__math_dispatch_error)
#ifdef TARGET_SUPPORTS_QUADFP
MTHINTRIN(log10, qs   , any        , __mth_i_qlog10        , __mth_i_qlog10        , __mth_i_qlog10        ,__math_dispatch_error)
#endif
MTHINTRIN(log10, sv4  , any        , __gs_log10_4_f        , __gs_log10_4_r        , __gs_log10_4_p        ,__math_dispatch_error)
MTHINTRIN(log10, dv2  , any        , __gd_log10_2_f        , __gd_log10_2_r        , __gd_log10_2_p        ,__math_dispatch_error)
MTHINTRIN(log10, sv4m , any        , __fs_log10_4_mn       , __rs_log10_4_mn       , __ps_log10_4_mn       ,__math_dispatch_error)
MTHINTRIN(log10, dv2m , any        , __fd_log10_2_mn       , __rd_log10_2_mn       , __pd_log10_2_mn       ,__math_dispatch_error)
