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

MTHINTRIN(floor  , ss   , any        ,  __mth_i_floor        ,  __mth_i_floor        , __mth_i_floor         ,__math_dispatch_error)
MTHINTRIN(floor  , ds   , any        ,  __mth_i_dfloor       ,  __mth_i_dfloor       , __mth_i_dfloor        ,__math_dispatch_error)
#ifdef TARGET_SUPPORTS_QUADFP
MTHINTRIN(floor  , qs   , any        ,  __mth_i_qfloor       ,  __mth_i_qfloor       , __mth_i_qfloor        ,__math_dispatch_error)
#endif
MTHINTRIN(floor  , sv4  , any        ,  __gs_floor_4_f       ,  __gs_floor_4_r       , __gs_floor_4_p        ,__math_dispatch_error)
MTHINTRIN(floor  , dv2  , any        ,  __gd_floor_2_f       ,  __gd_floor_2_r       , __gd_floor_2_p        ,__math_dispatch_error)
MTHINTRIN(floor  , sv4m , any        , __fs_floor_4_mn       , __rs_floor_4_mn       , __ps_floor_4_mn       ,__math_dispatch_error)
MTHINTRIN(floor  , dv2m , any        , __fd_floor_2_mn       , __rd_floor_2_mn       , __pd_floor_2_mn       ,__math_dispatch_error)
