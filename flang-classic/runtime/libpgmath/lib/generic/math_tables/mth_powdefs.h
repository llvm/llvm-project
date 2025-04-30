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

MTHINTRIN(pow  , ss   , any        ,  __mth_i_rpowr        ,  __mth_i_rpowr        , __mth_i_rpowr         ,__math_dispatch_error)
MTHINTRIN(pow  , ds   , any        ,  __mth_i_dpowd        ,  __mth_i_dpowd        , __mth_i_dpowd         ,__math_dispatch_error)
#ifdef TARGET_SUPPORTS_QUADFP
MTHINTRIN(pow  , qs   , any        ,  __mth_i_qpowq        ,  __mth_i_qpowq        , __mth_i_qpowq         ,__math_dispatch_error)
#endif
MTHINTRIN(pow  , sv4  , any        ,  __gs_pow_4_f         ,  __gs_pow_4_r         , __gs_pow_4_p          ,__math_dispatch_error)
MTHINTRIN(pow  , dv2  , any        ,  __gd_pow_2_f         ,  __gd_pow_2_r         , __gd_pow_2_p          ,__math_dispatch_error)
MTHINTRIN(pow  , sv4m , any        , __fs_pow_4_mn         , __rs_pow_4_mn         , __ps_pow_4_mn         ,__math_dispatch_error)
MTHINTRIN(pow  , dv2m , any        , __fd_pow_2_mn         , __rd_pow_2_mn         , __pd_pow_2_mn         ,__math_dispatch_error)
