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

MTHINTRIN(mod   , ss   , any        , __mth_i_amod           , __mth_i_amod           , __mth_i_amod           ,__math_dispatch_error)
MTHINTRIN(mod   , ds   , any        , __mth_i_dmod           , __mth_i_dmod           , __mth_i_dmod           ,__math_dispatch_error)
#ifdef TARGET_SUPPORTS_QUADFP
MTHINTRIN(mod   , qs   , any        , __mth_i_qmod           , __mth_i_qmod           , __mth_i_qmod           ,__math_dispatch_error)
#endif
MTHINTRIN(mod   , sv4  , any        , __gs_mod_4_f           , __gs_mod_4_r           , __gs_mod_4_p           ,__math_dispatch_error)
MTHINTRIN(mod   , dv2  , any        , __gd_mod_2_f           , __gd_mod_2_r           , __gd_mod_2_p           ,__math_dispatch_error)
MTHINTRIN(mod   , sv4m , any        , __fs_mod_4_mn          , __rs_mod_4_mn          , __ps_mod_4_mn          ,__math_dispatch_error)
MTHINTRIN(mod   , dv2m , any        , __fd_mod_2_mn          , __rd_mod_2_mn          , __pd_mod_2_mn          ,__math_dispatch_error)
