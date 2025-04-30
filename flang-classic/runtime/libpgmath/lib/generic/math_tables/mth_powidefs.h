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

/* R(:)**I4 */
MTHINTRIN(powi1, ss   , any        , __mth_i_rpowi         , __mth_i_rpowi         , __pmth_i_rpowi        ,__math_dispatch_error)
MTHINTRIN(powi1, ds   , any        , __mth_i_dpowi         , __mth_i_dpowi         , __pmth_i_dpowi        ,__math_dispatch_error)
MTHINTRIN(powi1, sv4  , any        , __fx_powi1_4          , __fx_powi1_4          , __px_powi1_4          ,__math_dispatch_error)
MTHINTRIN(powi1, dv2  , any        , __fx_powi1_2          , __fx_powi1_2          , __px_powi1_2          ,__math_dispatch_error)
MTHINTRIN(powi1, sv4m , any        , __fs_powi1_4_mn       , __rs_powi1_4_mn       , __ps_powi1_4_mn       ,__math_dispatch_error)
MTHINTRIN(powi1, dv2m , any        , __fd_powi1_2_mn       , __rd_powi1_2_mn       , __pd_powi1_2_mn       ,__math_dispatch_error)
/* R(:)**I4(:) */
MTHINTRIN(powi , ss   , any        , __mth_i_rpowi         , __mth_i_rpowi         , __pmth_i_rpowi        ,__math_dispatch_error)
MTHINTRIN(powi , ds   , any        , __mth_i_dpowi         , __mth_i_dpowi         , __pmth_i_dpowi        ,__math_dispatch_error)
#ifdef TARGET_SUPPORTS_QUADFP
MTHINTRIN(powi , qs   , any        , __mth_i_qpowi         , __mth_i_qpowi         , __pmth_i_qpowi        ,__math_dispatch_error)
#endif
MTHINTRIN(powi , sv4  , any        , __gs_powi_4_f         , __gs_powi_4_r         , __px_powi_4           ,__math_dispatch_error)
MTHINTRIN(powi , dv2  , any        , __gd_powi_2_f         , __gd_powi_2_r         , __px_powi_2           ,__math_dispatch_error)
MTHINTRIN(powi , sv4m , any        , __fs_powi_4_mn        , __rs_powi_4_mn        , __ps_powi_4_mn        ,__math_dispatch_error)
MTHINTRIN(powi , dv2m , any        , __fd_powi_2_mn        , __rd_powi_2_mn        , __pd_powi_2_mn        ,__math_dispatch_error)
/* R(:)**I8 */
MTHINTRIN(powk1, ss   , any        , __mth_i_rpowk         , __mth_i_rpowk         , __pmth_i_rpowk        ,__math_dispatch_error)
MTHINTRIN(powk1, ds   , any        , __mth_i_dpowk         , __mth_i_dpowk         , __pmth_i_dpowk        ,__math_dispatch_error)
MTHINTRIN(powk1, sv4  , any        , __fx_powk1_4          , __fx_powk1_4          , __px_powk1_4          ,__math_dispatch_error)
MTHINTRIN(powk1, dv2  , any        , __fx_powk1_2          , __fx_powk1_2          , __px_powk1_2          ,__math_dispatch_error)
MTHINTRIN(powk1, sv4m , any        , __fs_powk1_4_mn       , __rs_powk1_4_mn       , __ps_powk1_4_mn       ,__math_dispatch_error)
MTHINTRIN(powk1, dv2m , any        , __fd_powk1_2_mn       , __rd_powk1_2_mn       , __pd_powk1_2_mn       ,__math_dispatch_error)
/* R(:)**I8(:) */
MTHINTRIN(powk , ss   , any        , __mth_i_rpowk         , __mth_i_rpowk         , __pmth_i_rpowk        ,__math_dispatch_error)
MTHINTRIN(powk , ds   , any        , __mth_i_dpowk         , __mth_i_dpowk         , __pmth_i_dpowk        ,__math_dispatch_error)
#ifdef TARGET_SUPPORTS_QUADFP
MTHINTRIN(powk , qs   , any        , __mth_i_qpowk         , __mth_i_qpowk         , __pmth_i_qpowk        ,__math_dispatch_error)
#endif
MTHINTRIN(powk , sv4  , any        , __gs_powk_4_f         , __gs_powk_4_r         , __px_powk_4           ,__math_dispatch_error)
MTHINTRIN(powk , dv2  , any        , __gd_powk_2_f         , __gd_powk_2_r         , __px_powk_2           ,__math_dispatch_error)
MTHINTRIN(powk , sv4m , any        , __fs_powk_4_mn        , __rs_powk_4_mn        , __ps_powk_4_mn        ,__math_dispatch_error)
MTHINTRIN(powk , dv2m , any        , __fd_powk_2_mn        , __rd_powk_2_mn        , __pd_powk_2_mn        ,__math_dispatch_error)

