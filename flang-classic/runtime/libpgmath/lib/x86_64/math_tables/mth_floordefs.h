/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

MTHINTRIN(floor  , ss   , em64t      ,  __mth_i_floor         ,  __mth_i_floor         , __mth_i_floor          ,__math_dispatch_error)
MTHINTRIN(floor  , ds   , em64t      ,  __mth_i_dfloor        ,  __mth_i_dfloor        , __mth_i_dfloor         ,__math_dispatch_error)
MTHINTRIN(floor  , ss   , sse4       ,  __mth_i_floor_sse     ,  __mth_i_floor_sse     , __mth_i_floor_sse      ,__math_dispatch_error)
MTHINTRIN(floor  , ds   , sse4       ,  __mth_i_dfloor_sse    ,  __mth_i_dfloor_sse    , __mth_i_dfloor_sse     ,__math_dispatch_error)
MTHINTRIN(floor  , ss   , avx        ,  __mth_i_floor_avx     ,  __mth_i_floor_avx     , __mth_i_floor_avx      ,__math_dispatch_error)
MTHINTRIN(floor  , ds   , avx        ,  __mth_i_dfloor_avx    ,  __mth_i_dfloor_avx    , __mth_i_dfloor_avx     ,__math_dispatch_error)
MTHINTRIN(floor  , ss   , avxfma4    ,  __mth_i_floor_avx     ,  __mth_i_floor_avx     , __mth_i_floor_avx      ,__math_dispatch_error)
MTHINTRIN(floor  , ds   , avxfma4    ,  __mth_i_dfloor_avx    ,  __mth_i_dfloor_avx    , __mth_i_dfloor_avx     ,__math_dispatch_error)
MTHINTRIN(floor  , ss   , avx2       ,  __mth_i_floor_avx     ,  __mth_i_floor_avx     , __mth_i_floor_avx      ,__math_dispatch_error)
MTHINTRIN(floor  , ds   , avx2       ,  __mth_i_dfloor_avx    ,  __mth_i_dfloor_avx    , __mth_i_dfloor_avx     ,__math_dispatch_error)
MTHINTRIN(floor  , ss   , avx512knl  ,  __mth_i_floor_avx     ,  __mth_i_floor_avx     , __mth_i_floor_avx      ,__math_dispatch_error)
MTHINTRIN(floor  , ds   , avx512knl  ,  __mth_i_dfloor_avx    ,  __mth_i_dfloor_avx    , __mth_i_dfloor_avx     ,__math_dispatch_error)
MTHINTRIN(floor  , ss   , avx512     ,  __mth_i_floor_avx     ,  __mth_i_floor_avx     , __mth_i_floor_avx      ,__math_dispatch_error)
MTHINTRIN(floor  , ds   , avx512     ,  __mth_i_dfloor_avx    ,  __mth_i_dfloor_avx    , __mth_i_dfloor_avx     ,__math_dispatch_error)
MTHINTRIN(floor  , sv4  , any        ,  __gs_floor_4_f        ,  __gs_floor_4_r        , __gs_floor_4_p         ,__math_dispatch_error)
MTHINTRIN(floor  , dv2  , any        ,  __gd_floor_2_f        ,  __gd_floor_2_r        , __gd_floor_2_p         ,__math_dispatch_error)
MTHINTRIN(floor  , sv8  , any        ,  __gs_floor_8_f        ,  __gs_floor_8_r        , __gs_floor_8_p         ,__math_dispatch_error)
MTHINTRIN(floor  , dv4  , any        ,  __gd_floor_4_f        ,  __gd_floor_4_r        , __gd_floor_4_p         ,__math_dispatch_error)
MTHINTRIN(floor  , sv16 , any        ,  __gs_floor_16_f       ,  __gs_floor_16_r       , __gs_floor_16_p        ,__math_dispatch_error)
MTHINTRIN(floor  , dv8  , any        ,  __gd_floor_8_f        ,  __gd_floor_8_r        , __gd_floor_8_p         ,__math_dispatch_error)
MTHINTRIN(floor  , sv4m , any        , __fs_floor_4_mn        , __rs_floor_4_mn        , __ps_floor_4_mn        ,__math_dispatch_error)
MTHINTRIN(floor  , dv2m , any        , __fd_floor_2_mn        , __rd_floor_2_mn        , __pd_floor_2_mn        ,__math_dispatch_error)
MTHINTRIN(floor  , sv8m , any        , __fs_floor_8_mn        , __rs_floor_8_mn        , __ps_floor_8_mn        ,__math_dispatch_error)
MTHINTRIN(floor  , dv4m , any        , __fd_floor_4_mn        , __rd_floor_4_mn        , __pd_floor_4_mn        ,__math_dispatch_error)
MTHINTRIN(floor  , sv16m, any        , __fs_floor_16_mn       , __rs_floor_16_mn       , __ps_floor_16_mn       ,__math_dispatch_error)
MTHINTRIN(floor  , dv8m , any        , __fd_floor_8_mn        , __rd_floor_8_mn        , __pd_floor_8_mn        ,__math_dispatch_error)
