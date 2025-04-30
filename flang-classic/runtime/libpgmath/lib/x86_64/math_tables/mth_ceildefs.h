/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

MTHINTRIN(ceil  , ss   , em64t      ,  __mth_i_ceil         , __mth_i_ceil          , __mth_i_ceil          ,__math_dispatch_error)
MTHINTRIN(ceil  , ss   , sse4       ,  __mth_i_ceil_sse     , __mth_i_ceil_sse      , __mth_i_ceil_sse      ,__math_dispatch_error)
MTHINTRIN(ceil  , ss   , avx        ,  __mth_i_ceil_avx     , __mth_i_ceil_avx      , __mth_i_ceil_avx      ,__math_dispatch_error)
MTHINTRIN(ceil  , ss   , avxfma4    ,  __mth_i_ceil_avx     , __mth_i_ceil_avx      , __mth_i_ceil_avx      ,__math_dispatch_error)
MTHINTRIN(ceil  , ss   , avx2       ,  __mth_i_ceil_avx     , __mth_i_ceil_avx      , __mth_i_ceil_avx      ,__math_dispatch_error)
MTHINTRIN(ceil  , ss   , avx512knl  ,  __mth_i_ceil_avx     , __mth_i_ceil_avx      , __mth_i_ceil_avx      ,__math_dispatch_error)
MTHINTRIN(ceil  , ss   , avx512     ,  __mth_i_ceil_avx     , __mth_i_ceil_avx      , __mth_i_ceil_avx      ,__math_dispatch_error)
MTHINTRIN(ceil  , ds   , em64t      ,  __mth_i_dceil        , __mth_i_dceil         , __mth_i_dceil         ,__math_dispatch_error)
MTHINTRIN(ceil  , ds   , sse4       ,  __mth_i_dceil_sse    , __mth_i_dceil_sse     , __mth_i_dceil_sse     ,__math_dispatch_error)
MTHINTRIN(ceil  , ds   , avx        ,  __mth_i_dceil_avx    , __mth_i_dceil_avx     , __mth_i_dceil_avx     ,__math_dispatch_error)
MTHINTRIN(ceil  , ds   , avxfma4    ,  __mth_i_dceil_avx    , __mth_i_dceil_avx     , __mth_i_dceil_avx     ,__math_dispatch_error)
MTHINTRIN(ceil  , ds   , avx2       ,  __mth_i_dceil_avx    , __mth_i_dceil_avx     , __mth_i_dceil_avx     ,__math_dispatch_error)
MTHINTRIN(ceil  , ds   , avx512knl  ,  __mth_i_dceil_avx    , __mth_i_dceil_avx     , __mth_i_dceil_avx     ,__math_dispatch_error)
MTHINTRIN(ceil  , ds   , avx512     ,  __mth_i_dceil_avx    , __mth_i_dceil_avx     , __mth_i_dceil_avx     ,__math_dispatch_error)
MTHINTRIN(ceil  , sv4  , any        ,  __gs_ceil_4_f        ,  __gs_ceil_4_r        , __gs_ceil_4_p         ,__math_dispatch_error)
MTHINTRIN(ceil  , dv2  , any        ,  __gd_ceil_2_f        ,  __gd_ceil_2_r        , __gd_ceil_2_p         ,__math_dispatch_error)
MTHINTRIN(ceil  , sv8  , any        ,  __gs_ceil_8_f        ,  __gs_ceil_8_r        , __gs_ceil_8_p         ,__math_dispatch_error)
MTHINTRIN(ceil  , dv4  , any        ,  __gd_ceil_4_f        ,  __gd_ceil_4_r        , __gd_ceil_4_p         ,__math_dispatch_error)
MTHINTRIN(ceil  , sv16 , any        ,  __gs_ceil_16_f       ,  __gs_ceil_16_r       , __gs_ceil_16_p        ,__math_dispatch_error)
MTHINTRIN(ceil  , dv8  , any        ,  __gd_ceil_8_f        ,  __gd_ceil_8_r        , __gd_ceil_8_p         ,__math_dispatch_error)
MTHINTRIN(ceil  , sv4m , any        , __fs_ceil_4_mn        , __rs_ceil_4_mn        , __ps_ceil_4_mn        ,__math_dispatch_error)
MTHINTRIN(ceil  , dv2m , any        , __fd_ceil_2_mn        , __rd_ceil_2_mn        , __pd_ceil_2_mn        ,__math_dispatch_error)
MTHINTRIN(ceil  , sv8m , any        , __fs_ceil_8_mn        , __rs_ceil_8_mn        , __ps_ceil_8_mn        ,__math_dispatch_error)
MTHINTRIN(ceil  , dv4m , any        , __fd_ceil_4_mn        , __rd_ceil_4_mn        , __pd_ceil_4_mn        ,__math_dispatch_error)
MTHINTRIN(ceil  , sv16m, any        , __fs_ceil_16_mn       , __rs_ceil_16_mn       , __ps_ceil_16_mn       ,__math_dispatch_error)
MTHINTRIN(ceil  , dv8m , any        , __fd_ceil_8_mn        , __rd_ceil_8_mn        , __pd_ceil_8_mn        ,__math_dispatch_error)
