/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

MTHINTRIN(div  , sv4m , any        , __fs_div_4_mn         , __rs_div_4_mn         , __ps_div_4_mn         ,__math_dispatch_error)
MTHINTRIN(div  , dv2m , any        , __fd_div_2_mn         , __rd_div_2_mn         , __pd_div_2_mn         ,__math_dispatch_error)
MTHINTRIN(div  , sv8m , avx        , __fs_div_8_mn         , __rs_div_8_mn         , __ps_div_8_mn         ,__math_dispatch_error)
MTHINTRIN(div  , dv4m , avx        , __fd_div_4_mn         , __rd_div_4_mn         , __pd_div_4_mn         ,__math_dispatch_error)
MTHINTRIN(div  , sv8m , avxfma4    , __fs_div_8_mn         , __rs_div_8_mn         , __ps_div_8_mn         ,__math_dispatch_error)
MTHINTRIN(div  , dv4m , avxfma4    , __fd_div_4_mn         , __rd_div_4_mn         , __pd_div_4_mn         ,__math_dispatch_error)
MTHINTRIN(div  , sv8m , avx2       , __fs_div_8_mn         , __rs_div_8_mn         , __ps_div_8_mn         ,__math_dispatch_error)
MTHINTRIN(div  , dv4m , avx2       , __fd_div_4_mn         , __rd_div_4_mn         , __pd_div_4_mn         ,__math_dispatch_error)
MTHINTRIN(div  , sv8m , avx512knl  , __fs_div_8_mn         , __rs_div_8_mn         , __ps_div_8_mn         ,__math_dispatch_error)
MTHINTRIN(div  , dv4m , avx512knl  , __fd_div_4_mn         , __rd_div_4_mn         , __pd_div_4_mn         ,__math_dispatch_error)
MTHINTRIN(div  , sv16m, avx512knl  , __fs_div_16_mn        , __rs_div_16_mn        , __ps_div_16_mn        ,__math_dispatch_error)
MTHINTRIN(div  , dv8m , avx512knl  , __fd_div_8_mn         , __rd_div_8_mn         , __pd_div_8_mn         ,__math_dispatch_error)
MTHINTRIN(div  , sv8m , avx512     , __fs_div_8_mn         , __rs_div_8_mn         , __ps_div_8_mn         ,__math_dispatch_error)
MTHINTRIN(div  , dv4m , avx512     , __fd_div_4_mn         , __rd_div_4_mn         , __pd_div_4_mn         ,__math_dispatch_error)
MTHINTRIN(div  , sv16m, avx512     , __fs_div_16_mn        , __rs_div_16_mn        , __ps_div_16_mn        ,__math_dispatch_error)
MTHINTRIN(div  , dv8m , avx512     , __fd_div_8_mn         , __rd_div_8_mn         , __pd_div_8_mn         ,__math_dispatch_error)

MTHINTRIN(div  , cs   , em64t      , __mth_i_cdiv_c99      , __mth_i_cdiv_c99      , __mth_i_cdiv_c99      ,__math_dispatch_error)
MTHINTRIN(div  , zs   , em64t      , __mth_i_cddiv_c99     , __mth_i_cddiv_c99     , __mth_i_cddiv_c99     ,__math_dispatch_error)
MTHINTRIN(div  , zv1  , em64t      , __gz_div_1v_f         , __gz_div_1v_r         , __gz_div_1v_p         ,__math_dispatch_error)
MTHINTRIN(div  , cv2  , em64t      , __gc_div_2_f          , __gc_div_2_r          , __gc_div_2_p          ,__math_dispatch_error)

MTHINTRIN(div  , cs   , sse4       , __fsc_div             , __fsc_div             , __mth_i_cdiv_c99      ,__math_dispatch_error)
MTHINTRIN(div  , zs   , sse4       , __fsz_div_c99         , __fsz_div_c99         , __mth_i_cddiv_c99     ,__math_dispatch_error)
MTHINTRIN(div  , zv1  , sse4       , __fsz_div             , __fsz_div             , __gz_div_1v_p         ,__math_dispatch_error)
MTHINTRIN(div  , cv2  , sse4       , __fvc_div             , __fvc_div             , __gc_div_2_p          ,__math_dispatch_error)

MTHINTRIN(div  , cs   , avx        , __fsc_div_vex         , __fsc_div_vex         , __mth_i_cdiv_c99      ,__math_dispatch_error)
MTHINTRIN(div  , zs   , avx        , __fsz_div_vex_c99     , __fsz_div_vex_c99     , __mth_i_cddiv_c99     ,__math_dispatch_error)
MTHINTRIN(div  , zv1  , avx        , __fsz_div_vex         , __fsz_div_vex         , __gz_div_1v_p         ,__math_dispatch_error)
MTHINTRIN(div  , cv2  , avx        , __fvc_div_vex         , __fvc_div_vex         , __gc_div_2_p          ,__math_dispatch_error)
MTHINTRIN(div  , cv4  , avx        , __fvc_div_vex_256     , __fvc_div_vex_256     , __gc_div_4_p          ,__math_dispatch_error)
MTHINTRIN(div  , zv2  , avx        , __fvz_div_vex_256     , __fvz_div_vex_256     , __gz_div_2_p          ,__math_dispatch_error)

MTHINTRIN(div  , cs   , avxfma4    , __fsc_div_fma4        , __fsc_div_fma4        , __mth_i_cdiv_c99      ,__math_dispatch_error)
MTHINTRIN(div  , zs   , avxfma4    , __fsz_div_fma4_c99    , __fsz_div_fma4_c99    , __mth_i_cddiv_c99     ,__math_dispatch_error)
MTHINTRIN(div  , zv1  , avxfma4    , __fsz_div_fma4        , __fsz_div_fma4        , __gz_div_1v_p         ,__math_dispatch_error)
MTHINTRIN(div  , cv2  , avxfma4    , __fvc_div_fma4        , __fvc_div_fma4        , __gc_div_2_p          ,__math_dispatch_error)
MTHINTRIN(div  , cv4  , avxfma4    , __fvc_div_fma4_256    , __fvc_div_fma4_256    , __gc_div_4_p          ,__math_dispatch_error)
MTHINTRIN(div  , zv2  , avxfma4    , __fvz_div_fma4_256    , __fvz_div_fma4_256    , __gz_div_2_p          ,__math_dispatch_error)

MTHINTRIN(div  , cs   , avx2       , __fsc_div_avx2        , __fsc_div_avx2        , __mth_i_cdiv_c99      ,__math_dispatch_error)
MTHINTRIN(div  , zs   , avx2       , __fsz_div_avx2_c99    , __fsz_div_avx2_c99    , __mth_i_cddiv_c99     ,__math_dispatch_error)
MTHINTRIN(div  , zv1  , avx2       , __fsz_div_avx2        , __fsz_div_avx2        , __gz_div_1v_p         ,__math_dispatch_error)
MTHINTRIN(div  , cv2  , avx2       , __fvc_div_avx2        , __fvc_div_avx2        , __gc_div_2_p          ,__math_dispatch_error)
MTHINTRIN(div  , cv4  , avx2       , __fvc_div_avx2_256    , __fvc_div_avx2_256    , __gc_div_4_p          ,__math_dispatch_error)
MTHINTRIN(div  , zv2  , avx2       , __fvz_div_avx2_256    , __fvz_div_avx2_256    , __gz_div_2_p          ,__math_dispatch_error)

MTHINTRIN(div  , cs   , avx512knl  , __fsc_div_avx2        , __fsc_div_avx2        , __mth_i_cdiv_c99      ,__math_dispatch_error)
MTHINTRIN(div  , zs   , avx512knl  , __fsz_div_avx2_c99    , __fsz_div_avx2_c99    , __mth_i_cddiv_c99     ,__math_dispatch_error)
MTHINTRIN(div  , zv1  , avx512knl  , __fsz_div_avx2        , __fsz_div_avx2        , __gz_div_1v_p         ,__math_dispatch_error)
MTHINTRIN(div  , cv2  , avx512knl  , __fvc_div_avx2        , __fvc_div_avx2        , __gc_div_2_p          ,__math_dispatch_error)
MTHINTRIN(div  , cv4  , avx512knl  , __fvc_div_avx2_256    , __fvc_div_avx2_256    , __gc_div_4_p          ,__math_dispatch_error)
MTHINTRIN(div  , cv8  , avx512knl  , __fvc_div_evex_512    , __fvc_div_evex_512    , __gc_div_8_p          ,__math_dispatch_error)
MTHINTRIN(div  , zv2  , avx512knl  , __fvz_div_avx2_256    , __fvz_div_avx2_256    , __gz_div_2_p          ,__math_dispatch_error)
MTHINTRIN(div  , zv4  , avx512knl  , __fvz_div_evex_512    , __fvz_div_evex_512    , __gz_div_4_p          ,__math_dispatch_error)

MTHINTRIN(div  , cs   , avx512     , __fsc_div_avx2        , __fsc_div_avx2        , __mth_i_cdiv_c99      ,__math_dispatch_error)
MTHINTRIN(div  , zs   , avx512     , __fsz_div_avx2_c99    , __fsz_div_avx2_c99    , __mth_i_cddiv_c99     ,__math_dispatch_error)
MTHINTRIN(div  , zv1  , avx512     , __fsz_div_avx2        , __fsz_div_avx2        , __gz_div_1v_p         ,__math_dispatch_error)
MTHINTRIN(div  , cv2  , avx512     , __fvc_div_avx2        , __fvc_div_avx2        , __gc_div_2_p          ,__math_dispatch_error)
MTHINTRIN(div  , cv4  , avx512     , __fvc_div_avx2_256    , __fvc_div_avx2_256    , __gc_div_4_p          ,__math_dispatch_error)
MTHINTRIN(div  , cv8  , avx512     , __fvc_div_evex_512    , __fvc_div_evex_512    , __gc_div_8_p          ,__math_dispatch_error)
MTHINTRIN(div  , zv2  , avx512     , __fvz_div_avx2_256    , __fvz_div_avx2_256    , __gz_div_2_p          ,__math_dispatch_error)
MTHINTRIN(div  , zv4  , avx512     , __fvz_div_evex_512    , __fvz_div_evex_512    , __gz_div_4_p          ,__math_dispatch_error)
