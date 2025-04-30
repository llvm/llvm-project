/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

MTHINTRIN(mod  , ss   , em64t      , __fmth_i_amod         , __fmth_i_amod         , __mth_i_amod          ,__math_dispatch_error)
MTHINTRIN(mod  , ds   , em64t      , __fmth_i_dmod         , __fmth_i_dmod         , __mth_i_dmod          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4  , em64t      , __fvsmod              , __fvsmod              , __gs_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv2  , em64t      , __fvdmod              , __fvdmod              , __gd_mod_2_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4m , em64t      , __fs_mod_4_mn         , __rs_mod_4_mn         , __ps_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv2m , em64t      , __fd_mod_2_mn         , __rd_mod_2_mn         , __pd_mod_2_mn         ,__math_dispatch_error)

MTHINTRIN(mod  , ss   , sse4       , __fss_mod             , __fss_mod             , __mth_i_amod          ,__math_dispatch_error)
MTHINTRIN(mod  , ds   , sse4       , __fsd_mod             , __fsd_mod             , __mth_i_dmod          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4  , sse4       , __fvs_mod             , __fvs_mod             , __gs_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv2  , sse4       , __fvd_mod             , __fvd_mod             , __gd_mod_2_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4m , sse4       , __fs_mod_4_mn         , __rs_mod_4_mn         , __ps_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv2m , sse4       , __fd_mod_2_mn         , __rd_mod_2_mn         , __pd_mod_2_mn         ,__math_dispatch_error)

MTHINTRIN(mod  , ss   , avx        , __fss_mod_vex         , __fss_mod_vex         , __mth_i_amod          ,__math_dispatch_error)
MTHINTRIN(mod  , ds   , avx        , __fsd_mod_vex         , __fsd_mod_vex         , __mth_i_dmod          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4  , avx        , __fvs_mod_vex         , __fvs_mod_vex         , __gs_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv2  , avx        , __fvd_mod_vex         , __fvd_mod_vex         , __gd_mod_2_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv8  , avx        , __fvs_mod_vex_256     , __fvs_mod_vex_256     , __gs_mod_8_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv4  , avx        , __fvd_mod_vex_256     , __fvd_mod_vex_256     , __gd_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4m , avx        , __fs_mod_4_mn         , __rs_mod_4_mn         , __ps_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv2m , avx        , __fd_mod_2_mn         , __rd_mod_2_mn         , __pd_mod_2_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , sv8m , avx        , __fs_mod_8_mn         , __rs_mod_8_mn         , __ps_mod_8_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv4m , avx        , __fd_mod_4_mn         , __rd_mod_4_mn         , __pd_mod_4_mn         ,__math_dispatch_error)


MTHINTRIN(mod  , ss   , avxfma4    , __fss_mod_fma4        , __fss_mod_fma4        , __mth_i_amod          ,__math_dispatch_error)
MTHINTRIN(mod  , ds   , avxfma4    , __fsd_mod_fma4        , __fsd_mod_fma4        , __mth_i_dmod          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4  , avxfma4    , __fvs_mod_fma4        , __fvs_mod_fma4        , __gs_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv2  , avxfma4    , __fvd_mod_fma4        , __fvd_mod_fma4        , __gd_mod_2_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv8  , avxfma4    , __fvs_mod_fma4_256    , __fvs_mod_fma4_256    , __gs_mod_8_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv4  , avxfma4    , __fvd_mod_fma4_256    , __fvd_mod_fma4_256    , __gd_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4m , avxfma4    , __fs_mod_4_mn         , __rs_mod_4_mn         , __ps_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv2m , avxfma4    , __fd_mod_2_mn         , __rd_mod_2_mn         , __pd_mod_2_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , sv8m , avxfma4    , __fs_mod_8_mn         , __rs_mod_8_mn         , __ps_mod_8_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv4m , avxfma4    , __fd_mod_4_mn         , __rd_mod_4_mn         , __pd_mod_4_mn         ,__math_dispatch_error)


MTHINTRIN(mod  , ss   , avx2       , __fss_mod_vex         , __fss_mod_vex         , __mth_i_amod          ,__math_dispatch_error)
MTHINTRIN(mod  , ds   , avx2       , __fsd_mod_vex         , __fsd_mod_vex         , __mth_i_dmod          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4  , avx2       , __fvs_mod_vex         , __fvs_mod_vex         , __gs_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv2  , avx2       , __fvd_mod_vex         , __fvd_mod_vex         , __gd_mod_2_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv8  , avx2       , __fvs_mod_vex_256     , __fvs_mod_vex_256     , __gs_mod_8_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv4  , avx2       , __fvd_mod_vex_256     , __fvd_mod_vex_256     , __gd_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4m , avx2       , __fs_mod_4_mn         , __rs_mod_4_mn         , __ps_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv2m , avx2       , __fd_mod_2_mn         , __rd_mod_2_mn         , __pd_mod_2_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , sv8m , avx2       , __fs_mod_8_mn         , __rs_mod_8_mn         , __ps_mod_8_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv4m , avx2       , __fd_mod_4_mn         , __rd_mod_4_mn         , __pd_mod_4_mn         ,__math_dispatch_error)

MTHINTRIN(mod  , ss   , avx512knl  , __fss_mod_vex         , __fss_mod_vex         , __mth_i_amod          ,__math_dispatch_error)
MTHINTRIN(mod  , ds   , avx512knl  , __fsd_mod_vex         , __fsd_mod_vex         , __mth_i_dmod          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4  , avx512knl  , __fvs_mod_vex         , __fvs_mod_vex         , __gs_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv2  , avx512knl  , __fvd_mod_vex         , __fvd_mod_vex         , __gd_mod_2_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv8  , avx512knl  , __fvs_mod_vex_256     , __fvs_mod_vex_256     , __gs_mod_8_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv4  , avx512knl  , __fvd_mod_vex_256     , __fvd_mod_vex_256     , __gd_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv16 , avx512knl  , __fs_mod_16_z2yy      , __rs_mod_16_z2yy      , __gs_mod_16_p         ,__math_dispatch_error)
MTHINTRIN(mod  , dv8  , avx512knl  , __fd_mod_8_z2yy       , __rd_mod_8_z2yy       , __gd_mod_8_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4m , avx512knl  , __fs_mod_4_mn         , __rs_mod_4_mn         , __ps_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv2m , avx512knl  , __fd_mod_2_mn         , __rd_mod_2_mn         , __pd_mod_2_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , sv8m , avx512knl  , __fs_mod_8_mn         , __rs_mod_8_mn         , __ps_mod_8_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv4m , avx512knl  , __fd_mod_4_mn         , __rd_mod_4_mn         , __pd_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , sv16m, avx512knl  , __fs_mod_16_mn        , __rs_mod_16_mn        , __ps_mod_16_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv8m , avx512knl  , __fd_mod_8_mn         , __rd_mod_8_mn         , __pd_mod_8_mn         ,__math_dispatch_error)

MTHINTRIN(mod  , ss   , avx512     , __fss_mod_vex         , __fss_mod_vex         , __mth_i_amod          ,__math_dispatch_error)
MTHINTRIN(mod  , ds   , avx512     , __fsd_mod_vex         , __fsd_mod_vex         , __mth_i_dmod          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4  , avx512     , __fvs_mod_vex         , __fvs_mod_vex         , __gs_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv2  , avx512     , __fvd_mod_vex         , __fvd_mod_vex         , __gd_mod_2_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv8  , avx512     , __fvs_mod_vex_256     , __fvs_mod_vex_256     , __gs_mod_8_p          ,__math_dispatch_error)
MTHINTRIN(mod  , dv4  , avx512     , __fvd_mod_vex_256     , __fvd_mod_vex_256     , __gd_mod_4_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv16 , avx512     , __fs_mod_16_z2yy      , __rs_mod_16_z2yy      , __gs_mod_16_p         ,__math_dispatch_error)
MTHINTRIN(mod  , dv8  , avx512     , __fd_mod_8_z2yy       , __rd_mod_8_z2yy       , __gd_mod_8_p          ,__math_dispatch_error)
MTHINTRIN(mod  , sv4m , avx512     , __fs_mod_4_mn         , __rs_mod_4_mn         , __ps_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv2m , avx512     , __fd_mod_2_mn         , __rd_mod_2_mn         , __pd_mod_2_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , sv8m , avx512     , __fs_mod_8_mn         , __rs_mod_8_mn         , __ps_mod_8_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv4m , avx512     , __fd_mod_4_mn         , __rd_mod_4_mn         , __pd_mod_4_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , sv16m, avx512     , __fs_mod_16_mn        , __rs_mod_16_mn        , __ps_mod_16_mn         ,__math_dispatch_error)
MTHINTRIN(mod  , dv8m , avx512     , __fd_mod_8_mn         , __rd_mod_8_mn         , __pd_mod_8_mn         ,__math_dispatch_error)


/*
k8-64e fast
     10 	call	__fsd_mod
     10 	call	__fss_mod
     25 	call	__fvd_mod
     15 	call	__fvs_mod
k8-64e relaxed
      3 	call	__fsd_mod
      3 	call	__fss_mod
k8-64e Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
barcelona fast
     10 	call	__fsd_mod
     10 	call	__fss_mod
     25 	call	__fvd_mod
     15 	call	__fvs_mod
barcelona relaxed
      3 	call	__fsd_mod
      3 	call	__fss_mod
barcelona Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
shanghai fast
     10 	call	__fsd_mod
     10 	call	__fss_mod
     25 	call	__fvd_mod
     15 	call	__fvs_mod
shanghai relaxed
      3 	call	__fsd_mod
      3 	call	__fss_mod
shanghai Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
istanbul fast
     10 	call	__fsd_mod
     10 	call	__fss_mod
     25 	call	__fvd_mod
     15 	call	__fvs_mod
istanbul relaxed
      3 	call	__fsd_mod
      3 	call	__fss_mod
istanbul Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
bulldozer fast
     10 	call	__fsd_mod_fma4
     10 	call	__fss_mod_fma4
     25 	call	__fvd_mod_fma4
     15 	call	__fvs_mod_fma4
bulldozer relaxed
      3 	call	__fsd_mod_fma4
      3 	call	__fss_mod_fma4
bulldozer Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
piledriver fast
      7 	call	__fsd_mod_fma4
      7 	call	__fss_mod_fma4
     15 	call	__fvd_mod_fma4
      9 	call	__fvs_mod_fma4
piledriver relaxed
      3 	call	__fsd_mod_fma4
      3 	call	__fss_mod_fma4
piledriver Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
p7 fast
     10 	call	__fmth_i_amod
     10 	call	__fmth_i_dmod
     25 	call	__fvdmod
     15 	call	__fvsmod
p7 relaxed
      3 	call	__fmth_i_amod
      3 	call	__fmth_i_dmod
p7 Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
core2 fast
     10 	call	__fsd_mod
     10 	call	__fss_mod
     25 	call	__fvd_mod
     15 	call	__fvs_mod
core2 relaxed
      3 	call	__fsd_mod
      3 	call	__fss_mod
core2 Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
penryn fast
     10 	call	__fsd_mod
     10 	call	__fss_mod
     25 	call	__fvd_mod
     15 	call	__fvs_mod
penryn relaxed
      3 	call	__fsd_mod
      3 	call	__fss_mod
penryn Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
nehalem fast
     10 	call	__fsd_mod
     10 	call	__fss_mod
     25 	call	__fvd_mod
     15 	call	__fvs_mod
nehalem relaxed
      3 	call	__fsd_mod
      3 	call	__fss_mod
nehalem Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
sandybridge fast
     10 	call	__fsd_mod_vex
     10 	call	__fss_mod_vex
      5 	call	__fvd_mod_vex
     15 	call	__fvd_mod_vex_256
      5 	call	__fvs_mod_vex
      5 	call	__fvs_mod_vex_256
sandybridge relaxed
      3 	call	__fsd_mod_vex
      3 	call	__fss_mod_vex
sandybridge Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
haswell fast
      7 	call	__fsd_mod_vex
      7 	call	__fss_mod_vex
      3 	call	__fvd_mod_vex
      9 	call	__fvd_mod_vex_256
      3 	call	__fvs_mod_vex
      3 	call	__fvs_mod_vex_256
haswell relaxed
      3 	call	__fsd_mod_vex
      3 	call	__fss_mod_vex
haswell Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
skylake fast
      7 	call	__fsd_mod_vex
      7 	call	__fss_mod_vex
      3 	call	__fvd_mod_vex
     21 	call	__fvd_mod_vex_256
      3 	call	__fvs_mod_vex
      9 	call	__fvs_mod_vex_256
skylake relaxed
      3 	call	__fsd_mod_vex
      3 	call	__fss_mod_vex
skylake Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
knl fast
      7 	call	__fsd_mod_vex
      7 	call	__fss_mod_vex
      3 	call	__fvd_mod_vex
     21 	call	__fvd_mod_vex_256
      3 	call	__fvs_mod_vex
      9 	call	__fvs_mod_vex_256
knl relaxed
      3 	call	__fsd_mod_vex
      3 	call	__fss_mod_vex
knl Kieee
      3 	call	__mth_i_amod
      3 	call	__mth_i_dmod
      1 ## pgf90__hd6CtZ5V5S.ilm -fn mod.f90 -opt 1 -terse 1 -inform warn -x 51 0x20
*/
