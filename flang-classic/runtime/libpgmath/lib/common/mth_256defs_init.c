/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "mth_intrinsics.h"
#include "mth_tbldefs.h"
vrs8_t
MTH_DISPATCH_FUNC(__fs_acos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_acos][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_acos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_acos][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_acos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_acos][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_acos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_acos][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_acos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_acos][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_acos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_acos][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_acos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_acos][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_acos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_acos][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_acos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_acos][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_acos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_acos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_acos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_asin_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_asin][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_asin_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_asin][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_asin_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_asin][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_asin_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_asin][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_asin_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_asin][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_asin_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_asin][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_asin_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_asin][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_asin_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_asin][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_asin_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_asin][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_asin_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_asin_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_asin_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_atan_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_atan][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_atan_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_atan][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_atan_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_atan][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_atan_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_atan][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_atan_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_atan][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_atan_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_atan][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_atan_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_atan][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_atan_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_atan][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_atan_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_atan][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_atan_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_atan_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_atan_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_atan2_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_atan2][sv_sv8][frp_f];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_atan2_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_atan2][sv_sv8][frp_r];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_atan2_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_atan2][sv_sv8][frp_p];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_atan2_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_atan2][sv_sv8m][frp_f];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_atan2_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_atan2][sv_sv8m][frp_r];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_atan2_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_atan2][sv_sv8m][frp_p];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_atan2_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_atan2][sv_dv4][frp_f];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_atan2_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_atan2][sv_dv4][frp_r];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_atan2_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_atan2][sv_dv4][frp_p];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_atan2_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_atan2][sv_dv4m][frp_f];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_atan2_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_atan2][sv_dv4m][frp_r];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_atan2_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan2,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_atan2][sv_dv4m][frp_p];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_cos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_cos][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_cos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_cos][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_cos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_cos][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_cos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_cos][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_cos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_cos][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_cos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_cos][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_cos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_cos][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_cos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_cos][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_cos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_cos][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_cos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_cos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_cos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_sin_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sin][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_sin_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sin][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_sin_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sin][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_sin_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sin][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_sin_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sin][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_sin_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sin][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_sin_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sin][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_sin_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sin][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_sin_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sin][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_sin_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_sin_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_sin_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_tan_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_tan][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_tan_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_tan][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_tan_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_tan][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_tan_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_tan][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_tan_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_tan][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_tan_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_tan][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_tan_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_tan][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_tan_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_tan][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_tan_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_tan][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_tan_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_tan_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_tan_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_cosh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_cosh][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_cosh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_cosh][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_cosh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_cosh][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_cosh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_cosh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_cosh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_cosh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_cosh][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_cosh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_cosh][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_cosh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_cosh][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_cosh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_cosh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_cosh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_sinh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sinh][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_sinh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sinh][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_sinh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sinh][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_sinh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_sinh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_sinh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_sinh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sinh][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_sinh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sinh][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_sinh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sinh][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_sinh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_sinh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_sinh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_tanh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_tanh][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_tanh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_tanh][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_tanh_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_tanh][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_tanh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_tanh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_tanh_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_tanh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_tanh][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_tanh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_tanh][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_tanh_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_tanh][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_tanh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_tanh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_tanh_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_exp_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_exp][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_exp_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_exp][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_exp_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_exp][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_exp_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_exp][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_exp_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_exp][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_exp_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_exp][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_exp_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_exp][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_exp_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_exp][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_exp_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_exp][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_exp_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_exp_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_exp_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_log_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_log][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_log_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_log][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_log_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_log][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_log_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_log][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_log_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_log][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_log_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_log][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_log_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_log][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_log_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_log][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_log_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_log][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_log_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_log][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_log_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_log][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_log_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_log][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_log10_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_log10][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_log10_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_log10][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_log10_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_log10][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_log10_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_log10][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_log10_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_log10][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_log10_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_log10][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_log10_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_log10][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_log10_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_log10][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_log10_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_log10][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_log10_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_log10_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_log10_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_mod_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_mod][sv_sv8][frp_f];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_mod_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_mod][sv_sv8][frp_r];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_mod_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_mod][sv_sv8][frp_p];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_mod_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_mod][sv_sv8m][frp_f];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_mod_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_mod][sv_sv8m][frp_r];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_mod_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_mod][sv_sv8m][frp_p];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_mod_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_mod][sv_dv4][frp_f];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_mod_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_mod][sv_dv4][frp_r];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_mod_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_mod][sv_dv4][frp_p];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_mod_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_mod][sv_dv4m][frp_f];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_mod_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_mod][sv_dv4m][frp_r];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_mod_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_mod,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_mod][sv_dv4m][frp_p];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_pow_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_pow][sv_sv8][frp_f];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_pow_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_pow][sv_sv8][frp_r];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_pow_8)(vrs8_t x, vrs8_t y)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t))MTH_DISPATCH_TBL[func_pow][sv_sv8][frp_p];
  return (fptr(x, y));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_pow_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_pow][sv_sv8m][frp_f];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_pow_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_pow][sv_sv8m][frp_r];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_pow_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_pow][sv_sv8m][frp_p];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_pow_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_pow][sv_dv4][frp_f];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_pow_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_pow][sv_dv4][frp_r];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_pow_4)(vrd4_t x, vrd4_t y)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t))MTH_DISPATCH_TBL[func_pow][sv_dv4][frp_p];
  return (fptr(x, y));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_pow_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_pow][sv_dv4m][frp_f];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_pow_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_pow][sv_dv4m][frp_r];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_pow_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_pow][sv_dv4m][frp_p];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_powi1_8)(vrs8_t x, int32_t iy)
{
  vrs8_t (*fptr)(vrs8_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_sv8][frp_f];
  return(fptr(x,iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_powi1_8)(vrs8_t x, int32_t iy)
{
  vrs8_t (*fptr)(vrs8_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_sv8][frp_r];
  return(fptr(x,iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_powi1_8)(vrs8_t x, int32_t iy)
{
  vrs8_t (*fptr)(vrs8_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_sv8][frp_p];
  return(fptr(x,iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_powi1_8m)(vrs8_t x, int32_t iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, int32_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, int32_t, vis8_t))MTH_DISPATCH_TBL[func_powi1][sv_sv8m][frp_f];
  return(fptr(x,iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_powi1_8m)(vrs8_t x, int32_t iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, int32_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, int32_t, vis8_t))MTH_DISPATCH_TBL[func_powi1][sv_sv8m][frp_r];
  return(fptr(x,iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_powi1_8m)(vrs8_t x, int32_t iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, int32_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, int32_t, vis8_t))MTH_DISPATCH_TBL[func_powi1][sv_sv8m][frp_p];
  return(fptr(x,iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_powi_8)(vrs8_t x, vis8_t iy)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_powi][sv_sv8][frp_f];
  return(fptr(x, iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_powi_8)(vrs8_t x, vis8_t iy)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_powi][sv_sv8][frp_r];
  return(fptr(x, iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_powi_8)(vrs8_t x, vis8_t iy)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_powi][sv_sv8][frp_p];
  return(fptr(x, iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_powi_8m)(vrs8_t x, vis8_t iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vis8_t, vis8_t))MTH_DISPATCH_TBL[func_powi][sv_sv8m][frp_f];
  return(fptr(x, iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_powi_8m)(vrs8_t x, vis8_t iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vis8_t, vis8_t))MTH_DISPATCH_TBL[func_powi][sv_sv8m][frp_r];
  return(fptr(x, iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_powi_8m)(vrs8_t x, vis8_t iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vis8_t, vis8_t))MTH_DISPATCH_TBL[func_powi][sv_sv8m][frp_p];
  return(fptr(x, iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_powk1_8)(vrs8_t x, long long iy)
{
  vrs8_t (*fptr)(vrs8_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_sv8][frp_f];
  return(fptr(x, iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_powk1_8)(vrs8_t x, long long iy)
{
  vrs8_t (*fptr)(vrs8_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_sv8][frp_r];
  return(fptr(x, iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_powk1_8)(vrs8_t x, long long iy)
{
  vrs8_t (*fptr)(vrs8_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_sv8][frp_p];
  return(fptr(x, iy));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_powk1_8m)(vrs8_t x, long long iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, long long, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, long long, vis8_t))MTH_DISPATCH_TBL[func_powk1][sv_sv8m][frp_f];
  return(fptr(x, iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_powk1_8m)(vrs8_t x, long long iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, long long, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, long long, vis8_t))MTH_DISPATCH_TBL[func_powk1][sv_sv8m][frp_r];
  return(fptr(x, iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_powk1_8m)(vrs8_t x, long long iy, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, long long, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, long long, vis8_t))MTH_DISPATCH_TBL[func_powk1][sv_sv8m][frp_p];
  return(fptr(x, iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_powk_8)(vrs8_t x, vid4_t iyu, vid4_t iyl)
{
  vrs8_t (*fptr)(vrs8_t, vid4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vid4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_sv8][frp_f];
  return(fptr(x, iyu, iyl));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_powk_8)(vrs8_t x, vid4_t iyu, vid4_t iyl)
{
  vrs8_t (*fptr)(vrs8_t, vid4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vid4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_sv8][frp_r];
  return(fptr(x, iyu, iyl));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_powk_8)(vrs8_t x, vid4_t iyu, vid4_t iyl)
{
  vrs8_t (*fptr)(vrs8_t, vid4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vid4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_sv8][frp_p];
  return(fptr(x, iyu, iyl));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_powk_8m)(vrs8_t x, vid4_t iyu, vid4_t iyl, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, vid4_t, vid4_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vid4_t, vid4_t, vis8_t))MTH_DISPATCH_TBL[func_powk][sv_sv8m][frp_f];
  return(fptr(x, iyu, iyl, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_powk_8m)(vrs8_t x, vid4_t iyu, vid4_t iyl, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, vid4_t, vid4_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vid4_t, vid4_t, vis8_t))MTH_DISPATCH_TBL[func_powk][sv_sv8m][frp_r];
  return(fptr(x, iyu, iyl, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_powk_8m)(vrs8_t x, vid4_t iyu, vid4_t iyl, vis8_t mask)
{
  vrs8_t (*fptr)(vrs8_t, vid4_t, vid4_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vid4_t, vid4_t, vis8_t))MTH_DISPATCH_TBL[func_powk][sv_sv8m][frp_p];
  return(fptr(x, iyu, iyl, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_powi1_4)(vrd4_t x, int32_t iy)
{
  vrd4_t (*fptr)(vrd4_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_dv4][frp_f];
  return(fptr(x,iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_powi1_4)(vrd4_t x, int32_t iy)
{
  vrd4_t (*fptr)(vrd4_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_dv4][frp_r];
  return(fptr(x,iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_powi1_4)(vrd4_t x, int32_t iy)
{
  vrd4_t (*fptr)(vrd4_t, int32_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, int32_t))MTH_DISPATCH_TBL[func_powi1][sv_dv4][frp_p];
  return(fptr(x,iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_powi1_4m)(vrd4_t x, int32_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, int32_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, int32_t, vid4_t))MTH_DISPATCH_TBL[func_powi1][sv_dv4m][frp_f];
  return(fptr(x,iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_powi1_4m)(vrd4_t x, int32_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, int32_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, int32_t, vid4_t))MTH_DISPATCH_TBL[func_powi1][sv_dv4m][frp_r];
  return(fptr(x,iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_powi1_4m)(vrd4_t x, int32_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, int32_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi1,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, int32_t, vid4_t))MTH_DISPATCH_TBL[func_powi1][sv_dv4m][frp_p];
  return(fptr(x,iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_powi_4)(vrd4_t x, vis4_t iy)
{
  vrd4_t (*fptr)(vrd4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_dv4][frp_f];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_powi_4)(vrd4_t x, vis4_t iy)
{
  vrd4_t (*fptr)(vrd4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_dv4][frp_r];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_powi_4)(vrd4_t x, vis4_t iy)
{
  vrd4_t (*fptr)(vrd4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vis4_t))MTH_DISPATCH_TBL[func_powi][sv_dv4][frp_p];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_powi_4m)(vrd4_t x, vis4_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, vis4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vis4_t, vid4_t))MTH_DISPATCH_TBL[func_powi][sv_dv4m][frp_f];
  return(fptr(x, iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_powi_4m)(vrd4_t x, vis4_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, vis4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vis4_t, vid4_t))MTH_DISPATCH_TBL[func_powi][sv_dv4m][frp_r];
  return(fptr(x, iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_powi_4m)(vrd4_t x, vis4_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, vis4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powi,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vis4_t, vid4_t))MTH_DISPATCH_TBL[func_powi][sv_dv4m][frp_p];
  return(fptr(x, iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_powk1_4)(vrd4_t x, long long iy)
{
  vrd4_t (*fptr)(vrd4_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_dv4][frp_f];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_powk1_4)(vrd4_t x, long long iy)
{
  vrd4_t (*fptr)(vrd4_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_dv4][frp_r];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_powk1_4)(vrd4_t x, long long iy)
{
  vrd4_t (*fptr)(vrd4_t, long long);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, long long))MTH_DISPATCH_TBL[func_powk1][sv_dv4][frp_p];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_powk1_4m)(vrd4_t x, long long iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, long long, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, long long, vid4_t))MTH_DISPATCH_TBL[func_powk1][sv_dv4m][frp_f];
  return(fptr(x, iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_powk1_4m)(vrd4_t x, long long iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, long long, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, long long, vid4_t))MTH_DISPATCH_TBL[func_powk1][sv_dv4m][frp_r];
  return(fptr(x, iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_powk1_4m)(vrd4_t x, long long iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, long long, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk1,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, long long, vid4_t))MTH_DISPATCH_TBL[func_powk1][sv_dv4m][frp_p];
  return(fptr(x, iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_powk_4)(vrd4_t x, vid4_t iy)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_dv4][frp_f];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_powk_4)(vrd4_t x, vid4_t iy)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_dv4][frp_r];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_powk_4)(vrd4_t x, vid4_t iy)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_dv4][frp_p];
  return(fptr(x, iy));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_powk_4m)(vrd4_t x, vid4_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vid4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_dv4m][frp_f];
  return(fptr(x, iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_powk_4m)(vrd4_t x, vid4_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vid4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_dv4m][frp_r];
  return(fptr(x, iy, mask));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_powk_4m)(vrd4_t x, vid4_t iy, vid4_t mask)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_powk,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vid4_t, vid4_t))MTH_DISPATCH_TBL[func_powk][sv_dv4m][frp_p];
  return(fptr(x, iy, mask));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_sincos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sincos][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_sincos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sincos][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_sincos_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_sincos][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_sincos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_sincos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_sincos_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_sincos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sincos][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_sincos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sincos][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_sincos_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_sincos][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_sincos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_sincos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_sincos_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sincos,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sincos][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_aint_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_aint][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_aint_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_aint][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_aint_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_aint][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_aint_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_aint][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_aint_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_aint][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_aint_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_aint][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_aint_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_aint][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_aint_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_aint][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_aint_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_aint][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_aint_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_aint][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_aint_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_aint][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_aint_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_aint,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_aint][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_ceil_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_ceil][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_ceil_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_ceil][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_ceil_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_ceil][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_ceil_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_ceil_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_ceil_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_ceil_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_ceil][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_ceil_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_ceil][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_ceil_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_ceil][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_ceil_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_ceil_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_ceil_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_ceil,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_ceil][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_floor_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv8,frp_f);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_floor][sv_sv8][frp_f];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_floor_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv8,frp_r);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_floor][sv_sv8][frp_r];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_floor_8)(vrs8_t x)
{
  vrs8_t (*fptr)(vrs8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv8,frp_p);
  fptr = (vrs8_t(*)(vrs8_t))MTH_DISPATCH_TBL[func_floor][sv_sv8][frp_p];
  return (fptr(x));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_floor_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_floor][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_floor_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_floor][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_floor_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_floor][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_floor_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv4,frp_f);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_floor][sv_dv4][frp_f];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_floor_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv4,frp_r);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_floor][sv_dv4][frp_r];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_floor_4)(vrd4_t x)
{
  vrd4_t (*fptr)(vrd4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv4,frp_p);
  fptr = (vrd4_t(*)(vrd4_t))MTH_DISPATCH_TBL[func_floor][sv_dv4][frp_p];
  return (fptr(x));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_floor_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_floor][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_floor_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_floor][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_floor_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_floor,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_floor][sv_dv4m][frp_p];
  return (fptr(x, m));
}

//////////
//// EXPERIMENTAL - COMPLEX - start
////////////
vcs4_t
MTH_DISPATCH_FUNC(__fc_acos_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_acos][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_acos_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_acos][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_acos_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_acos][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_acos_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_acos_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_acos_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_acos][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_acos_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_acos][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_acos_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_acos][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_acos_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_acos][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_acos_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_acos_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_acos_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_acos,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_acos][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_asin_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_asin][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_asin_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_asin][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_asin_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_asin][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_asin_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_asin_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_asin_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_asin][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_asin_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_asin][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_asin_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_asin][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_asin_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_asin][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_asin_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_asin_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_asin_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_asin,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_asin][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_atan_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_atan][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_atan_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_atan][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_atan_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_atan][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_atan_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_atan_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_atan_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_atan][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_atan_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_atan][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_atan_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_atan][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_atan_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_atan][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_atan_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_atan_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_atan_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_atan,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_atan][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_cos_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_cos][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_cos_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_cos][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_cos_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_cos][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_cos_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_cos_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_cos_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_cos][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_cos_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_cos][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_cos_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_cos][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_cos_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_cos][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_cos_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_cos_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_cos_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cos,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_cos][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_sin_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sin][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_sin_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sin][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_sin_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sin][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_sin_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_sin_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_sin_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sin][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_sin_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sin][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_sin_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sin][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_sin_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sin][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_sin_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_sin_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_sin_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sin,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sin][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_tan_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_tan][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_tan_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_tan][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_tan_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_tan][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_tan_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_tan_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_tan_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_tan][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_tan_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_tan][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_tan_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_tan][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_tan_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_tan][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_tan_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_tan_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_tan_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tan,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_tan][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_cosh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_cosh][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_cosh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_cosh][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_cosh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_cosh][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_cosh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_cosh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_cosh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_cosh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_cosh][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_cosh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_cosh][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_cosh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_cosh][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_cosh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_cosh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_cosh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_cosh,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_cosh][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_sinh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sinh][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_sinh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sinh][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_sinh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sinh][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_sinh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_sinh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_sinh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_sinh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sinh][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_sinh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sinh][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_sinh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sinh][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_sinh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_sinh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_sinh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sinh,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sinh][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_tanh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_tanh][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_tanh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_tanh][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_tanh_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_tanh][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_tanh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_tanh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_tanh_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_tanh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_tanh][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_tanh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_tanh][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_tanh_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_tanh][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_tanh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_tanh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_tanh_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_tanh,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_tanh][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_exp_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_exp][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_exp_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_exp][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_exp_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_exp][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_exp_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_exp_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_exp_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_exp][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_exp_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_exp][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_exp_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_exp][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_exp_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_exp][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_exp_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_exp_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_exp_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_exp,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_exp][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_log_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_log][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_log_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_log][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_log_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_log][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_log_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_log][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_log_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_log][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_log_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_log][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_log_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_log][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_log_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_log][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_log_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_log][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_log_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_log][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_log_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_log][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_log_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_log][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_log10_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_log10][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_log10_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_log10][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_log10_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_log10][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_log10_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_log10_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_log10_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_log10][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_log10_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_log10][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_log10_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_log10][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_log10_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_log10][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_log10_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_log10_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_log10_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_log10,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_log10][sv_zv2m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_pow_4)(vcs4_t x, vcs4_t y)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t))MTH_DISPATCH_TBL[func_pow][sv_cv4][frp_f];
  return (fptr(x, y));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_pow_4)(vcs4_t x, vcs4_t y)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t))MTH_DISPATCH_TBL[func_pow][sv_cv4][frp_r];
  return (fptr(x, y));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_pow_4)(vcs4_t x, vcs4_t y)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t))MTH_DISPATCH_TBL[func_pow][sv_cv4][frp_p];
  return (fptr(x, y));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_pow_4m)(vcs4_t x, vcs4_t y, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t, vis4_t))MTH_DISPATCH_TBL[func_pow][sv_cv4m][frp_f];
  return (fptr(x, y, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_pow_4m)(vcs4_t x, vcs4_t y, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t, vis4_t))MTH_DISPATCH_TBL[func_pow][sv_cv4m][frp_r];
  return (fptr(x, y, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_pow_4m)(vcs4_t x, vcs4_t y, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t, vis4_t))MTH_DISPATCH_TBL[func_pow][sv_cv4m][frp_p];
  return (fptr(x, y, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_pow_2)(vcd2_t x, vcd2_t y)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t))MTH_DISPATCH_TBL[func_pow][sv_zv2][frp_f];
  return (fptr(x, y));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_pow_2)(vcd2_t x, vcd2_t y)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t))MTH_DISPATCH_TBL[func_pow][sv_zv2][frp_r];
  return (fptr(x, y));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_pow_2)(vcd2_t x, vcd2_t y)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t))MTH_DISPATCH_TBL[func_pow][sv_zv2][frp_p];
  return (fptr(x, y));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_pow_2m)(vcd2_t x, vcd2_t y, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t, vid2_t))MTH_DISPATCH_TBL[func_pow][sv_zv2m][frp_f];
  return (fptr(x, y, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_pow_2m)(vcd2_t x, vcd2_t y, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t, vid2_t))MTH_DISPATCH_TBL[func_pow][sv_zv2m][frp_r];
  return (fptr(x, y, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_pow_2m)(vcd2_t x, vcd2_t y, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_pow,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t, vid2_t))MTH_DISPATCH_TBL[func_pow][sv_zv2m][frp_p];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_div_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_div][sv_sv8m][frp_f];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_div_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_div][sv_sv8m][frp_r];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_div_8m)(vrs8_t x, vrs8_t y, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)(vrs8_t, vrs8_t, vis8_t))MTH_DISPATCH_TBL[func_div][sv_sv8m][frp_p];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_div_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_div][sv_dv4m][frp_f];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_div_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_div][sv_dv4m][frp_r];
  return (fptr(x, y, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_div_4m)(vrd4_t x, vrd4_t y, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)(vrd4_t, vrd4_t, vid4_t))MTH_DISPATCH_TBL[func_div][sv_dv4m][frp_p];
  return (fptr(x, y, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_div_4)(vcs4_t x, vcs4_t y)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t))MTH_DISPATCH_TBL[func_div][sv_cv4][frp_f];
  return (fptr(x, y));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_div_4)(vcs4_t x, vcs4_t y)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t))MTH_DISPATCH_TBL[func_div][sv_cv4][frp_r];
  return (fptr(x, y));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_div_4)(vcs4_t x, vcs4_t y)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t))MTH_DISPATCH_TBL[func_div][sv_cv4][frp_p];
  return (fptr(x, y));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_div_4m)(vcs4_t x, vcs4_t y, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t, vis4_t))MTH_DISPATCH_TBL[func_div][sv_cv4m][frp_f];
  return (fptr(x, y, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_div_4m)(vcs4_t x, vcs4_t y, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t, vis4_t))MTH_DISPATCH_TBL[func_div][sv_cv4m][frp_r];
  return (fptr(x, y, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_div_4m)(vcs4_t x, vcs4_t y, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)(vcs4_t, vcs4_t, vis4_t))MTH_DISPATCH_TBL[func_div][sv_cv4m][frp_p];
  return (fptr(x, y, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_div_2)(vcd2_t x, vcd2_t y)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t))MTH_DISPATCH_TBL[func_div][sv_zv2][frp_f];
  return (fptr(x, y));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_div_2)(vcd2_t x, vcd2_t y)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t))MTH_DISPATCH_TBL[func_div][sv_zv2][frp_r];
  return (fptr(x, y));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_div_2)(vcd2_t x, vcd2_t y)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t))MTH_DISPATCH_TBL[func_div][sv_zv2][frp_p];
  return (fptr(x, y));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_div_2m)(vcd2_t x, vcd2_t y, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t, vid2_t))MTH_DISPATCH_TBL[func_div][sv_zv2m][frp_f];
  return (fptr(x, y, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_div_2m)(vcd2_t x, vcd2_t y, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t, vid2_t))MTH_DISPATCH_TBL[func_div][sv_zv2m][frp_r];
  return (fptr(x, y, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_div_2m)(vcd2_t x, vcd2_t y, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_div,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)(vcd2_t, vcd2_t, vid2_t))MTH_DISPATCH_TBL[func_div][sv_zv2m][frp_p];
  return (fptr(x, y, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__fs_sqrt_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_sv8m,frp_f);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_sv8m][frp_f];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__rs_sqrt_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_sv8m,frp_r);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_sv8m][frp_r];
  return (fptr(x, m));
}

vrs8_t
MTH_DISPATCH_FUNC(__ps_sqrt_8m)(vrs8_t x, vis8_t m)
{
  vrs8_t (*fptr)(vrs8_t, vis8_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_sv8m,frp_p);
  fptr = (vrs8_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_sv8m][frp_p];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__fd_sqrt_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_dv4m,frp_f);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_dv4m][frp_f];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__rd_sqrt_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_dv4m,frp_r);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_dv4m][frp_r];
  return (fptr(x, m));
}

vrd4_t
MTH_DISPATCH_FUNC(__pd_sqrt_4m)(vrd4_t x, vid4_t m)
{
  vrd4_t (*fptr)(vrd4_t, vid4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_dv4m,frp_p);
  fptr = (vrd4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_dv4m][frp_p];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_sqrt_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv4,frp_f);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv4][frp_f];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_sqrt_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv4,frp_r);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv4][frp_r];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_sqrt_4)(vcs4_t x)
{
  vcs4_t (*fptr)(vcs4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv4,frp_p);
  fptr = (vcs4_t(*)(vcs4_t))MTH_DISPATCH_TBL[func_sqrt][sv_cv4][frp_p];
  return (fptr(x));
}

vcs4_t
MTH_DISPATCH_FUNC(__fc_sqrt_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv4m,frp_f);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_cv4m][frp_f];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__rc_sqrt_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv4m,frp_r);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_cv4m][frp_r];
  return (fptr(x, m));
}

vcs4_t
MTH_DISPATCH_FUNC(__pc_sqrt_4m)(vcs4_t x, vis4_t m)
{
  vcs4_t (*fptr)(vcs4_t, vis4_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_cv4m,frp_p);
  fptr = (vcs4_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_cv4m][frp_p];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_sqrt_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv2,frp_f);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sqrt][sv_zv2][frp_f];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_sqrt_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv2,frp_r);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sqrt][sv_zv2][frp_r];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_sqrt_2)(vcd2_t x)
{
  vcd2_t (*fptr)(vcd2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv2,frp_p);
  fptr = (vcd2_t(*)(vcd2_t))MTH_DISPATCH_TBL[func_sqrt][sv_zv2][frp_p];
  return (fptr(x));
}

vcd2_t
MTH_DISPATCH_FUNC(__fz_sqrt_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv2m,frp_f);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_zv2m][frp_f];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__rz_sqrt_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv2m,frp_r);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_zv2m][frp_r];
  return (fptr(x, m));
}

vcd2_t
MTH_DISPATCH_FUNC(__pz_sqrt_2m)(vcd2_t x, vid2_t m)
{
  vcd2_t (*fptr)(vcd2_t, vid2_t);
  _MTH_I_INIT();
  _MTH_I_STATS_INC(func_sqrt,sv_zv2m,frp_p);
  fptr = (vcd2_t(*)())MTH_DISPATCH_TBL[func_sqrt][sv_zv2m][frp_p];
  return (fptr(x, m));
}
////////////
//// EXPERIMENTAL - COMPLEX - end
////////////

#if     defined(TARGET_LINUX_X8664) && ! defined(MTH_I_INTRIN_STATS) && ! defined(MTH_I_INTRIN_INIT)
vrd4_t __gvd_atan4(vrd4_t) __attribute__ ((weak, alias ("__fd_atan_4")));
vrd4_t __gvd_atan4_mask(vrd4_t,vid4_t) __attribute__ ((weak, alias ("__fd_atan_4m")));
vrs8_t __gvs_atan8(vrs8_t) __attribute__ ((weak, alias ("__fs_atan_8")));
vrs8_t __gvs_atan8_mask(vrs8_t,vis8_t) __attribute__ ((weak, alias ("__fs_atan_8m")));
vrd4_t __gvd_exp4(vrd4_t) __attribute__ ((weak, alias ("__fd_exp_4")));
vrd4_t __gvd_exp4_mask(vrd4_t,vid4_t) __attribute__ ((weak, alias ("__fd_exp_4m")));
vrs8_t __gvs_exp8(vrs8_t) __attribute__ ((weak, alias ("__fs_exp_8")));
vrs8_t __gvs_exp8_mask(vrs8_t,vis8_t) __attribute__ ((weak, alias ("__fs_exp_8m")));
vrd4_t __gvd_log4(vrd4_t) __attribute__ ((weak, alias ("__fd_log_4")));
vrd4_t __gvd_log4_mask(vrd4_t,vid4_t) __attribute__ ((weak, alias ("__fd_log_4m")));
vrs8_t __gvs_log8(vrs8_t) __attribute__ ((weak, alias ("__fs_log_8")));
vrs8_t __gvs_log8_mask(vrs8_t,vis8_t) __attribute__ ((weak, alias ("__fs_log_8m")));
vrd4_t __gvd_pow4(vrd4_t,vrd4_t) __attribute__ ((weak, alias ("__fd_pow_4")));
vrd4_t __gvd_pow4_mask(vrd4_t,vrd4_t,vid4_t) __attribute__ ((weak, alias ("__fd_pow_4m")));
vrs8_t __gvs_pow8(vrs8_t,vrs8_t) __attribute__ ((weak, alias ("__fs_pow_8")));
vrs8_t __gvs_pow8_mask(vrs8_t,vrs8_t,vis8_t) __attribute__ ((weak, alias ("__fs_pow_8m")));
#endif
