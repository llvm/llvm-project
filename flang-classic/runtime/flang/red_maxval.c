/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* red_maxval.c -- intrinsic reduction function */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

CONDFNLKN(>, maxval_int1, __INT1_T, 1)
CONDFNLKN(>, maxval_int2, __INT2_T, 1)
CONDFNLKN(>, maxval_int4, __INT4_T, 1)
CONDFNLKN(>, maxval_int8, __INT8_T, 1)
CONDFNLKN(>, maxval_real4, __REAL4_T, 1)
CONDFNLKN(>, maxval_real8, __REAL8_T, 1)
CONDFNLKN(>, maxval_real16, __REAL16_T, 1)
CONDSTRFNLKN(>, maxval_str, __STR_T, 1)

CONDFNLKN(>, maxval_int1, __INT1_T, 2)
CONDFNLKN(>, maxval_int2, __INT2_T, 2)
CONDFNLKN(>, maxval_int4, __INT4_T, 2)
CONDFNLKN(>, maxval_int8, __INT8_T, 2)
CONDFNLKN(>, maxval_real4, __REAL4_T, 2)
CONDFNLKN(>, maxval_real8, __REAL8_T, 2)
CONDFNLKN(>, maxval_real16, __REAL16_T, 2)
CONDSTRFNLKN(>, maxval_str, __STR_T, 2)

CONDFNLKN(>, maxval_int1, __INT1_T, 4)
CONDFNLKN(>, maxval_int2, __INT2_T, 4)
CONDFNLKN(>, maxval_int4, __INT4_T, 4)
CONDFNLKN(>, maxval_int8, __INT8_T, 4)
CONDFNLKN(>, maxval_real4, __REAL4_T, 4)
CONDFNLKN(>, maxval_real8, __REAL8_T, 4)
CONDFNLKN(>, maxval_real16, __REAL16_T, 4)
CONDSTRFNLKN(>, maxval_str, __STR_T, 4)

CONDFNLKN(>, maxval_int1, __INT1_T, 8)
CONDFNLKN(>, maxval_int2, __INT2_T, 8)
CONDFNLKN(>, maxval_int4, __INT4_T, 8)
CONDFNLKN(>, maxval_int8, __INT8_T, 8)
CONDFNLKN(>, maxval_real4, __REAL4_T, 8)
CONDFNLKN(>, maxval_real8, __REAL8_T, 8)
CONDFNLKN(>, maxval_real16, __REAL16_T, 8)
CONDSTRFNLKN(>, maxval_str, __STR_T, 8)

CONDFNG(>, maxval_int1, __INT1_T)
CONDFNG(>, maxval_int2, __INT2_T)
CONDFNG(>, maxval_int4, __INT4_T)
CONDFNG(>, maxval_int8, __INT8_T)
CONDFNG(>, maxval_real4, __REAL4_T)
CONDFNG(>, maxval_real8, __REAL8_T)
CONDFNG(>, maxval_real16, __REAL16_T)
CONDSTRFNG(>, maxval_str, __STR_T)

static void (*l_maxval[4][__NTYPES])() = TYPELIST3LK(l_maxval_);
static void (*g_maxval[__NTYPES])() = TYPELIST3(g_maxval_);

/* dim absent */

void ENTFTN(MAXVALS, maxvals)(char *rb, char *ab, char *mb, F90_Desc *rs,
                              F90_Desc *as, F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "MAXVAL";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_maxval[z.lk_shift][z.kind];
  z.g_fn = g_maxval[z.kind];
  z.zb = GET_DIST_MINS(z.kind);
  if (z.kind == __STR)
    memset(rb, *((char *)(z.zb)), z.len);
  I8(__fort_red_scalarlk)(&z, rb, ab, mb, rs, as, ms, NULL, __MAXVAL);
}

/* dim present */

void ENTFTN(MAXVAL, maxval)(char *rb, char *ab, char *mb, char *db,
                            F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                            F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "MAXVAL";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_maxval[z.lk_shift][z.kind];
  z.g_fn = g_maxval[z.kind];
  z.zb = GET_DIST_MINS(z.kind);
  if (z.kind == __STR)
    memset(rb, *((char *)(z.zb)), z.len);
  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(&z, rb, ab, mb, db, rs, as, ms2, ds, __MAXVAL);
    __fort_gfree(mb);
  } else {
    I8(__fort_red_arraylk)(&z, rb, ab, mb, db, rs, as, ms, ds, __MAXVAL);
  }
}

/* global MAXVAL accumulation */

void ENTFTN(REDUCE_MAXVAL, reduce_maxval)(char *hb, __INT_T *dimsb,
                                          __INT_T *nargb, char *rb,
                                          F90_Desc *hd, F90_Desc *dimsd,
                                          F90_Desc *nargd, F90_Desc *rd)

{
#if defined(DEBUG)
  if (dimsd == NULL || F90_TAG_G(dimsd) != __INT)
    __fort_abort("GLOBAL_MAXVAL: invalid dims descriptor");
  if (nargd == NULL || F90_TAG_G(nargd) != __INT)
    __fort_abort("REDUCE_MAXVAL: invalid arg count descriptor");
  if (*nargb != 1)
    __fort_abort("REDUCE_MAXVAL: arg count not 1");
#endif
  I8(__fort_global_reduce)(rb, hb, *dimsb, rd, hd, "MAXVAL", g_maxval);
}

void ENTFTN(GLOBAL_MAXVAL, global_maxval)(char *rb, char *hb, __INT_T *dimsb,
                                          F90_Desc *rd, F90_Desc *hd,
                                          F90_Desc *dimsd)
{
  I8(__fort_global_reduce)(rb, hb, *dimsb, rd, hd, "MAXVAL", g_maxval);
}
