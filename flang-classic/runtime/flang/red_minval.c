/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* red_minval.c -- intrinsic reduction function */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

CONDFNLKN(<, minval_int1, __INT1_T, 1)
CONDFNLKN(<, minval_int2, __INT2_T, 1)
CONDFNLKN(<, minval_int4, __INT4_T, 1)
CONDFNLKN(<, minval_int8, __INT8_T, 1)
CONDFNLKN(<, minval_real4, __REAL4_T, 1)
CONDFNLKN(<, minval_real8, __REAL8_T, 1)
CONDFNLKN(<, minval_real16, __REAL16_T, 1)
CONDSTRFNLKN(<, minval_str, __STR_T, 1)

CONDFNLKN(<, minval_int1, __INT1_T, 2)
CONDFNLKN(<, minval_int2, __INT2_T, 2)
CONDFNLKN(<, minval_int4, __INT4_T, 2)
CONDFNLKN(<, minval_int8, __INT8_T, 2)
CONDFNLKN(<, minval_real4, __REAL4_T, 2)
CONDFNLKN(<, minval_real8, __REAL8_T, 2)
CONDFNLKN(<, minval_real16, __REAL16_T, 2)
CONDSTRFNLKN(<, minval_str, __STR_T, 2)

CONDFNLKN(<, minval_int1, __INT1_T, 4)
CONDFNLKN(<, minval_int2, __INT2_T, 4)
CONDFNLKN(<, minval_int4, __INT4_T, 4)
CONDFNLKN(<, minval_int8, __INT8_T, 4)
CONDFNLKN(<, minval_real4, __REAL4_T, 4)
CONDFNLKN(<, minval_real8, __REAL8_T, 4)
CONDFNLKN(<, minval_real16, __REAL16_T, 4)
CONDSTRFNLKN(<, minval_str, __STR_T, 4)

CONDFNLKN(<, minval_int1, __INT1_T, 8)
CONDFNLKN(<, minval_int2, __INT2_T, 8)
CONDFNLKN(<, minval_int4, __INT4_T, 8)
CONDFNLKN(<, minval_int8, __INT8_T, 8)
CONDFNLKN(<, minval_real4, __REAL4_T, 8)
CONDFNLKN(<, minval_real8, __REAL8_T, 8)
CONDFNLKN(<, minval_real16, __REAL16_T, 8)
CONDSTRFNLKN(<, minval_str, __STR_T, 8)

CONDFNG(<, minval_int1, __INT1_T)
CONDFNG(<, minval_int2, __INT2_T)
CONDFNG(<, minval_int4, __INT4_T)
CONDFNG(<, minval_int8, __INT8_T)
CONDFNG(<, minval_real4, __REAL4_T)
CONDFNG(<, minval_real8, __REAL8_T)
CONDFNG(<, minval_real16, __REAL16_T)
CONDSTRFNG(<, minval_str, __STR_T)

static void (*l_minval[4][__NTYPES])() = TYPELIST3LK(l_minval_);
static void (*g_minval[__NTYPES])() = TYPELIST3(g_minval_);

/* dim absent */

void ENTFTN(MINVALS, minvals)(char *rb, char *ab, char *mb, F90_Desc *rs,
                              F90_Desc *as, F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "MINVAL";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_minval[z.lk_shift][z.kind];
  z.g_fn = g_minval[z.kind];
  z.zb = GET_DIST_MAXS(z.kind);
  if (z.kind == __STR)
    memset(rb, *((char *)(z.zb)), z.len);
  I8(__fort_red_scalarlk)(&z, rb, ab, mb, rs, as, ms, NULL, __MINVAL);
}

/* dim present */

void ENTFTN(MINVAL, minval)(char *rb, char *ab, char *mb, char *db,
                            F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                            F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "MINVAL";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_minval[z.lk_shift][z.kind];
  z.g_fn = g_minval[z.kind];
  z.zb = GET_DIST_MAXS(z.kind);
  if (z.kind == __STR)
    memset(rb, *((char *)(z.zb)), z.len);

  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(&z, rb, ab, mb, db, rs, as, ms2, ds, __MINVAL);
    __fort_gfree(mb);
  } else {
    I8(__fort_red_arraylk)(&z, rb, ab, mb, db, rs, as, ms, ds, __MINVAL);
  }
}

/* global MINVAL accumulation */

void ENTFTN(REDUCE_MINVAL, reduce_minval)(char *hb, __INT_T *dimsb,
                                          __INT_T *nargb, char *rb,
                                          F90_Desc *hd, F90_Desc *dimsd,
                                          F90_Desc *nargd, F90_Desc *rd)
{
#if defined(DEBUG)
  if (dimsd == NULL || F90_TAG_G(dimsd) != __INT)
    __fort_abort("GLOBAL_MINVAL: invalid dims descriptor");
  if (nargd == NULL || F90_TAG_G(nargd) != __INT)
    __fort_abort("REDUCE_MINVAL: invalid arg count descriptor");
  if (*nargb != 1)
    __fort_abort("REDUCE_MINVAL: arg count not 1");
#endif
  I8(__fort_global_reduce)(rb, hb, *dimsb, rd, hd, "MINVAL", g_minval);
}

void ENTFTN(GLOBAL_MINVAL, global_minval)(char *rb, char *hb, __INT_T *dimsb,
                                          F90_Desc *rd, F90_Desc *hd,
                                          F90_Desc *dimsd)
{
  I8(__fort_global_reduce)(rb, hb, *dimsb, rd, hd, "MINVAL", g_minval);
}
