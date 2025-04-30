/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* red_minloc.c -- intrinsic reduction function */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"
#include <string.h>

MLOCFNLKN(<, minloc_int1, __INT1_T, 1)
MLOCFNLKN(<, minloc_int2, __INT2_T, 1)
MLOCFNLKN(<, minloc_int4, __INT4_T, 1)
MLOCFNLKN(<, minloc_int8, __INT8_T, 1)
MLOCFNLKN(<, minloc_real4, __REAL4_T, 1)
MLOCFNLKN(<, minloc_real8, __REAL8_T, 1)
MLOCFNLKN(<, minloc_real16, __REAL16_T, 1)
MLOCSTRFNLKN(<, minloc_str, __STR_T, 1)

MLOCFNLKN(<, minloc_int1, __INT1_T, 2)
MLOCFNLKN(<, minloc_int2, __INT2_T, 2)
MLOCFNLKN(<, minloc_int4, __INT4_T, 2)
MLOCFNLKN(<, minloc_int8, __INT8_T, 2)
MLOCFNLKN(<, minloc_real4, __REAL4_T, 2)
MLOCFNLKN(<, minloc_real8, __REAL8_T, 2)
MLOCFNLKN(<, minloc_real16, __REAL16_T, 2)
MLOCSTRFNLKN(<, minloc_str, __STR_T, 2)

MLOCFNLKN(<, minloc_int1, __INT1_T, 4)
MLOCFNLKN(<, minloc_int2, __INT2_T, 4)
MLOCFNLKN(<, minloc_int4, __INT4_T, 4)
MLOCFNLKN(<, minloc_int8, __INT8_T, 4)
MLOCFNLKN(<, minloc_real4, __REAL4_T, 4)
MLOCFNLKN(<, minloc_real8, __REAL8_T, 4)
MLOCFNLKN(<, minloc_real16, __REAL16_T, 4)
MLOCSTRFNLKN(<, minloc_str, __STR_T, 4)

MLOCFNLKN(<, minloc_int1, __INT1_T, 8)
MLOCFNLKN(<, minloc_int2, __INT2_T, 8)
MLOCFNLKN(<, minloc_int4, __INT4_T, 8)
MLOCFNLKN(<, minloc_int8, __INT8_T, 8)
MLOCFNLKN(<, minloc_real4, __REAL4_T, 8)
MLOCFNLKN(<, minloc_real8, __REAL8_T, 8)
MLOCFNLKN(<, minloc_real16, __REAL16_T, 8)
MLOCSTRFNLKN(<, minloc_str, __STR_T, 8)

MLOCFNG(<, minloc_int1, __INT1_T)
MLOCFNG(<, minloc_int2, __INT2_T)
MLOCFNG(<, minloc_int4, __INT4_T)
MLOCFNG(<, minloc_int8, __INT8_T)
MLOCFNG(<, minloc_real4, __REAL4_T)
MLOCFNG(<, minloc_real8, __REAL8_T)
MLOCFNG(<, minloc_real16, __REAL16_T)
MLOCSTRFNG(<, minloc_str, __STR_T)

static void (*l_minloc_b[4][__NTYPES])() = TYPELIST3LK(l_minloc_);
static void (*g_minloc[__NTYPES])() = TYPELIST3(g_minloc_);

KMLOCFNLKN(<, kminloc_int1, __INT1_T, 1)
KMLOCFNLKN(<, kminloc_int2, __INT2_T, 1)
KMLOCFNLKN(<, kminloc_int4, __INT4_T, 1)
KMLOCFNLKN(<, kminloc_int8, __INT8_T, 1)
KMLOCFNLKN(<, kminloc_real4, __REAL4_T, 1)
KMLOCFNLKN(<, kminloc_real8, __REAL8_T, 1)
KMLOCFNLKN(<, kminloc_real16, __REAL16_T, 1)
KMLOCSTRFNLKN(<, kminloc_str, __STR_T, 1)

KMLOCFNLKN(<, kminloc_int1, __INT1_T, 2)
KMLOCFNLKN(<, kminloc_int2, __INT2_T, 2)
KMLOCFNLKN(<, kminloc_int4, __INT4_T, 2)
KMLOCFNLKN(<, kminloc_int8, __INT8_T, 2)
KMLOCFNLKN(<, kminloc_real4, __REAL4_T, 2)
KMLOCFNLKN(<, kminloc_real8, __REAL8_T, 2)
KMLOCFNLKN(<, kminloc_real16, __REAL16_T, 2)
KMLOCSTRFNLKN(<, kminloc_str, __STR_T, 2)

KMLOCFNLKN(<, kminloc_int1, __INT1_T, 4)
KMLOCFNLKN(<, kminloc_int2, __INT2_T, 4)
KMLOCFNLKN(<, kminloc_int4, __INT4_T, 4)
KMLOCFNLKN(<, kminloc_int8, __INT8_T, 4)
KMLOCFNLKN(<, kminloc_real4, __REAL4_T, 4)
KMLOCFNLKN(<, kminloc_real8, __REAL8_T, 4)
KMLOCFNLKN(<, kminloc_real16, __REAL16_T, 4)
KMLOCSTRFNLKN(<, kminloc_str, __STR_T, 4)

KMLOCFNLKN(<, kminloc_int1, __INT1_T, 8)
KMLOCFNLKN(<, kminloc_int2, __INT2_T, 8)
KMLOCFNLKN(<, kminloc_int4, __INT4_T, 8)
KMLOCFNLKN(<, kminloc_int8, __INT8_T, 8)
KMLOCFNLKN(<, kminloc_real4, __REAL4_T, 8)
KMLOCFNLKN(<, kminloc_real8, __REAL8_T, 8)
KMLOCFNLKN(<, kminloc_real16, __REAL16_T, 8)
KMLOCSTRFNLKN(<, kminloc_str, __STR_T, 8)

KMLOCFNG(<, kminloc_int1, __INT1_T)
KMLOCFNG(<, kminloc_int2, __INT2_T)
KMLOCFNG(<, kminloc_int4, __INT4_T)
KMLOCFNG(<, kminloc_int8, __INT8_T)
KMLOCFNG(<, kminloc_real4, __REAL4_T)
KMLOCFNG(<, kminloc_real8, __REAL8_T)
KMLOCFNG(<, kminloc_real16, __REAL16_T)
KMLOCSTRFNG(<, kminloc_str, __STR_T)

static void (*l_kminloc_b[4][__NTYPES])() = TYPELIST3LK(l_kminloc_);
static void (*g_kminloc[__NTYPES])() = TYPELIST3(g_kminloc_);

/* dim absent */
static void minlocs_common(red_parm *z, __INT_T *rb, char *ab, char *mb,
                           F90_Desc *rs, F90_Desc *as, F90_Desc *ms)
{
  double vb[4];
  char *strvb;

  __fort_red_what = "MINLOC";

  z->kind = F90_KIND_G(as);
  z->len = F90_LEN_G(as);
  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z->mask_present) {
    z->lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z->lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z->l_fn_b = l_minloc_b[z->lk_shift][z->kind];
  z->g_fn = g_minloc[z->kind];
  z->zb = GET_DIST_MAXS(z->kind);

  if (z->kind == __STR) {
    strvb = (char *)__fort_gmalloc(z->len);
    memset(strvb, *((char *)z->zb), z->len);
    I8(__fort_red_scalarlk)(z, strvb, ab, mb, rs, as, ms, rb, __MINLOC);
    __fort_gfree(strvb);
  } else {
    I8(__fort_red_scalarlk)(z, (char *)vb, ab, mb, rs, as, ms, rb, __MINLOC);
  }
}

void ENTFTN(MINLOCS, minlocs)(__INT_T *rb, char *ab, char *mb, F90_Desc *rs,
                              F90_Desc *as, F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);

  minlocs_common(&z, rb, ab, mb, rs, as, ms);
}

void ENTFTN(MINLOCS_B, minlocs_b)(__INT_T *rb, char *ab, char *mb,
                                  __INT_T *back, F90_Desc *rs, F90_Desc *as,
                                  F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);

  z.back = *(__LOG_T *)back;
  minlocs_common(&z, rb, ab, mb, rs, as, ms);
}

/* dim present */
static void minloc_common(red_parm *z, char *rb, char *ab, char *mb, char *db,
                          F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                          F90_Desc *ds)
{
  __fort_red_what = "MINLOC";

  z->kind = F90_KIND_G(as);
  z->len = F90_LEN_G(as);
  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z->mask_present) {
    z->lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z->lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z->l_fn_b = l_minloc_b[z->lk_shift][z->kind];
  z->g_fn = g_minloc[z->kind];
  z->zb = GET_DIST_MAXS(z->kind);

  if (z->kind == __STR)
    memset(rb, *((char *)(z->zb)), z->len);
  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(z, rb, ab, mb, db, rs, as, ms2, ds, __MINLOC);
    __fort_gfree(mb);
  } else {
    I8(__fort_red_arraylk)(z, rb, ab, mb, db, rs, as, ms, ds, __MINLOC);
  }
}

void ENTFTN(MINLOC, minloc)(char *rb, char *ab, char *mb, char *db,
                            F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                            F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  minloc_common(&z, rb, ab, mb, db, rs, as, ms, ds);
}

void ENTFTN(MINLOC_B, minloc_b)(char *rb, char *ab, char *mb, char *db,
                                __INT_T *back, F90_Desc *rs, F90_Desc *as,
                                F90_Desc *ms, F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  z.back = *(__LOG_T *)back;
  minloc_common(&z, rb, ab, mb, db, rs, as, ms, ds);
}

/* dim absent */
static void kminlocs_common(red_parm *z, __INT8_T *rb, char *ab, char *mb,
                            F90_Desc *rs, F90_Desc *as, F90_Desc *ms)
{
  double vb[4];
  char *strvb;

  __fort_red_what = "MINLOC";

  z->kind = F90_KIND_G(as);
  z->len = F90_LEN_G(as);
  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z->mask_present) {
    z->lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z->lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z->l_fn_b = l_kminloc_b[z->lk_shift][z->kind];
  z->g_fn = g_kminloc[z->kind];
  z->zb = GET_DIST_MAXS(z->kind);

  if (z->kind == __STR) {
    strvb = (char *)__fort_gmalloc(z->len);
    memset(strvb, *((char *)z->zb), z->len);
    I8(__fort_kred_scalarlk)(z, strvb, ab, mb, rs, as, ms, rb, __MINLOC);
    __fort_gfree(strvb);
  } else {
    I8(__fort_kred_scalarlk)(z, (char *)vb, ab, mb, rs, as, ms, rb, __MINLOC);
  }
}

void ENTFTN(KMINLOCS, kminlocs)(__INT8_T *rb, char *ab, char *mb, F90_Desc *rs,
                                F90_Desc *as, F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  kminlocs_common(&z, rb, ab, mb, rs, as, ms);
}

void ENTFTN(KMINLOCS_B, kminlocs_b)(__INT8_T *rb, char *ab, char *mb,
                                __INT8_T *back, F90_Desc *rs, F90_Desc *as,
                                F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  z.back = *(__LOG_T *)back;
  kminlocs_common(&z, rb, ab, mb, rs, as, ms);
}

/* dim present */
static void kminloc_common(red_parm *z, char *rb, char *ab, char *mb, char *db,
                           F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                           F90_Desc *ds)
{
  __fort_red_what = "MINLOC";

  z->kind = F90_KIND_G(as);
  z->len = F90_LEN_G(as);
  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z->mask_present) {
    z->lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z->lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z->l_fn_b = l_kminloc_b[z->lk_shift][z->kind];
  z->g_fn = g_kminloc[z->kind];
  z->zb = GET_DIST_MAXS(z->kind);

  if (z->kind == __STR)
    memset(rb, *((char *)z->zb), z->len);
  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(z, rb, ab, mb, db, rs, as, ms2, ds, __MINLOC);
    __fort_gfree(mb);
  } else {
    I8(__fort_kred_arraylk)(z, rb, ab, mb, db, rs, as, ms, ds, __MINLOC);
  }
}

void ENTFTN(KMINLOC, kminloc)(char *rb, char *ab, char *mb, char *db,
                              F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                              F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  kminloc_common(&z, rb, ab, mb, db, rs, as, ms, ds);
}

void ENTFTN(KMINLOC_B, kminloc_b)(char *rb, char *ab, char *mb, char *db,
                                  __INT8_T *back, F90_Desc *rs, F90_Desc *as,
                                  F90_Desc *ms, F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  z.back = *(__LOG_T *)back;
  kminloc_common(&z, rb, ab, mb, db, rs, as, ms, ds);
}
