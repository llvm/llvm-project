/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* red_maxloc.c -- intrinsic reduction function */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

MLOCFNLKN(>, maxloc_int1, __INT1_T, 1)
MLOCFNLKN(>, maxloc_int2, __INT2_T, 1)
MLOCFNLKN(>, maxloc_int4, __INT4_T, 1)
MLOCFNLKN(>, maxloc_int8, __INT8_T, 1)
MLOCFNLKN(>, maxloc_real4, __REAL4_T, 1)
MLOCFNLKN(>, maxloc_real8, __REAL8_T, 1)
MLOCFNLKN(>, maxloc_real16, __REAL16_T, 1)
MLOCSTRFNLKN(>, maxloc_str, __STR_T, 1)

MLOCFNLKN(>, maxloc_int1, __INT1_T, 2)
MLOCFNLKN(>, maxloc_int2, __INT2_T, 2)
MLOCFNLKN(>, maxloc_int4, __INT4_T, 2)
MLOCFNLKN(>, maxloc_int8, __INT8_T, 2)
MLOCFNLKN(>, maxloc_real4, __REAL4_T, 2)
MLOCFNLKN(>, maxloc_real8, __REAL8_T, 2)
MLOCFNLKN(>, maxloc_real16, __REAL16_T, 2)
MLOCSTRFNLKN(>, maxloc_str, __STR_T, 2)

MLOCFNLKN(>, maxloc_int1, __INT1_T, 4)
MLOCFNLKN(>, maxloc_int2, __INT2_T, 4)
MLOCFNLKN(>, maxloc_int4, __INT4_T, 4)
MLOCFNLKN(>, maxloc_int8, __INT8_T, 4)
MLOCFNLKN(>, maxloc_real4, __REAL4_T, 4)
MLOCFNLKN(>, maxloc_real8, __REAL8_T, 4)
MLOCFNLKN(>, maxloc_real16, __REAL16_T, 4)
MLOCSTRFNLKN(>, maxloc_str, __STR_T, 4)

MLOCFNLKN(>, maxloc_int1, __INT1_T, 8)
MLOCFNLKN(>, maxloc_int2, __INT2_T, 8)
MLOCFNLKN(>, maxloc_int4, __INT4_T, 8)
MLOCFNLKN(>, maxloc_int8, __INT8_T, 8)
MLOCFNLKN(>, maxloc_real4, __REAL4_T, 8)
MLOCFNLKN(>, maxloc_real8, __REAL8_T, 8)
MLOCFNLKN(>, maxloc_real16, __REAL16_T, 8)
MLOCSTRFNLKN(>, maxloc_str, __STR_T, 8)

MLOCFNG(>, maxloc_int1, __INT1_T)
MLOCFNG(>, maxloc_int2, __INT2_T)
MLOCFNG(>, maxloc_int4, __INT4_T)
MLOCFNG(>, maxloc_int8, __INT8_T)
MLOCFNG(>, maxloc_real4, __REAL4_T)
MLOCFNG(>, maxloc_real8, __REAL8_T)
MLOCFNG(>, maxloc_real16, __REAL16_T)
MLOCSTRFNG(>, maxloc_str, __STR_T)

static void (*l_maxloc_b[4][__NTYPES])() = TYPELIST3LK(l_maxloc_);
static void (*g_maxloc[__NTYPES])() = TYPELIST3(g_maxloc_);

KMLOCFNLKN(>, kmaxloc_int1, __INT1_T, 1)
KMLOCFNLKN(>, kmaxloc_int2, __INT2_T, 1)
KMLOCFNLKN(>, kmaxloc_int4, __INT4_T, 1)
KMLOCFNLKN(>, kmaxloc_int8, __INT8_T, 1)
KMLOCFNLKN(>, kmaxloc_real4, __REAL4_T, 1)
KMLOCFNLKN(>, kmaxloc_real8, __REAL8_T, 1)
KMLOCFNLKN(>, kmaxloc_real16, __REAL16_T, 1)
KMLOCSTRFNLKN(>, kmaxloc_str, __STR_T, 1)

KMLOCFNLKN(>, kmaxloc_int1, __INT1_T, 2)
KMLOCFNLKN(>, kmaxloc_int2, __INT2_T, 2)
KMLOCFNLKN(>, kmaxloc_int4, __INT4_T, 2)
KMLOCFNLKN(>, kmaxloc_int8, __INT8_T, 2)
KMLOCFNLKN(>, kmaxloc_real4, __REAL4_T, 2)
KMLOCFNLKN(>, kmaxloc_real8, __REAL8_T, 2)
KMLOCFNLKN(>, kmaxloc_real16, __REAL16_T, 2)
KMLOCSTRFNLKN(>, kmaxloc_str, __STR_T, 2)

KMLOCFNLKN(>, kmaxloc_int1, __INT1_T, 4)
KMLOCFNLKN(>, kmaxloc_int2, __INT2_T, 4)
KMLOCFNLKN(>, kmaxloc_int4, __INT4_T, 4)
KMLOCFNLKN(>, kmaxloc_int8, __INT8_T, 4)
KMLOCFNLKN(>, kmaxloc_real4, __REAL4_T, 4)
KMLOCFNLKN(>, kmaxloc_real8, __REAL8_T, 4)
KMLOCFNLKN(>, kmaxloc_real16, __REAL16_T, 4)
KMLOCSTRFNLKN(>, kmaxloc_str, __STR_T, 4)

KMLOCFNLKN(>, kmaxloc_int1, __INT1_T, 8)
KMLOCFNLKN(>, kmaxloc_int2, __INT2_T, 8)
KMLOCFNLKN(>, kmaxloc_int4, __INT4_T, 8)
KMLOCFNLKN(>, kmaxloc_int8, __INT8_T, 8)
KMLOCFNLKN(>, kmaxloc_real4, __REAL4_T, 8)
KMLOCFNLKN(>, kmaxloc_real8, __REAL8_T, 8)
KMLOCFNLKN(>, kmaxloc_real16, __REAL16_T, 8)
KMLOCSTRFNLKN(>, kmaxloc_str, __STR_T, 8)

KMLOCFNG(>, kmaxloc_int1, __INT1_T)
KMLOCFNG(>, kmaxloc_int2, __INT2_T)
KMLOCFNG(>, kmaxloc_int4, __INT4_T)
KMLOCFNG(>, kmaxloc_int8, __INT8_T)
KMLOCFNG(>, kmaxloc_real4, __REAL4_T)
KMLOCFNG(>, kmaxloc_real8, __REAL8_T)
KMLOCFNG(>, kmaxloc_real16, __REAL16_T)
KMLOCSTRFNG(>, kmaxloc_str, __STR_T)

static void (*l_kmaxloc_b[4][__NTYPES])() = TYPELIST3LK(l_kmaxloc_);
static void (*g_kmaxloc[__NTYPES])() = TYPELIST3(g_kmaxloc_);

/* dim absent */
/* Shared code for MAXLOC with and without BACK for backward compatibility */
static void
maxlocs_common(red_parm *z, __INT_T *rb, char *ab, char *mb, F90_Desc *rs,
                F90_Desc *as, F90_Desc *ms)
{
  double vb[4];
  char *strvb;

  __fort_red_what = "MAXLOC";
  z->kind = F90_KIND_G(as);
  z->len = F90_LEN_G(as);
  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z->mask_present) {
    z->lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z->lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z->l_fn_b = l_maxloc_b[z->lk_shift][z->kind];
  z->g_fn = g_maxloc[z->kind];
  z->zb = GET_DIST_MINS(z->kind);

  if (z->kind == __STR) {
    strvb = (char *)__fort_gmalloc(z->len);
    memset(strvb, *((char *)z->zb), z->len);
    I8(__fort_red_scalarlk)(z, strvb, ab, mb, rs, as, ms, rb, __MAXLOC);
    __fort_gfree(strvb);
  } else {
    I8(__fort_red_scalarlk)(z, (char *)vb, ab, mb, rs, as, ms, rb, __MAXLOC);
  }
}

/* MAXLOC without BACK */
void ENTFTN(MAXLOCS, maxlocs)(__INT_T *rb, char *ab, char *mb, F90_Desc *rs,
                              F90_Desc *as, F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);

  maxlocs_common(&z, rb, ab, mb, rs, as, ms);
}

/* MAXLOC with BACK */
void ENTFTN(MAXLOCS_B, maxlocs_b)(__INT_T *rb, char *ab, char *mb,
                                  __INT_T *back, F90_Desc *rs, F90_Desc *as,
                                  F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);

  z.back = *(__LOG_T *)back;
  maxlocs_common(&z, rb, ab, mb, rs, as, ms);
}

/* dim present */
static void maxloc_common(red_parm *z, char *rb, char *ab, char *mb, char *db,
                           F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                           F90_Desc *ds)
{
  __fort_red_what = "MAXLOC";
  z->kind = F90_KIND_G(as);
  z->len = F90_LEN_G(as);
  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z->mask_present) {
    z->lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z->lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z->l_fn_b = l_maxloc_b[z->lk_shift][z->kind];
  z->g_fn = g_maxloc[z->kind];
  z->zb = GET_DIST_MINS(z->kind);
  if (z->kind == __STR)
    memset(rb, *((char *)z->zb), z->len);
  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(z, rb, ab, mb, db, rs, as, ms2, ds, __MAXLOC);
    __fort_gfree(mb);
  } else {
    I8(__fort_red_arraylk)(z, rb, ab, mb, db, rs, as, ms, ds, __MAXLOC);
  }
}

void ENTFTN(MAXLOC, maxloc)(char *rb, char *ab, char *mb, char *db,
                            F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                            F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  maxloc_common(&z, rb, ab, mb, db, rs, as, ms, ds);
}

void ENTFTN(MAXLOC_B, maxloc_b)(char *rb, char *ab, char *mb, char *db,
                            __INT_T *back, F90_Desc *rs, F90_Desc *as,
                            F90_Desc *ms, F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  z.back = *(__LOG_T *)back;
  maxloc_common(&z, rb, ab, mb, db, rs, as, ms, ds);
}

/* dim absent */
static void kmaxlocs_common(red_parm *z, __INT8_T *rb, char *ab, char *mb,
                            F90_Desc *rs, F90_Desc *as, F90_Desc *ms)
{
  double vb[4];
  char *strvb;

  z->kind = F90_KIND_G(as);
  z->len = F90_LEN_G(as);
  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z->mask_present) {
    z->lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z->lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z->l_fn_b = l_kmaxloc_b[z->lk_shift][z->kind];
  z->g_fn = g_kmaxloc[z->kind];
  z->zb = GET_DIST_MINS(z->kind);

  if (z->kind == __STR) {
    strvb = (char *)__fort_gmalloc(z->len);
    memset(strvb, *((char *)z->zb), z->len);
    I8(__fort_kred_scalarlk)(z, strvb, ab, mb, rs, as, ms, rb, __MAXLOC);
    __fort_gfree(strvb);
  } else {
    I8(__fort_kred_scalarlk)(z, (char *)vb, ab, mb, rs, as, ms, rb, __MAXLOC);
  }
}

void ENTFTN(KMAXLOCS, kmaxlocs)(__INT8_T *rb, char *ab, char *mb, F90_Desc *rs,
                                F90_Desc *as, F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "MAXLOC";

  kmaxlocs_common(&z, rb, ab, mb, rs, as, ms);
}

void ENTFTN(KMAXLOCS_B, kmaxlocs_b)(__INT8_T *rb, char *ab, char *mb,
                                __INT8_T *back, F90_Desc *rs, F90_Desc *as,
                                F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "MAXLOC";

  z.back = *(__LOG_T *)back;
  kmaxlocs_common(&z, rb, ab, mb, rs, as, ms);
}

/* dim present */
static void kmaxloc_common(red_parm *z, char *rb, char *ab, char *mb, char *db,
                           F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                           F90_Desc *ds)
{
  __fort_red_what = "MAXLOC";

  z->kind = F90_KIND_G(as);
  z->len = F90_LEN_G(as);
  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z->mask_present) {
    z->lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z->lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z->l_fn_b = l_kmaxloc_b[z->lk_shift][z->kind];
  z->g_fn = g_kmaxloc[z->kind];
  z->zb = GET_DIST_MINS(z->kind);

  if (z->kind == __STR)
    memset(rb, *((char *)z->zb), z->len);
  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(z, rb, ab, mb, db, rs, as, ms2, ds, __MAXLOC);
    __fort_gfree(mb);
  } else {
    I8(__fort_kred_arraylk)(z, rb, ab, mb, db, rs, as, ms, ds, __MAXLOC);
  }
}

void ENTFTN(KMAXLOC, kmaxloc)(char *rb, char *ab, char *mb, char *db,
                              F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                              F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);

  kmaxloc_common(&z, rb, ab, mb, db, rs, as, ms, ds);
}

void ENTFTN(KMAXLOC_B, kmaxloc_b)(char *rb, char *ab, char *mb, char *db,
                              __INT8_T *back, F90_Desc *rs, F90_Desc *as,
                              F90_Desc *ms, F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);

  z.back = *(__LOG_T *)back;
  kmaxloc_common(&z, rb, ab, mb, db, rs, as, ms, ds);
}
