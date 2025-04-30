/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* red_findloc.c -- intrinsic reduction function */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

FLOCFNLKN(==, findloc_int1, __INT1_T, 1)
FLOCFNLKN(==, findloc_int2, __INT2_T, 1)
FLOCFNLKN(==, findloc_int4, __INT4_T, 1)
FLOCFNLKN(==, findloc_int8, __INT8_T, 1)
FLOCFNLKN(==, findloc_real4, __REAL4_T, 1)
FLOCFNLKN(==, findloc_real8, __REAL8_T, 1)
FLOCFNLKN(==, findloc_real16, __REAL16_T, 1)
FLOCSTRFNLKN(==, findloc_str, __STR_T, 1)

FLOCFNLKN(==, findloc_int1, __INT1_T, 2)
FLOCFNLKN(==, findloc_int2, __INT2_T, 2)
FLOCFNLKN(==, findloc_int4, __INT4_T, 2)
FLOCFNLKN(==, findloc_int8, __INT8_T, 2)
FLOCFNLKN(==, findloc_real4, __REAL4_T, 2)
FLOCFNLKN(==, findloc_real8, __REAL8_T, 2)
FLOCFNLKN(==, findloc_real16, __REAL16_T, 2)
FLOCSTRFNLKN(==, findloc_str, __STR_T, 2)

FLOCFNLKN(==, findloc_int1, __INT1_T, 4)
FLOCFNLKN(==, findloc_int2, __INT2_T, 4)
FLOCFNLKN(==, findloc_int4, __INT4_T, 4)
FLOCFNLKN(==, findloc_int8, __INT8_T, 4)
FLOCFNLKN(==, findloc_real4, __REAL4_T, 4)
FLOCFNLKN(==, findloc_real8, __REAL8_T, 4)
FLOCFNLKN(==, findloc_real16, __REAL16_T, 4)
FLOCSTRFNLKN(==, findloc_str, __STR_T, 4)

FLOCFNLKN(==, findloc_int1, __INT1_T, 8)
FLOCFNLKN(==, findloc_int2, __INT2_T, 8)
FLOCFNLKN(==, findloc_int4, __INT4_T, 8)
FLOCFNLKN(==, findloc_int8, __INT8_T, 8)
FLOCFNLKN(==, findloc_real4, __REAL4_T, 8)
FLOCFNLKN(==, findloc_real8, __REAL8_T, 8)
FLOCFNLKN(==, findloc_real16, __REAL16_T, 8)
FLOCSTRFNLKN(==, findloc_str, __STR_T, 8)

FLOCFNG(==, findloc_int1, __INT1_T)
FLOCFNG(==, findloc_int2, __INT2_T)
FLOCFNG(==, findloc_int4, __INT4_T)
FLOCFNG(==, findloc_int8, __INT8_T)
FLOCFNG(==, findloc_real4, __REAL4_T)
FLOCFNG(==, findloc_real8, __REAL8_T)
FLOCFNG(==, findloc_real16, __REAL16_T)
FLOCSTRFNG(==, findloc_str, __STR_T)

static void (*l_findloc_b[4][__NTYPES])() = TYPELIST3LK(l_findloc_);
static void (*g_findloc[__NTYPES])() = TYPELIST3(g_findloc_);

KFLOCFNLKN(==, kfindloc_int1, __INT1_T, 1)
KFLOCFNLKN(==, kfindloc_int2, __INT2_T, 1)
KFLOCFNLKN(==, kfindloc_int4, __INT4_T, 1)
KFLOCFNLKN(==, kfindloc_int8, __INT8_T, 1)
KFLOCFNLKN(==, kfindloc_real4, __REAL4_T, 1)
KFLOCFNLKN(==, kfindloc_real8, __REAL8_T, 1)
KFLOCFNLKN(==, kfindloc_real16, __REAL16_T, 1)
KFLOCSTRFNLKN(==, kfindloc_str, __STR_T, 1)

KFLOCFNLKN(==, kfindloc_int1, __INT1_T, 2)
KFLOCFNLKN(==, kfindloc_int2, __INT2_T, 2)
KFLOCFNLKN(==, kfindloc_int4, __INT4_T, 2)
KFLOCFNLKN(==, kfindloc_int8, __INT8_T, 2)
KFLOCFNLKN(==, kfindloc_real4, __REAL4_T, 2)
KFLOCFNLKN(==, kfindloc_real8, __REAL8_T, 2)
KFLOCFNLKN(==, kfindloc_real16, __REAL16_T, 2)
KFLOCSTRFNLKN(==, kfindloc_str, __STR_T, 2)

KFLOCFNLKN(==, kfindloc_int1, __INT1_T, 4)
KFLOCFNLKN(==, kfindloc_int2, __INT2_T, 4)
KFLOCFNLKN(==, kfindloc_int4, __INT4_T, 4)
KFLOCFNLKN(==, kfindloc_int8, __INT8_T, 4)
KFLOCFNLKN(==, kfindloc_real4, __REAL4_T, 4)
KFLOCFNLKN(==, kfindloc_real8, __REAL8_T, 4)
KFLOCFNLKN(==, kfindloc_real16, __REAL16_T, 4)
KFLOCSTRFNLKN(==, kfindloc_str, __STR_T, 4)

KFLOCFNLKN(==, kfindloc_int1, __INT1_T, 8)
KFLOCFNLKN(==, kfindloc_int2, __INT2_T, 8)
KFLOCFNLKN(==, kfindloc_int4, __INT4_T, 8)
KFLOCFNLKN(==, kfindloc_int8, __INT8_T, 8)
KFLOCFNLKN(==, kfindloc_real4, __REAL4_T, 8)
KFLOCFNLKN(==, kfindloc_real8, __REAL8_T, 8)
KFLOCFNLKN(==, kfindloc_real16, __REAL16_T, 8)
KFLOCSTRFNLKN(==, kfindloc_str, __STR_T, 8)

KFLOCFNG(==, kfindloc_int1, __INT1_T)
KFLOCFNG(==, kfindloc_int2, __INT2_T)
KFLOCFNG(==, kfindloc_int4, __INT4_T)
KFLOCFNG(==, kfindloc_int8, __INT8_T)
KFLOCFNG(==, kfindloc_real4, __REAL4_T)
KFLOCFNG(==, kfindloc_real8, __REAL8_T)
KFLOCFNG(==, kfindloc_real16, __REAL16_T)
KFLOCSTRFNG(==, kfindloc_str, __STR_T)

static void (*l_kfindloc[4][__NTYPES])() = TYPELIST3LK(l_kfindloc_);
static void (*g_kfindloc[__NTYPES])() = TYPELIST3(g_kfindloc_);

/* dim absent */

void ENTFTN(FINDLOCS, findlocs)(__INT_T *rb, char *ab, char *val, char *mb,
                                __INT_T *back, F90_Desc *rs, F90_Desc *as,
                                F90_Desc *vs, F90_Desc *ms, F90_Desc *bs)
{
  red_parm z;
  double vb[4];
  char *strvb;

  INIT_RED_PARM(z);
  __fort_red_what = "FINDLOC";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn_b = l_findloc_b[z.lk_shift][z.kind];
  z.g_fn = g_findloc[z.kind];
  z.zb = val;
  z.back = *(__LOG_T *)back;

  if (z.kind == __STR) {
    strvb = (char *)__fort_gmalloc(z.len);
    memcpy(strvb, ((char *)z.zb), z.len);
    I8(__fort_red_scalarlk)(&z, strvb, ab, mb, rs, as, ms, rb, __FINDLOC);
    __fort_gfree(strvb);
  } else {
    I8(__fort_red_scalarlk)(&z, (char *)vb, ab, mb, rs, as, ms, rb, __FINDLOC);
  }
}

/* dim present */

void ENTFTN(FINDLOC, findloc)(char *rb, char *ab, char *val, char *mb, char *db,
                              __INT_T *back, F90_Desc *rs, F90_Desc *as,
                              F90_Desc *vs, F90_Desc *ms, F90_Desc *ds,
                              F90_Desc *bs)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "FINDLOC";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn_b = l_findloc_b[z.lk_shift][z.kind];
  z.g_fn = g_findloc[z.kind];
  z.zb = val;
  z.back = *(__LOG_T *)back;
  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(&z, rb, ab, mb, db, rs, as, ms2, ds, __FINDLOC);
    __fort_gfree(mb);
  } else {
    I8(__fort_red_arraylk)(&z, rb, ab, mb, db, rs, as, ms, ds, __FINDLOC);
  }
}

/* dim absent */

void ENTFTN(KFINDLOCS, kfindlocs)(__INT8_T *rb, char *ab, char *val, char *mb,
                                  __INT8_T *back, F90_Desc *rs, F90_Desc *as,
                                  F90_Desc *vs, F90_Desc *ms, F90_Desc *bs)
{
  red_parm z;
  double vb[4];
  char *strvb;

  INIT_RED_PARM(z);
  __fort_red_what = "FINDLOC";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn_b = l_kfindloc[z.lk_shift][z.kind];
  z.g_fn = g_kfindloc[z.kind];
  z.zb = val;
  z.back = *(__LOG_T *)back;

  if (z.kind == __STR) {
    strvb = (char *)__fort_gmalloc(z.len);
    memcpy(strvb, ((char *)z.zb), z.len);
    I8(__fort_kred_scalarlk)(&z, strvb, ab, mb, rs, as, ms, rb, __FINDLOC);
    __fort_gfree(strvb);
  } else {
    memcpy(vb, val, z.len);
    I8(__fort_kred_scalarlk)(&z, (char *)vb, ab, mb, rs, as, ms, rb, __FINDLOC);
  }
}

/* dim present */

void ENTFTN(KFINDLOC, kfindloc)(char *rb, char *ab, char *val, char *mb,
                                char *db, __INT8_T *back, F90_Desc *rs,
                                F90_Desc *as, F90_Desc *vs, F90_Desc *ms,
                                F90_Desc *ds, F90_Desc *bs)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "FINDLOC";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn_b = l_kfindloc[z.lk_shift][z.kind];
  z.g_fn = g_kfindloc[z.kind];
  z.zb = val;
  z.back = *(__LOG_T *)back;
  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(&z, rb, ab, mb, db, rs, as, ms2, ds, __FINDLOC);
    __fort_gfree(mb);
  } else {
    I8(__fort_kred_arraylk)(&z, rb, ab, mb, db, rs, as, ms, ds, __FINDLOC);
  }
}

static char *
__chk_str_val(char *s, __INT8_T slen, __INT8_T tlen)
{
  char *news = s;

  if (slen < tlen) {
    news = (char *)__fort_gmalloc(tlen);
    memset(news, ' ', tlen);
    memcpy(news, s, slen);
  }
  return news;
}

void ENTFTN(FINDLOCSTRS, findlocstrs)(__INT_T *rb, char *ab, char *val,
                                      __INT_T *vlen, char *mb, __INT_T *back,
                                      F90_Desc *rs, F90_Desc *as, F90_Desc *vs,
                                      F90_Desc *vls, F90_Desc *ms, F90_Desc *bs)
{
  char *newval = __chk_str_val(val, (__INT8_T)*vlen, as->len);

  ENTFTN(FINDLOCS, findlocs)(rb, ab, newval, mb, back, rs, as, vs, ms, bs);
}

void ENTFTN(FINDLOCSTR, findlocstr)(char *rb, char *ab, char *val,
                                    __INT_T *vlen, char *mb, char *db,
                                    __INT_T *back, F90_Desc *rs, F90_Desc *as,
                                    F90_Desc *vs, F90_Desc *vls, F90_Desc *ms,
                                    F90_Desc *ds, F90_Desc *bs)
{
  char *newval = __chk_str_val(val, (__INT8_T)*vlen, as->len);

  ENTFTN(FINDLOC,findloc) (rb, ab, newval, mb, db, back, rs,as, vs, ms,ds, bs);
}

void ENTFTN(KFINDLOCSTRS,
            kfindlocstrs)(__INT8_T *rb, char *ab, char *val, __INT8_T *vlen,
                          char *mb, __INT8_T *back, F90_Desc *rs, F90_Desc *as,
                          F90_Desc *vs, F90_Desc *vls, F90_Desc *ms,
                          F90_Desc *bs)
{
  char *newval = __chk_str_val(val, (__INT8_T)*vlen, as->len);

  ENTFTN(KFINDLOCS, kfindlocs)(rb, ab, newval, mb, back, rs, as, vs, ms, bs);
}

void ENTFTN(KFINDLOCSTR, kfindlocstr)(char *rb, char *ab, char *val,
                                      __INT8_T *vlen, char *mb, char *db,
                                      __INT8_T *back, F90_Desc *rs,
                                      F90_Desc *as, F90_Desc *vs, F90_Desc *vls,
                                      F90_Desc *ms, F90_Desc *ds, F90_Desc *bs)
{
  char *newval = __chk_str_val(val, (__INT8_T)*vlen, as->len);

  ENTFTN(KFINDLOC,kfindloc) (rb, ab, newval, mb, db, back, rs,as, vs, ms,ds, bs);
}
