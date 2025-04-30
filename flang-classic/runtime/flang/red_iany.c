/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* red_iany.c -- reduction function */
/* FIXME: still used */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

ARITHFNG(|, iany_log1, __LOG1_T, __LOG1_T)
ARITHFNG(|, iany_log2, __LOG2_T, __LOG2_T)
ARITHFNG(|, iany_log4, __LOG4_T, __LOG4_T)
ARITHFNG(|, iany_log8, __LOG8_T, __LOG8_T)
ARITHFNG(|, iany_int1, __INT1_T, __INT1_T)
ARITHFNG(|, iany_int2, __INT2_T, __INT2_T)
ARITHFNG(|, iany_int4, __INT4_T, __INT4_T)
ARITHFNG(|, iany_int8, __INT8_T, __INT8_T)

ARITHFNLKN(|, iany_log1, __LOG1_T, __LOG1_T, 1)
ARITHFNLKN(|, iany_log2, __LOG2_T, __LOG2_T, 1)
ARITHFNLKN(|, iany_log4, __LOG4_T, __LOG4_T, 1)
ARITHFNLKN(|, iany_log8, __LOG8_T, __LOG8_T, 1)
ARITHFNLKN(|, iany_int1, __INT1_T, __INT1_T, 1)
ARITHFNLKN(|, iany_int2, __INT2_T, __INT2_T, 1)
ARITHFNLKN(|, iany_int4, __INT4_T, __INT4_T, 1)
ARITHFNLKN(|, iany_int8, __INT8_T, __INT8_T, 1)

ARITHFNLKN(|, iany_log1, __LOG1_T, __LOG1_T, 2)
ARITHFNLKN(|, iany_log2, __LOG2_T, __LOG2_T, 2)
ARITHFNLKN(|, iany_log4, __LOG4_T, __LOG4_T, 2)
ARITHFNLKN(|, iany_log8, __LOG8_T, __LOG8_T, 2)
ARITHFNLKN(|, iany_int1, __INT1_T, __INT1_T, 2)
ARITHFNLKN(|, iany_int2, __INT2_T, __INT2_T, 2)
ARITHFNLKN(|, iany_int4, __INT4_T, __INT4_T, 2)
ARITHFNLKN(|, iany_int8, __INT8_T, __INT8_T, 2)

ARITHFNLKN(|, iany_log1, __LOG1_T, __LOG1_T, 4)
ARITHFNLKN(|, iany_log2, __LOG2_T, __LOG2_T, 4)
ARITHFNLKN(|, iany_log4, __LOG4_T, __LOG4_T, 4)
ARITHFNLKN(|, iany_log8, __LOG8_T, __LOG8_T, 4)
ARITHFNLKN(|, iany_int1, __INT1_T, __INT1_T, 4)
ARITHFNLKN(|, iany_int2, __INT2_T, __INT2_T, 4)
ARITHFNLKN(|, iany_int4, __INT4_T, __INT4_T, 4)
ARITHFNLKN(|, iany_int8, __INT8_T, __INT8_T, 4)

ARITHFNLKN(|, iany_log1, __LOG1_T, __LOG1_T, 8)
ARITHFNLKN(|, iany_log2, __LOG2_T, __LOG2_T, 8)
ARITHFNLKN(|, iany_log4, __LOG4_T, __LOG4_T, 8)
ARITHFNLKN(|, iany_log8, __LOG8_T, __LOG8_T, 8)
ARITHFNLKN(|, iany_int1, __INT1_T, __INT1_T, 8)
ARITHFNLKN(|, iany_int2, __INT2_T, __INT2_T, 8)
ARITHFNLKN(|, iany_int4, __INT4_T, __INT4_T, 8)
ARITHFNLKN(|, iany_int8, __INT8_T, __INT8_T, 8)

static void (*l_iany[4][__NTYPES])() = TYPELIST2LK(l_iany_);
static void (*g_iany[__NTYPES])() = TYPELIST2(g_iany_);

/* dim absent */

void ENTFTN(IANYS, ianys)(char *rb, char *ab, char *mb, F90_Desc *rs,
                          F90_Desc *as, F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "IANY";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_iany[z.lk_shift][z.kind];
  z.g_fn = g_iany[z.kind];
  z.zb = GET_DIST_ZED;
  I8(__fort_red_scalar)(&z, rb, ab, mb, rs, as, ms, NULL, __IANY);
}

/* dim present */

void ENTFTN(IANY, iany)(char *rb, char *ab, char *mb, char *db, F90_Desc *rs,
                        F90_Desc *as, F90_Desc *ms, F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "IANY";

  z.kind = F90_KIND_G(as);
  z.len = F90_LEN_G(as);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_iany[z.lk_shift][z.kind];
  z.g_fn = g_iany[z.kind];
  z.zb = GET_DIST_ZED;
  if (ISSCALAR(ms)) {
    DECL_HDR_VARS(ms2);

    mb = (char *)I8(__fort_create_conforming_mask_array)(__fort_red_what, ab, mb,
                                                        as, ms, ms2);
    I8(__fort_red_array)(&z, rb, ab, mb, db, rs, as, ms2, ds, __IANY);
    __fort_gfree(mb);

  } else {
    I8(__fort_red_array)(&z, rb, ab, mb, db, rs, as, ms, ds, __IANY);
  }
}

/* global IANY accumulation */

void ENTFTN(REDUCE_IANY, reduce_iany)(char *hb, __INT_T *dimsb, __INT_T *nargb,
                                      char *rb, F90_Desc *hd, F90_Desc *dimsd,
                                      F90_Desc *nargd, F90_Desc *rd)

{
#if defined(DEBUG)
  if (dimsd == NULL || F90_TAG_G(dimsd) != __INT)
    __fort_abort("GLOBAL_IANY: invalid dims descriptor");
  if (nargd == NULL || F90_TAG_G(nargd) != __INT)
    __fort_abort("REDUCE_IANY: invalid arg count descriptor");
  if (*nargb != 1)
    __fort_abort("REDUCE_IANY: arg count not 1");
#endif
  I8(__fort_global_reduce)(rb, hb, *dimsb, rd, hd, "IANY", g_iany);
}

void ENTFTN(GLOBAL_IANY, global_iany)(char *rb, char *hb, __INT_T *dimsb,
                                      F90_Desc *rd, F90_Desc *hd,
                                      F90_Desc *dimsd)
{
  I8(__fort_global_reduce)(rb, hb, *dimsb, rd, hd, "IANY", g_iany);
}
