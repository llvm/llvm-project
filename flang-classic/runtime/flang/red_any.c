/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* red_any.c -- intrinsic reduction function */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

static __INT_T mask_desc = __LOG; /* scalar mask descriptor */

LOGFNG(|, any_log1, __LOG1_T)
LOGFNG(|, any_log2, __LOG2_T)
LOGFNG(|, any_log4, __LOG4_T)
LOGFNG(|, any_log8, __LOG8_T)
LOGFNG(|, any_int1, __INT1_T)
LOGFNG(|, any_int2, __INT2_T)
LOGFNG(|, any_int4, __INT4_T)
LOGFNG(|, any_int8, __INT8_T)

LOGFNLKN(|, any_log1, __LOG1_T, 1)
LOGFNLKN(|, any_log2, __LOG2_T, 1)
LOGFNLKN(|, any_log4, __LOG4_T, 1)
LOGFNLKN(|, any_log8, __LOG8_T, 1)
LOGFNLKN(|, any_int1, __INT1_T, 1)
LOGFNLKN(|, any_int2, __INT2_T, 1)
LOGFNLKN(|, any_int4, __INT4_T, 1)
LOGFNLKN(|, any_int8, __INT8_T, 1)

LOGFNLKN(|, any_log1, __LOG1_T, 2)
LOGFNLKN(|, any_log2, __LOG2_T, 2)
LOGFNLKN(|, any_log4, __LOG4_T, 2)
LOGFNLKN(|, any_log8, __LOG8_T, 2)
LOGFNLKN(|, any_int1, __INT1_T, 2)
LOGFNLKN(|, any_int2, __INT2_T, 2)
LOGFNLKN(|, any_int4, __INT4_T, 2)
LOGFNLKN(|, any_int8, __INT8_T, 2)

LOGFNLKN(|, any_log1, __LOG1_T, 4)
LOGFNLKN(|, any_log2, __LOG2_T, 4)
LOGFNLKN(|, any_log4, __LOG4_T, 4)
LOGFNLKN(|, any_log8, __LOG8_T, 4)
LOGFNLKN(|, any_int1, __INT1_T, 4)
LOGFNLKN(|, any_int2, __INT2_T, 4)
LOGFNLKN(|, any_int4, __INT4_T, 4)
LOGFNLKN(|, any_int8, __INT8_T, 4)

LOGFNLKN(|, any_log1, __LOG1_T, 8)
LOGFNLKN(|, any_log2, __LOG2_T, 8)
LOGFNLKN(|, any_log4, __LOG4_T, 8)
LOGFNLKN(|, any_log8, __LOG8_T, 8)
LOGFNLKN(|, any_int1, __INT1_T, 8)
LOGFNLKN(|, any_int2, __INT2_T, 8)
LOGFNLKN(|, any_int4, __INT4_T, 8)
LOGFNLKN(|, any_int8, __INT8_T, 8)

static void (*l_any[4][__NTYPES])() = TYPELIST2LK(l_any_);
static void (*g_any[__NTYPES])() = TYPELIST2(g_any_);

/* dim absent */

void ENTFTN(ANYS, anys)(char *rb, char *mb, F90_Desc *rs, F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "ANY";

  z.kind = F90_KIND_G(ms);
  z.len = F90_LEN_G(ms);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_any[z.lk_shift][z.kind];
  z.g_fn = g_any[z.kind];
  z.zb = GET_DIST_ZED;
  I8(__fort_red_scalar)(&z, rb, mb, (char *)GET_DIST_TRUE_LOG_ADDR,
		         rs, ms, (F90_Desc *)&mask_desc, NULL, __ANY);
}

/* dim present */

void ENTFTN(ANY, any)(char *rb, char *mb, char *db, F90_Desc *rs, F90_Desc *ms,
                      F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "ANY";

  z.kind = F90_KIND_G(ms);
  z.len = F90_LEN_G(ms);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_any[z.lk_shift][z.kind];
  z.g_fn = g_any[z.kind];
  z.zb = GET_DIST_ZED;
  I8(__fort_red_array)(&z, rb, mb, (char *)GET_DIST_TRUE_LOG_ADDR, db,
		        rs, ms, (F90_Desc *)&mask_desc, ds, __ANY);
}

/* global ANY accumulation */

void ENTFTN(REDUCE_ANY, reduce_any)(char *hb, __INT_T *dimsb, __INT_T *nargb,
                                    char *rb, F90_Desc *hd, F90_Desc *dimsd,
                                    F90_Desc *nargd, F90_Desc *rd)

{
#if defined(DEBUG)
  if (dimsd == NULL || F90_TAG_G(dimsd) != __INT)
    __fort_abort("GLOBAL_ANY: invalid dims descriptor");
  if (nargd == NULL || F90_TAG_G(nargd) != __INT)
    __fort_abort("REDUCE_ANY: invalid arg count descriptor");
  if (*nargb != 1)
    __fort_abort("REDUCE_ANY: arg count not 1");
#endif
  I8(__fort_global_reduce)(rb, hb, *dimsb, rd, hd, "ANY", g_any);
}

void ENTFTN(GLOBAL_ANY, global_any)(char *rb, char *hb, __INT_T *dimsb,
                                    F90_Desc *rd, F90_Desc *hd, F90_Desc *dimsd)
{
  I8(__fort_global_reduce)(rb, hb, *dimsb, rd, hd, "ANY", g_any);
}
