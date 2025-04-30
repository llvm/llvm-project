/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* red_count.c -- intrinsic reduction function */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

static __INT_T mask_desc = __LOG; /* scalar mask descriptor */

#define COUNTFN(NAME, RTYP)                                                    \
  static void l_##NAME(int *r, __INT_T n, RTYP *v, __INT_T vs, __LOG_T *m,     \
                       __INT_T ms, __INT_T *loc, __INT_T li, __INT_T ls)       \
  {                                                                            \
    __INT_T i;                                                                 \
    int x = *r;                                                                \
    __LOG_T mask_log = GET_DIST_MASK_LOG;                                     \
    for (i = 0; n > 0; n--, i += vs) {                                         \
      if (v[i] & mask_log) {                                                   \
        x++;                                                                   \
      }                                                                        \
    }                                                                          \
    *r = x;                                                                    \
  }

#define COUNTFNLKN(NAME, RTYP, N)                                              \
  static void l_##NAME##l##N(int *r, __INT_T n, RTYP *v, __INT_T vs,           \
                             __LOG##N##_T *m, __INT_T ms, __INT_T *loc,        \
                             __INT_T li, __INT_T ls)                           \
  {                                                                            \
    __INT_T i;                                                                 \
    int x = *r;                                                                \
    __LOG##N##_T mask_log = GET_DIST_MASK_LOG##N;                             \
    for (i = 0; n > 0; n--, i += vs) {                                         \
      if (v[i] & mask_log) {                                                   \
        x++;                                                                   \
      }                                                                        \
    }                                                                          \
    *r = x;                                                                    \
  }

COUNTFNLKN(count_log1, __LOG1_T, 1)
COUNTFNLKN(count_log2, __LOG2_T, 1)
COUNTFNLKN(count_log4, __LOG4_T, 1)
COUNTFNLKN(count_log8, __LOG8_T, 1)
COUNTFNLKN(count_int1, __INT1_T, 1)
COUNTFNLKN(count_int2, __INT2_T, 1)
COUNTFNLKN(count_int4, __INT4_T, 1)
COUNTFNLKN(count_int8, __INT8_T, 1)

COUNTFNLKN(count_log1, __LOG1_T, 2)
COUNTFNLKN(count_log2, __LOG2_T, 2)
COUNTFNLKN(count_log4, __LOG4_T, 2)
COUNTFNLKN(count_log8, __LOG8_T, 2)
COUNTFNLKN(count_int1, __INT1_T, 2)
COUNTFNLKN(count_int2, __INT2_T, 2)
COUNTFNLKN(count_int4, __INT4_T, 2)
COUNTFNLKN(count_int8, __INT8_T, 2)

COUNTFNLKN(count_log1, __LOG1_T, 4)
COUNTFNLKN(count_log2, __LOG2_T, 4)
COUNTFNLKN(count_log4, __LOG4_T, 4)
COUNTFNLKN(count_log8, __LOG8_T, 4)
COUNTFNLKN(count_int1, __INT1_T, 4)
COUNTFNLKN(count_int2, __INT2_T, 4)
COUNTFNLKN(count_int4, __INT4_T, 4)
COUNTFNLKN(count_int8, __INT8_T, 4)

COUNTFNLKN(count_log1, __LOG1_T, 8)
COUNTFNLKN(count_log2, __LOG2_T, 8)
COUNTFNLKN(count_log4, __LOG4_T, 8)
COUNTFNLKN(count_log8, __LOG8_T, 8)
COUNTFNLKN(count_int1, __INT1_T, 8)
COUNTFNLKN(count_int2, __INT2_T, 8)
COUNTFNLKN(count_int4, __INT4_T, 8)
COUNTFNLKN(count_int8, __INT8_T, 8)

static void (*l_count[4][__NTYPES])() = TYPELIST2LK(l_count_);

static void I8(g_count)(__INT_T n, __INT_T *lr, __INT_T *rr, void *lv, void *rv)
{
  __INT_T i;
  for (i = 0; i < n; i++) {
    lr[i] = lr[i] + rr[i];
  }
}

/* dim absent */

void ENTFTN(COUNTS, counts)(char *rb, char *mb, DECL_HDR_PTRS(rs), F90_Desc *ms)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "COUNT";

  z.kind = __INT;
  z.len = sizeof(__STAT_T);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_count[z.lk_shift][ms->kind];
  z.g_fn =
      (void (*)(__INT_T, void *, void *, void *, void *, __INT_T))I8(g_count);
  z.zb = GET_DIST_ZED;
  *(__INT_T *)rb = 0;
  I8(__fort_red_scalar)(&z, rb, mb, (char *)GET_DIST_TRUE_LOG_ADDR,
		         rs, ms, (F90_Desc *)&mask_desc, NULL, __COUNT);
}

/* dim present */

void ENTFTN(COUNT, count)(char *rb, char *mb, char *db, DECL_HDR_PTRS(rs),
                          F90_Desc *ms, F90_Desc *ds)
{
  red_parm z;

  INIT_RED_PARM(z);
  __fort_red_what = "COUNT";

  z.kind = __INT;
  z.len = sizeof(__STAT_T);
  z.mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (!z.mask_present) {
    z.lk_shift = GET_DIST_SHIFTS(__LOG);
  } else {
    z.lk_shift = GET_DIST_SHIFTS(F90_KIND_G(ms));
  }
  z.l_fn = l_count[z.lk_shift][ms->kind];
  z.g_fn =
      (void (*)(__INT_T, void *, void *, void *, void *, __INT_T))I8(g_count);
  z.zb = GET_DIST_ZED;
  I8(__fort_red_array)(&z, rb, mb, (char *)GET_DIST_TRUE_LOG_ADDR, db,
		        rs, ms, (F90_Desc *)&mask_desc, ds, __COUNT);
}
