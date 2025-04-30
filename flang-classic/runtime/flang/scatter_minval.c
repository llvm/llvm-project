/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "scatter.h"

static __INT_T _1 = 1;

static void
scatter_minval_int1(int n, __INT1_T *r, int *sv, __INT1_T *a)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[i] < r[sv[i]])
      r[sv[i]] = a[i];
}
static void
scatter_minval_int2(int n, __INT2_T *r, int *sv, __INT2_T *a)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[i] < r[sv[i]])
      r[sv[i]] = a[i];
}
static void
scatter_minval_int4(int n, __INT4_T *r, int *sv, __INT4_T *a)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[i] < r[sv[i]])
      r[sv[i]] = a[i];
}
static void
scatter_minval_int8(int n, __INT8_T *r, int *sv, __INT8_T *a)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[i] < r[sv[i]])
      r[sv[i]] = a[i];
}
static void
scatter_minval_real4(int n, __REAL4_T *r, int *sv, __REAL4_T *a)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[i] < r[sv[i]])
      r[sv[i]] = a[i];
}
static void
scatter_minval_real8(int n, __REAL8_T *r, int *sv, __REAL8_T *a)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[i] < r[sv[i]])
      r[sv[i]] = a[i];
}
static void
scatter_minval_real16(int n, __REAL16_T *r, int *sv, __REAL16_T *a)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[i] < r[sv[i]])
      r[sv[i]] = a[i];
}

static void (*scatter_minval[__NTYPES])() = {
    NULL,                  /*     no type (absent optional argument) */
    NULL,                  /* C   signed short */
    NULL,                  /* C   unsigned short */
    NULL,                  /* C   signed int */
    NULL,                  /* C   unsigned int */
    NULL,                  /* C   signed long int */
    NULL,                  /* C   unsigned long int */
    NULL,                  /* C   float */
    NULL,                  /* C   double */
    NULL,                  /*   F complex*8 (2x real*4) */
    NULL,                  /*   F complex*16 (2x real*8) */
    NULL,                  /* C   signed char */
    NULL,                  /* C   unsigned char */
    NULL,                  /* C   long double */
    NULL,                  /*   F character */
    NULL,                  /* C   long long */
    NULL,                  /* C   unsigned long long */
    NULL,                  /*   F logical*1 */
    NULL,                  /*   F logical*2 */
    NULL,                  /*   F logical*4 */
    NULL,                  /*   F logical*8 */
    NULL,                  /*   F typeless */
    NULL,                  /*   F double typeless */
    NULL,                  /*   F ncharacter - kanji */
    scatter_minval_int2,   /*   F integer*2 */
    scatter_minval_int4,   /*   F integer*4, integer */
    scatter_minval_int8,   /*   F integer*8 */
    scatter_minval_real4,  /*   F real*4, real */
    scatter_minval_real8,  /*   F real*8, double precision */
    scatter_minval_real16, /*   F real*16 */
    NULL,                  /*   F complex*32 (2x real*16) */
    NULL,                  /*   F quad typeless */
    scatter_minval_int1,   /*   F integer*1 */
    NULL                   /*   F derived type */
};

static void
gathscat_minval_int1(int n, __INT1_T *r, int *sv, __INT1_T *a, int *gv)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[gv[i]] < r[sv[i]])
      r[sv[i]] = a[gv[i]];
}
static void
gathscat_minval_int2(int n, __INT2_T *r, int *sv, __INT2_T *a, int *gv)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[gv[i]] < r[sv[i]])
      r[sv[i]] = a[gv[i]];
}
static void
gathscat_minval_int4(int n, __INT4_T *r, int *sv, __INT4_T *a, int *gv)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[gv[i]] < r[sv[i]])
      r[sv[i]] = a[gv[i]];
}
static void
gathscat_minval_int8(int n, __INT8_T *r, int *sv, __INT8_T *a, int *gv)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[gv[i]] < r[sv[i]])
      r[sv[i]] = a[gv[i]];
}
static void
gathscat_minval_real4(int n, __REAL4_T *r, int *sv, __REAL4_T *a, int *gv)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[gv[i]] < r[sv[i]])
      r[sv[i]] = a[gv[i]];
}
static void
gathscat_minval_real8(int n, __REAL8_T *r, int *sv, __REAL8_T *a, int *gv)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[gv[i]] < r[sv[i]])
      r[sv[i]] = a[gv[i]];
}
static void
gathscat_minval_real16(int n, __REAL16_T *r, int *sv, __REAL16_T *a, int *gv)
{
  int i;
  for (i = 0; i < n; ++i)
    if (a[gv[i]] < r[sv[i]])
      r[sv[i]] = a[gv[i]];
}

static void (*gathscat_minval[__NTYPES])() = {
    NULL,                   /*     no type (absent optional argument) */
    NULL,                   /* C   signed short */
    NULL,                   /* C   unsigned short */
    NULL,                   /* C   signed int */
    NULL,                   /* C   unsigned int */
    NULL,                   /* C   signed long int */
    NULL,                   /* C   unsigned long int */
    NULL,                   /* C   float */
    NULL,                   /* C   double */
    NULL,                   /*   F complex*8 (2x real*4) */
    NULL,                   /*   F complex*16 (2x real*8) */
    NULL,                   /* C   signed char */
    NULL,                   /* C   unsigned char */
    NULL,                   /* C   long double */
    NULL,                   /*   F character */
    NULL,                   /* C   long long */
    NULL,                   /* C   unsigned long long */
    NULL,                   /*   F logical*1 */
    NULL,                   /*   F logical*2 */
    NULL,                   /*   F logical*4 */
    NULL,                   /*   F logical*8 */
    NULL,                   /*   F typeless */
    NULL,                   /*   F double typeless */
    NULL,                   /*   F ncharacter - kanji */
    gathscat_minval_int2,   /*   F integer*2 */
    gathscat_minval_int4,   /*   F integer*4, integer */
    gathscat_minval_int8,   /*   F integer*8 */
    gathscat_minval_real4,  /*   F real*4, real */
    gathscat_minval_real8,  /*   F real*8, double precision */
    gathscat_minval_real16, /*   F real*16 */
    NULL,                   /*   F complex*32 (2x real*16) */
    NULL,                   /*   F quad typeless */
    gathscat_minval_int1,   /*   F integer*1 */
    NULL                    /*   F derived type */
};

void ENTFTN(MINVAL_SCATTER, minval_scatter)(char *rb, char *ab, char *bb,
                                            char *mb, F90_Desc *rd,
                                            F90_Desc *ad, F90_Desc *bd,
                                            F90_Desc *md, ...)
/* ... = int *xb1,F90_Desc *xd1, ... int *xbn,F90_Desc *xdn */
{
  gathscat_parm z;
  int i;
  va_list va;
  char *bp, *rp;
  chdr *ch;
  sked *sk;
  char *xp;
  DECL_HDR_PTRS(new_d)[MAXDIMS];
  void *new_xb[MAXDIMS];

  /* initialize new_d and new_xb */

  for (i = 0; i < MAXDIMS; ++i)
    new_d[i] = new_xb[i] = NULL;

  /* result is vectored, array is unvectored */

  z.vb = z.rb = rb;
  z.vd = z.rd = rd;
  z.ub = z.ab = ab;
  z.ud = z.ad = ad;
  z.mb = mb;
  z.md = md;
  z.indirect = ~(-1 << F90_RANK_G(rd));
  z.permuted = 0;

  z.what = "MINVAL_SCATTER";
  va_start(va, md);
  for (i = 0; i < F90_RANK_G(rd); ++i) {
    DECL_HDR_PTRS(td);
    __INT_T *ti;
    ti = z.dim[i].xb = va_arg(va, __INT_T *);
    td = z.dim[i].xd = va_arg(va, F90_Desc *);
    if (ISSCALAR(td)) {
      new_d[i] = __fort_malloc(SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(rd)));
      z.dim[i].xb = I8(__fort_create_conforming_index_array)(z.what, ab, ti, ad,
                                                            td, new_d[i]);
      z.dim[i].xd = new_d[i];

      if (F90_DIM_LBOUND_G(bd, i) != 1) {
        /* adjust index array */
        I8(__fort_adjust_index_array)(z.what,(char*)z.dim[i].xb,
                                             (char*)z.dim[i].xb,
                                              i,new_d[i],bd);
      }

    } else if (F90_DIM_LBOUND_G(bd, i) != 1) {

      /* make new index array with adjusted index */

      new_xb[i] = I8(__fort_adjust_index_array)(z.what, NULL,
                                               (char *)z.dim[i].xb, i, td, bd);
      z.dim[i].xb = new_xb[i];
    }
  }
  va_end(va);

  /*z.what = "MINVAL_SCATTER";*/
  z.dir = __SCATTER;
  z.xfer_request = __fort_sendl;
  z.xfer_respond = __fort_recvl;
  z.gathscatfn = gathscat_minval[F90_KIND_G(rd)];
  z.scatterfn = scatter_minval[F90_KIND_G(rd)];

  /* copy base to result (if it's different) */

  rp = rb + DIST_SCOFF_G(rd) * F90_LEN_G(rd);
  bp = bb + DIST_SCOFF_G(bd) * F90_LEN_G(bd);
  if (rp != bp || !I8(__fort_stored_alike)(rd, bd)) {
    ch = I8(__fort_copy)(rp, bp, rd, bd, NULL);
    __fort_doit(ch);
    __fort_frechn(ch);
  }

  sk = I8(__fort_gathscat)(&z);
  xp = ENTFTN(COMM_START, comm_start)(&sk, rb, rd, ab, ad);
  ENTFTN(COMM_FINISH, comm_finish)(&xp);
  ENTFTN(COMM_FREE, comm_free)(&_1, &sk);

  /* free any new descriptors and adjusted copies of xb */

  for (i = 0; i < F90_RANK_G(rd); ++i) {
    if (new_xb[i])
      __fort_gfree(new_xb[i]);
    if (new_d[i]) {
      __fort_free(new_d[i]);
      __fort_gfree(z.dim[i].xb);
    }
  }
}

void ENTFTN(MINVAL_SCATTERX, minval_scatterx)(char *rb, char *ab, char *bb,
                                              char *mb, F90_Desc *rd,
                                              F90_Desc *ad, F90_Desc *bd,
                                              F90_Desc *md, __INT_T *indirect,
                                              __INT_T *permuted, ...)
/* ... = { [ __INT_T *xb,F90_Desc *xd, ] [ __INT_T *xmap,] }* */
{
  gathscat_parm z;
  int i;
  va_list va;
  char *bp, *rp;
  chdr *ch;
  sked *sk;
  char *xp;

  /* result is vectored, array is unvectored */

  z.vb = z.rb = rb;
  z.vd = z.rd = rd;
  z.ub = z.ab = ab;
  z.ud = z.ad = ad;
  z.mb = mb;
  z.md = md;
  z.indirect = *indirect;
  z.permuted = *permuted;

  va_start(va, permuted);
  for (i = 0; i < rd->rank; ++i) {
    if (z.indirect >> i & 1) {
      z.dim[i].xb = va_arg(va, __INT_T *);
      z.dim[i].xd = va_arg(va, F90_Desc *);
    }
    if (z.permuted >> i & 1)
      z.dim[i].xmap = va_arg(va, __INT_T *);
  }
  va_end(va);

  /* set up for scatter */

  z.what = "MINVAL_SCATTER";
  z.dir = __SCATTER;
  z.xfer_request = __fort_sendl;
  z.xfer_respond = __fort_recvl;
  z.gathscatfn = gathscat_minval[F90_KIND_G(rd)];
  z.scatterfn = scatter_minval[F90_KIND_G(rd)];

  /* copy base to result (if it's different) */

  rp = rb + DIST_SCOFF_G(rd) * F90_LEN_G(rd);
  bp = bb + DIST_SCOFF_G(bd) * F90_LEN_G(bd);
  if (rp != bp || !I8(__fort_stored_alike)(rd, bd)) {
    ch = I8(__fort_copy)(rp, bp, rd, bd, NULL);
    __fort_doit(ch);
    __fort_frechn(ch);
  }

  sk = I8(__fort_gathscat)(&z);
  xp = ENTFTN(COMM_START, comm_start)(&sk, rb, rd, ab, ad);
  ENTFTN(COMM_FINISH, comm_finish)(&xp);
  ENTFTN(COMM_FREE, comm_free)(&_1, &sk);
}

/* Note: the comm_xxx_scatter routines do not have a separate base
   argument, so they only support the case where the result and the
   base are the same array. To support a base array different from the
   result, comm_start and comm_finish would need to be extended to
   pass the base argument. */

sked *ENTFTN(COMM_MINVAL_SCATTER,
             comm_minval_scatter)(char *rb, char *ab, char *mb, F90_Desc *rd,
                                  F90_Desc *ad, F90_Desc *md, __INT_T *indirect,
                                  __INT_T *permuted, ...)
/* ... = { [ __INT_T *xb,F90_Desc *xd, ] [ __INT_T *xmap, ] }* */
{
  gathscat_parm z;
  int i;
  va_list va;

  /* result is vectored, array is unvectored */

  z.vb = z.rb = rb;
  z.vd = z.rd = rd;
  z.ub = z.ab = ab;
  z.ud = z.ad = ad;
  z.mb = mb;
  z.md = md;
  z.indirect = *indirect;
  z.permuted = *permuted;

  va_start(va, permuted);
  for (i = 0; i < rd->rank; ++i) {
    if (z.indirect >> i & 1) {
      z.dim[i].xb = va_arg(va, __INT_T *);
      z.dim[i].xd = va_arg(va, F90_Desc *);
    }
    if (z.permuted >> i & 1)
      z.dim[i].xmap = va_arg(va, __INT_T *);
  }
  va_end(va);

  z.what = "MINVAL_SCATTER";
  z.dir = __SCATTER;
  z.xfer_request = __fort_sendl;
  z.xfer_respond = __fort_recvl;
  z.gathscatfn = gathscat_minval[F90_KIND_G(rd)];
  z.scatterfn = scatter_minval[F90_KIND_G(rd)];

  return I8(__fort_gathscat)(&z);
}

sked *ENTFTN(COMM_MINVAL_SCATTER_1,
             comm_minval_scatter_1)(char *rb, char *ab, char *mb, __INT_T *xb1,
                                    F90_Desc *rd, F90_Desc *ad, F90_Desc *md,
                                    F90_Desc *xd1)
{
  __INT_T indirect = 0x1;
  __INT_T permuted = 0x0;
  return ENTFTN(COMM_MINVAL_SCATTER, comm_minval_scatter)(
      rb, ab, mb, rd, ad, md, &indirect, &permuted, xb1, xd1);
}

sked *ENTFTN(COMM_MINVAL_SCATTER_2,
             comm_minval_scatter_2)(char *rb, char *ab, char *mb, __INT_T *xb1,
                                    __INT_T *xb2, F90_Desc *rd, F90_Desc *ad,
                                    F90_Desc *md, F90_Desc *xd1, F90_Desc *xd2)
{
  __INT_T indirect = 0x3;
  __INT_T permuted = 0x0;
  return ENTFTN(COMM_MINVAL_SCATTER, comm_minval_scatter)(
      rb, ab, mb, rd, ad, md, &indirect, &permuted, xb1, xd1, xb2, xd2);
}

sked *ENTFTN(COMM_MINVAL_SCATTER_3,
             comm_minval_scatter_3)(char *rb, char *ab, char *mb, __INT_T *xb1,
                                    __INT_T *xb2, __INT_T *xb3, F90_Desc *rd,
                                    F90_Desc *ad, F90_Desc *md, F90_Desc *xd1,
                                    F90_Desc *xd2, F90_Desc *xd3)
{
  __INT_T indirect = 0x7;
  __INT_T permuted = 0x0;
  return ENTFTN(COMM_MINVAL_SCATTER,
                comm_minval_scatter)(rb, ab, mb, rd, ad, md, &indirect,
                                     &permuted, xb1, xd1, xb2, xd2, xb3, xd3);
}

sked *ENTFTN(COMM_MINVAL_SCATTER_4,
             comm_minval_scatter_4)(char *rb, char *ab, char *mb, __INT_T *xb1,
                                    __INT_T *xb2, __INT_T *xb3, __INT_T *xb4,
                                    F90_Desc *rd, F90_Desc *ad, F90_Desc *md,
                                    F90_Desc *xd1, F90_Desc *xd2, F90_Desc *xd3,
                                    F90_Desc *xd4)
{
  __INT_T indirect = 0xf;
  __INT_T permuted = 0x0;
  return ENTFTN(COMM_MINVAL_SCATTER, comm_minval_scatter)(
      rb, ab, mb, rd, ad, md, &indirect, &permuted, xb1, xd1, xb2, xd2, xb3,
      xd3, xb4, xd4);
}

sked *ENTFTN(COMM_MINVAL_SCATTER_5,
             comm_minval_scatter_5)(char *rb, char *ab, char *mb, __INT_T *xb1,
                                    __INT_T *xb2, __INT_T *xb3, __INT_T *xb4,
                                    __INT_T *xb5, F90_Desc *rd, F90_Desc *ad,
                                    F90_Desc *md, F90_Desc *xd1, F90_Desc *xd2,
                                    F90_Desc *xd3, F90_Desc *xd4, F90_Desc *xd5)
{
  __INT_T indirect = 0x1f;
  __INT_T permuted = 0x0;
  return ENTFTN(COMM_MINVAL_SCATTER, comm_minval_scatter)(
      rb, ab, mb, rd, ad, md, &indirect, &permuted, xb1, xd1, xb2, xd2, xb3,
      xd3, xb4, xd4, xb5, xd5);
}

sked *ENTFTN(COMM_MINVAL_SCATTER_6,
             comm_minval_scatter_6)(char *rb, char *ab, char *mb, __INT_T *xb1,
                                    __INT_T *xb2, __INT_T *xb3, __INT_T *xb4,
                                    __INT_T *xb5, __INT_T *xb6, F90_Desc *rd,
                                    F90_Desc *ad, F90_Desc *md, F90_Desc *xd1,
                                    F90_Desc *xd2, F90_Desc *xd3, F90_Desc *xd4,
                                    F90_Desc *xd5, F90_Desc *xd6)
{
  __INT_T indirect = 0x3f;
  __INT_T permuted = 0x0;
  return ENTFTN(COMM_MINVAL_SCATTER, comm_minval_scatter)(
      rb, ab, mb, rd, ad, md, &indirect, &permuted, xb1, xd1, xb2, xd2, xb3,
      xd3, xb4, xd4, xb5, xd5, xb6, xd6);
}

sked *ENTFTN(COMM_MINVAL_SCATTER_7, comm_minval_scatter_7)(
    char *rb, char *ab, char *mb, __INT_T *xb1, __INT_T *xb2, __INT_T *xb3,
    __INT_T *xb4, __INT_T *xb5, __INT_T *xb6, __INT_T *xb7, F90_Desc *rd,
    F90_Desc *ad, F90_Desc *md, F90_Desc *xd1, F90_Desc *xd2, F90_Desc *xd3,
    F90_Desc *xd4, F90_Desc *xd5, F90_Desc *xd6, F90_Desc *xd7)
{
  __INT_T indirect = 0x7f;
  __INT_T permuted = 0x0;
  return ENTFTN(COMM_MINVAL_SCATTER, comm_minval_scatter)(
      rb, ab, mb, rd, ad, md, &indirect, &permuted, xb1, xd1, xb2, xd2, xb3,
      xd3, xb4, xd4, xb5, xd5, xb6, xd6, xb7, xd7);
}
