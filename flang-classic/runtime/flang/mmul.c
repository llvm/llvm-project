/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* mmul.c -- DOT_PRODUCT and MATMUL intrinsics */

#include <memory.h>
#include "stdioInterf.h"
#include "fioMacros.h"
#include "fort_vars.h"

/* Declare pointer-type parameters in overloaded function pointer types
   as void *, then cast the function pointers during function selection. */
typedef void (*mmul_fn)(void *, int, void *, int, int, void *, int, int);

extern void (*I8(__fort_g_sum)[__NTYPES])();
extern void (*__fort_scalar_copy[__NTYPES])(void *rp, const void *sp, int len);

void ENTFTN(QOPY_IN, qopy_in)(char **dptr, __POINT_T *doff, char *dbase,
                              F90_Desc *dd, char *ab, F90_Desc *ad,
                              __INT_T *p_rank, __INT_T *p_kind, __INT_T *p_len,
                              __INT_T *p_flags, ...);

static void
dotp_real4(__REAL4_T *c, int nj, __REAL4_T *a, int ai, int ais, __REAL4_T *b,
           int bk, int bks)
{
  register double cc;
  cc = *c;
  for (; --nj >= 0; ai += ais, bk += bks)
    cc += a[ai] * b[bk];
  *c = cc;
}

static void
dotp_real8(__REAL8_T *c, int nj, __REAL8_T *a, int ai, int ais, __REAL8_T *b,
           int bk, int bks)
{
  register double cc;
  cc = *c;
  for (; --nj >= 0; ai += ais, bk += bks)
    cc += a[ai] * b[bk];
  *c = cc;
}

static void
dotp_cplx8(__CPLX8_T *c, int nj, __CPLX8_T *a, int ai, int ais, __CPLX8_T *b,
           int bk, int bks)
{
  register double cr, ci;
  cr = c->r;
  ci = c->i;
  for (; --nj >= 0; ai += ais, bk += bks) {
    cr += a[ai].r * b[bk].r + a[ai].i * b[bk].i;
    ci += a[ai].r * b[bk].i - a[ai].i * b[bk].r;
  }
  c->r = cr;
  c->i = ci;
}

static void
mmul_cplx8(__CPLX8_T *c, int nj, __CPLX8_T *a, int ai, int ais, __CPLX8_T *b,
           int bk, int bks)
{
  register double cr, ci;
  cr = c->r;
  ci = c->i;
  for (; --nj >= 0; ai += ais, bk += bks) {
    cr += a[ai].r * b[bk].r - a[ai].i * b[bk].i;
    ci += a[ai].r * b[bk].i + a[ai].i * b[bk].r;
  }
  c->r = cr;
  c->i = ci;
}

static void
dotp_cplx16(__CPLX16_T *c, int nj, __CPLX16_T *a, int ai, int ais,
            __CPLX16_T *b, int bk, int bks)
{
  register double cr, ci;
  cr = c->r;
  ci = c->i;
  for (; --nj >= 0; ai += ais, bk += bks) {
    cr += a[ai].r * b[bk].r + a[ai].i * b[bk].i;
    ci += a[ai].r * b[bk].i - a[ai].i * b[bk].r;
  }
  c->r = cr;
  c->i = ci;
}

static void
mmul_cplx16(__CPLX16_T *c, int nj, __CPLX16_T *a, int ai, int ais,
            __CPLX16_T *b, int bk, int bks)
{
  register double cr, ci;
  cr = c->r;
  ci = c->i;
  for (; --nj >= 0; ai += ais, bk += bks) {
    cr += a[ai].r * b[bk].r - a[ai].i * b[bk].i;
    ci += a[ai].r * b[bk].i + a[ai].i * b[bk].r;
  }
  c->r = cr;
  c->i = ci;
}

static void
dotp_int1(__INT1_T *c, int nj, __INT1_T *a, int ai, int ais, __INT1_T *b,
          int bk, int bks)
{
  register long cc;
  cc = *c;
  for (; --nj >= 0; ai += ais, bk += bks)
    cc += a[ai] * b[bk];
  *c = cc;
}

static void
dotp_int2(__INT2_T *c, int nj, __INT2_T *a, int ai, int ais, __INT2_T *b,
          int bk, int bks)
{
  register long cc;
  cc = *c;
  for (; --nj >= 0; ai += ais, bk += bks)
    cc += a[ai] * b[bk];
  *c = cc;
}

static void
dotp_int4(__INT4_T *c, int nj, __INT4_T *a, int ai, int ais, __INT4_T *b,
          int bk, int bks)
{
  register long cc;
  cc = *c;
  for (; --nj >= 0; ai += ais, bk += bks)
    cc += a[ai] * b[bk];
  *c = cc;
}

static void
dotp_log1(__LOG1_T *c, int nj, __LOG1_T *a, int ai, int ais, __LOG1_T *b,
          int bk, int bks)
{
  __LOG1_T mask_log1;

  mask_log1 = GET_DIST_MASK_LOG1;
  for (; --nj >= 0; ai += ais, bk += bks) {
    if ((a[ai] & mask_log1) && (b[bk] & mask_log1)) {
      *c = GET_DIST_TRUE_LOG1;
      break;
    }
  }
}

static void
dotp_log2(__LOG2_T *c, int nj, __LOG2_T *a, int ai, int ais, __LOG2_T *b,
          int bk, int bks)
{
  __LOG2_T mask_log2;

  mask_log2 = GET_DIST_MASK_LOG2;
  for (; --nj >= 0; ai += ais, bk += bks) {
    if ((a[ai] & mask_log2) && (b[bk] & mask_log2)) {
      *c = GET_DIST_TRUE_LOG2;
      break;
    }
  }
}

static void
dotp_log4(__LOG4_T *c, int nj, __LOG4_T *a, int ai, int ais, __LOG4_T *b,
          int bk, int bks)
{
  __LOG4_T mask_log4;

  mask_log4 = GET_DIST_MASK_LOG4;
  for (; --nj >= 0; ai += ais, bk += bks) {
    if ((a[ai] & mask_log4) && (b[bk] & mask_log4)) {
      *c = GET_DIST_TRUE_LOG4;
      break;
    }
  }
}

static void
dotp_int8(__INT8_T *c, int nj, __INT8_T *a, int ai, int ais, __INT8_T *b,
          int bk, int bks)
{
  register long cc;
  cc = *c;
  for (; --nj >= 0; ai += ais, bk += bks)
    cc += a[ai] * b[bk];
  *c = cc;
}

static void
dotp_log8(__LOG8_T *c, int nj, __LOG8_T *a, int ai, int ais, __LOG8_T *b,
          int bk, int bks)
{
  __LOG8_T mask_log8;

  mask_log8 = GET_DIST_MASK_LOG8;
  for (; --nj >= 0; ai += ais, bk += bks) {
    if ((a[ai] & mask_log8) && (b[bk] & mask_log8)) {
      *c = GET_DIST_TRUE_LOG8;
      break;
    }
  }
}

static __INT_T _0 = 0, _1 = 1, _2 = 2;

void ENTFTN(DOTPR, dotpr)(char *cb, char *ab0, char *bb0, F90_Desc *cs,
                          F90_Desc *as0, F90_Desc *bs0)
{
  char *ab = 0, *bb = 0;
  DECL_HDR_PTRS(as);
  DECL_HDR_PTRS(bs);
  DECL_HDR_VARS(as1);
  DECL_HDR_VARS(bs1);
  DECL_DIM_PTRS(ajd);
  DECL_DIM_PTRS(bjd);
  __INT_T flags, kind, len;
  mmul_fn dotp;
  __INT_T al, alof, an, aoff, astr, au, bl, blof, bn, boff, bstr, bu, acl, bcl,
      cn;
  int copy_required;

#if defined(DEBUG)
  if (__fort_test & DEBUG_MMUL) {
    printf("%d dotpr a", GET_DIST_LCPU);
    I8(__fort_show_section)(as0);
    printf(" x b");
    I8(__fort_show_section)(bs0);
    printf("\n");
  }
  if (F90_TAG_G(cs) == __DESC && F90_RANK_G(cs) != 0)
    __fort_abort("DOT_PRODUCT: result must be scalar");
#endif

  kind = F90_KIND_G(as0);
  len = F90_LEN_G(as0);

  copy_required = I8(is_nonsequential_section)(as0, F90_RANK_G(as0));
  if (copy_required) {
    as = as1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_IN + __INHERIT +
             __TRANSCRIPTIVE_DIST_TARGET + __TRANSCRIPTIVE_DIST_FORMAT);
    ENTFTN(QOPY_IN, qopy_in)
    (&ab, (__POINT_T *)ABSENT, ab0, as, ab0, as0, &_1, &kind, &len, &flags,
     &_1);
  } else {
    ab = ab0;
    as = as0;
  }

  copy_required = I8(is_nonsequential_section)(bs0, F90_RANK_G(bs0));
  if (copy_required)
  {
    bs = bs1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_IN +
             __PRESCRIPTIVE_ALIGN_TARGET + __IDENTITY_MAP);
    ENTFTN(QOPY_IN, qopy_in)
    (&bb, (__POINT_T *)ABSENT, bb0, bs, bb0, bs0, &_1, &kind, &len, &flags, as,
     &_1, &_1); /* conform, lb */
  } else {
    bb = bb0;
    bs = bs0;
  }

  switch (kind) {
  case __REAL4:
    dotp = (mmul_fn)dotp_real4;
    break;
  case __REAL8:
    dotp = (mmul_fn)dotp_real8;
    break;
  case __CPLX8:
    dotp = (mmul_fn)dotp_cplx8;
    break;
  case __CPLX16:
    dotp = (mmul_fn)dotp_cplx16;
    break;
  case __INT1:
    dotp = (mmul_fn)dotp_int1;
    break;
  case __INT2:
    dotp = (mmul_fn)dotp_int2;
    break;
  case __INT4:
    dotp = (mmul_fn)dotp_int4;
    break;
  case __LOG1:
    dotp = (mmul_fn)dotp_log1;
    break;
  case __LOG2:
    dotp = (mmul_fn)dotp_log2;
    break;
  case __LOG4:
    dotp = (mmul_fn)dotp_log4;
    break;
  case __INT8:
    dotp = (mmul_fn)dotp_int8;
    break;
  case __LOG8:
    dotp = (mmul_fn)dotp_log8;
    break;
  default:
    __fort_abort("DOT_PRODUCT: unimplemented for data type");
  }

  __fort_scalar_copy[kind](cb, GET_DIST_ZED, len);

#if defined(DEBUG)
  if ((F90_FLAGS_G(as) ^ F90_FLAGS_G(bs)) & __OFF_TEMPLATE)
    __fort_abort("DOT_PRODUCT: misaligned arguments");
#endif

  if (~(F90_FLAGS_G(as) | F90_FLAGS_G(bs)) & __OFF_TEMPLATE) {
    I8(__fort_cycle_bounds)(as);
    I8(__fort_cycle_bounds)(bs);

    SET_DIM_PTRS(ajd, as, 0);
    SET_DIM_PTRS(bjd, bs, 0);

    cn = DIST_DPTR_CN_G(ajd);
    acl = DIST_DPTR_CL_G(ajd);
    bcl = DIST_DPTR_CL_G(ajd);
    alof = DIST_DPTR_CLOF_G(ajd);
    blof = DIST_DPTR_CLOF_G(bjd);
    astr = F90_DPTR_SSTRIDE_G(ajd) * F90_DPTR_LSTRIDE_G(ajd);
    bstr = F90_DPTR_SSTRIDE_G(bjd) * F90_DPTR_LSTRIDE_G(bjd);

    /* compute dot product of local elements */

    for (; cn > 0; --cn) {
      an = I8(__fort_block_bounds)(as, 1, acl, &al, &au);
      bn = I8(__fort_block_bounds)(bs, 1, bcl, &bl, &bu);

      aoff = F90_LBASE_G(as) - 1 +
             F90_DPTR_LSTRIDE_G(ajd) * (F90_DPTR_SSTRIDE_G(ajd) * al +
                                        F90_DPTR_SOFFSET_G(ajd) - alof);

      boff = F90_LBASE_G(bs) - 1 +
             F90_DPTR_LSTRIDE_G(bjd) * (F90_DPTR_SSTRIDE_G(bjd) * bl +
                                        F90_DPTR_SOFFSET_G(bjd) - blof);

      dotp(cb, bn, ab, aoff, astr, bb, boff, bstr);

      acl += DIST_DPTR_CS_G(ajd);
      bcl += DIST_DPTR_CS_G(bjd);
      alof += DIST_DPTR_CLOS_G(ajd);
      blof += DIST_DPTR_CLOS_G(bjd);
    }
  }

  /* do global reduction to complete dot product */

  I8(__fort_reduce_section)(cb, kind, len, NULL, kind, len, 1,
			     I8(__fort_g_sum)[kind], 1, as);

  I8(__fort_replicate_result)(cb, kind, len, NULL, kind, len, 1, as);

  if (bs == bs1)
    I8(__fort_copy_out)(bb0, bb, bs0, bs, __INTENT_IN);
  if (as == as1)
    I8(__fort_copy_out)(ab0, ab, as0, as, __INTENT_IN);

#if defined(DEBUG)
  if (__fort_test & DEBUG_MMUL) {
    printf("%d dotpr ", GET_DIST_LCPU);
    __fort_show_scalar(cb, kind);
    printf("\n");
  }
#endif
}

static void I8(mmul_mxm)(char *cb0, char *ab0, char *bb0, F90_Desc *cs0,
                         F90_Desc *as0, F90_Desc *bs0)
{
  char *ab = 0, *bb = 0, *cb = 0;
  DECL_HDR_PTRS(as);
  DECL_HDR_PTRS(bs);
  DECL_HDR_PTRS(cs);
  DECL_HDR_VARS(as1);
  DECL_HDR_VARS(bs1);
  DECL_HDR_VARS(cs1);
  DECL_DIM_PTRS(aid);
  DECL_DIM_PTRS(ajd);
  DECL_DIM_PTRS(bjd);
  DECL_DIM_PTRS(bkd);
  DECL_DIM_PTRS(cid);
  DECL_DIM_PTRS(ckd);
  __INT_T flags, kind, len;
  mmul_fn mmul;
  __INT_T a0, ai, ais, ajs, b0, bjs, bk, bks, c0, cik, cis, ck, cks, icl, icn,
      il, ilof, in, iu, kcl, kcn, kl, klof, kn, ku, jn;
  int copy_required;

  kind = F90_KIND_G(as0);
  len = F90_LEN_G(as0);

  copy_required = I8(is_nonsequential_section)(cs0, F90_RANK_G(cs0));

  if (copy_required) {
    cs = cs1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_OUT + __INHERIT +
             __TRANSCRIPTIVE_DIST_TARGET + __TRANSCRIPTIVE_DIST_FORMAT);

    ENTFTN(QOPY_IN, qopy_in)
    (&cb, (__POINT_T *)ABSENT, cb0, cs, cb0, cs0, &_2, &kind, &len, &flags, &_1,
     &_1); /* lb(1), lb(2) */
  } else {
    cb = cb0;
    cs = cs0;
  }

  copy_required = I8(is_nonsequential_section)(as0, F90_RANK_G(as0));

  if (copy_required) {
    as = as1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_IN +
             __PRESCRIPTIVE_ALIGN_TARGET);
    ENTFTN(QOPY_IN, qopy_in)
    (&ab, (__POINT_T *)ABSENT, ab0, as, ab0, as0, &_2, &kind, &len, &flags, cs,
     &_1, &_2,      /* align-target, conform, collapse */
     &_1, &_1, &_0, /* axis, stride, offset */
     &_0,           /* single */
     &_1, &_1);     /* lb(1), lb(2) */
  } else {
    ab = ab0;
    as = as0;
  }

  copy_required = I8(is_nonsequential_section)(bs0, F90_RANK_G(bs0));

  if (copy_required) {
    bs = bs1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_IN +
             __PRESCRIPTIVE_ALIGN_TARGET);
    ENTFTN(QOPY_IN, qopy_in)
    (&bb, (__POINT_T *)ABSENT, bb0, bs, bb0, bs0, &_2, &kind, &len, &flags, cs,
     &_2, &_1,      /* align-target, conform, collapse */
     &_2, &_1, &_0, /* axis, stride, offset */
     &_0,           /* single */
     &_1, &_1);     /* lb(1), lb(2) */
  } else {
    bb = bb0;
    bs = bs0;
  }

  switch (kind) {
  case __REAL4:
    mmul = (mmul_fn)dotp_real4;
    break;
  case __REAL8:
    mmul = (mmul_fn)dotp_real8;
    break;
  case __CPLX8:
    mmul = (mmul_fn)mmul_cplx8;
    break;
  case __CPLX16:
    mmul = (mmul_fn)mmul_cplx16;
    break;
  case __INT1:
    mmul = (mmul_fn)dotp_int1;
    break;
  case __INT2:
    mmul = (mmul_fn)dotp_int2;
    break;
  case __INT4:
    mmul = (mmul_fn)dotp_int4;
    break;
  case __LOG1:
    mmul = (mmul_fn)dotp_log1;
    break;
  case __LOG2:
    mmul = (mmul_fn)dotp_log2;
    break;
  case __LOG4:
    mmul = (mmul_fn)dotp_log4;
    break;
  case __INT8:
    mmul = (mmul_fn)dotp_int8;
    break;
  case __LOG8:
    mmul = (mmul_fn)dotp_log8;
    break;
  default:
    __fort_abort("MATMUL: unimplemented for data type");
  }

  SET_DIM_PTRS(aid, as, 0);
  SET_DIM_PTRS(ajd, as, 1);
  SET_DIM_PTRS(bjd, bs, 0);
  SET_DIM_PTRS(bkd, bs, 1);
  SET_DIM_PTRS(cid, cs, 0);
  SET_DIM_PTRS(ckd, cs, 1);

  in = F90_DPTR_EXTENT_G(aid);
  jn = F90_DPTR_EXTENT_G(bjd);
  kn = F90_DPTR_EXTENT_G(ckd);
  if (F90_DPTR_EXTENT_G(cid) != in || F90_DPTR_EXTENT_G(ajd) != jn ||
      F90_DPTR_EXTENT_G(bkd) != kn)
    __fort_abort("MATMUL: nonconforming array shapes");

#if defined(DEBUG)
  if (__fort_test & DEBUG_MMUL) {
    printf("%d mmul A\n", GET_DIST_LCPU);
    I8(__fort_print_local)(ab, as);
    printf("%d mmul B\n", GET_DIST_LCPU);
    I8(__fort_print_local)(bb, bs);
  }
#endif

  I8(__fort_fills)(cb, cs, GET_DIST_ZED);

  if (~F90_FLAGS_G(cs) & __OFF_TEMPLATE) {
    I8(__fort_cycle_bounds)(cs);

/* common dimension of a and b must be collapsed... */

    I8(__fort_cycle_bounds)(as);
    a0 = F90_LBASE_G(as) - 1 +
         F90_DPTR_LSTRIDE_G(ajd) *
             (F90_DPTR_LBOUND_G(ajd) * F90_DPTR_SSTRIDE_G(ajd) +
              F90_DPTR_SOFFSET_G(ajd) - DIST_DPTR_CLOF_G(ajd));

    I8(__fort_cycle_bounds)(bs);
    b0 = F90_LBASE_G(bs) - 1 +
         F90_DPTR_LSTRIDE_G(bjd) *
             (F90_DPTR_LBOUND_G(bjd) * F90_DPTR_SSTRIDE_G(bjd) +
              F90_DPTR_SOFFSET_G(bjd) - DIST_DPTR_CLOF_G(bjd));

    c0 = F90_LBASE_G(cs) - 1;

    ais = F90_DPTR_SSTRIDE_G(aid) * F90_DPTR_LSTRIDE_G(aid);
    ajs = F90_DPTR_SSTRIDE_G(ajd) * F90_DPTR_LSTRIDE_G(ajd);
    bjs = F90_DPTR_SSTRIDE_G(bjd) * F90_DPTR_LSTRIDE_G(bjd);
    bks = F90_DPTR_SSTRIDE_G(bkd) * F90_DPTR_LSTRIDE_G(bkd);
    cis = F90_DPTR_SSTRIDE_G(cid) * F90_DPTR_LSTRIDE_G(cid);
    cks = F90_DPTR_SSTRIDE_G(ckd) * F90_DPTR_LSTRIDE_G(ckd);

    kcl = DIST_DPTR_CL_G(ckd);
    kcn = DIST_DPTR_CN_G(ckd);
    klof = DIST_DPTR_CLOF_G(ckd);
    for (; kcn > 0; --kcn, kcl += DIST_DPTR_CS_G(ckd)) {

      kn = I8(__fort_block_bounds)(cs, 2, kcl, &kl, &ku);

      bk = b0 +
           F90_DPTR_LSTRIDE_G(bkd) *
               (F90_DPTR_SSTRIDE_G(bkd) * kl + F90_DPTR_SOFFSET_G(bkd) - klof);
      ck = c0 +
           F90_DPTR_LSTRIDE_G(ckd) *
               (F90_DPTR_SSTRIDE_G(ckd) * kl + F90_DPTR_SOFFSET_G(ckd) - klof);

      for (; kn > 0; --kn, ++kl) {

        icl = DIST_DPTR_CL_G(cid);
        icn = DIST_DPTR_CN_G(cid);
        ilof = DIST_DPTR_CLOF_G(cid);
        for (; icn > 0; icn--, icl += DIST_DPTR_CS_G(cid)) {

          in = I8(__fort_block_bounds)(cs, 1, icl, &il, &iu);

          ai = a0 +
               F90_DPTR_LSTRIDE_G(aid) * (F90_DPTR_SSTRIDE_G(aid) * il +
                                          F90_DPTR_SOFFSET_G(aid) - ilof);

          cik = ck +
                F90_DPTR_LSTRIDE_G(cid) * (F90_DPTR_SSTRIDE_G(cid) * il +
                                           F90_DPTR_SOFFSET_G(cid) - ilof);

          for (; in > 0; --in, ++il) {

            mmul(cb + cik * len, jn, ab, ai, ajs, bb, bk, bjs);

            ai += ais;
            cik += cis;
          }
          ilof += DIST_DPTR_CLOS_G(cid);
        }
        bk += bks;
        ck += cks;
      }
      klof += DIST_DPTR_CLOS_G(ckd);
    }
  }

#if defined(DEBUG)
  if (__fort_test & DEBUG_MMUL) {
    printf("%d mmul C=AxB\n", GET_DIST_LCPU);
    I8(__fort_print_local)(cb, cs);
  }
#endif

  if (bs == bs1)
    I8(__fort_copy_out)(bb0, bb, bs0, bs, __INTENT_IN);
  if (as == as1)
    I8(__fort_copy_out)(ab0, ab, as0, as, __INTENT_IN);
  if (cs == cs1)
    I8(__fort_copy_out)(cb0, cb, cs0, cs, __INTENT_OUT);
}

static void I8(mmul_vxm)(char *cb0, char *ab0, char *bb0, F90_Desc *cs0,
                         F90_Desc *as0, F90_Desc *bs0)
{
  char *ab = 0, *bb = 0, *cb = 0;
  DECL_HDR_PTRS(as);
  DECL_HDR_PTRS(bs);
  DECL_HDR_PTRS(cs);
  DECL_HDR_VARS(as1);
  DECL_HDR_VARS(bs1);
  DECL_HDR_VARS(cs1);
  DECL_DIM_PTRS(ajd);
  DECL_DIM_PTRS(bjd);
  DECL_DIM_PTRS(bkd);
  DECL_DIM_PTRS(ckd);
  __INT_T flags, kind, len;
  mmul_fn mmul;
  __INT_T a0, aj, ajcl, ajclof, ajclos, ajcn, ajcs, ajcu, ajl, ajn, ajs, aju,
      b0, bj, bjcl, bjclof, bjcn, bjk, bjl, bjn, bjs, bju, bkcl, bkclof, bkcn,
      bkl, bkn, bks, bku, c0, ck, ckcl, ckclof, ckclos, ckcn, ckcs, ckcu, ckl,
      ckn, cks, cku, nj, nk;
  int copy_required;

  kind = F90_KIND_G(bs0);
  len = F90_LEN_G(bs0);

  copy_required = I8(is_nonsequential_section)(bs0, F90_RANK_G(bs0));
  if (copy_required) {
    bs = bs1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_IN + __INHERIT +
             __TRANSCRIPTIVE_DIST_TARGET + __TRANSCRIPTIVE_DIST_FORMAT);
    ENTFTN(QOPY_IN, qopy_in)
    (&bb, (__POINT_T *)ABSENT, bb0, bs, bb0, bs0, &_2, &kind, &len, &flags, &_1,
     &_1); /* lb(1), lb(2) */
  } else {
    bb = bb0;
    bs = bs0;
  }

  copy_required = I8(is_nonsequential_section)(as0, F90_RANK_G(as0));
  if (copy_required) {
    as = as1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_IN +
             __PRESCRIPTIVE_ALIGN_TARGET);
    ENTFTN(QOPY_IN, qopy_in)
    (&ab, (__POINT_T *)ABSENT, ab0, as, ab0, as0, &_1, &kind, &len, &flags, bs,
     &_1, &_0,      /* align-target, conform, collapse */
     &_1, &_1, &_0, /* axis, stride, offset */
     &_0,           /* single */
     &_1);          /* lb */
  } else {
    ab = ab0;
    as = as0;
  }

  copy_required = I8(is_nonsequential_section)(cs0, F90_RANK_G(cs0));
  if (copy_required) {
    cs = cs1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_OUT +
             __PRESCRIPTIVE_ALIGN_TARGET);
    ENTFTN(QOPY_IN, qopy_in)
    (&cb, (__POINT_T *)ABSENT, cb0, cs, cb0, cs0, &_1, &kind, &len, &flags, bs,
     &_1, &_0,      /* align-target, conform, collapse */
     &_2, &_1, &_0, /* axis, stride, offset */
     &_0,           /* single */
     &_1);          /* lb */
  } else {
    cb = cb0;
    cs = cs0;
  }

  switch (kind) {
  case __REAL4:
    mmul = (mmul_fn)dotp_real4;
    break;
  case __REAL8:
    mmul = (mmul_fn)dotp_real8;
    break;
  case __CPLX8:
    mmul = (mmul_fn)mmul_cplx8;
    break;
  case __CPLX16:
    mmul = (mmul_fn)mmul_cplx16;
    break;
  case __INT1:
    mmul = (mmul_fn)dotp_int1;
    break;
  case __INT2:
    mmul = (mmul_fn)dotp_int2;
    break;
  case __INT4:
    mmul = (mmul_fn)dotp_int4;
    break;
  case __LOG1:
    mmul = (mmul_fn)dotp_log1;
    break;
  case __LOG2:
    mmul = (mmul_fn)dotp_log2;
    break;
  case __LOG4:
    mmul = (mmul_fn)dotp_log4;
    break;
  case __INT8:
    mmul = (mmul_fn)dotp_int8;
    break;
  case __LOG8:
    mmul = (mmul_fn)dotp_log8;
    break;
  default:
    __fort_abort("MATMUL: unimplemented for data type");
  }

  SET_DIM_PTRS(ajd, as, 0);
  SET_DIM_PTRS(bjd, bs, 0);
  SET_DIM_PTRS(bkd, bs, 1);
  SET_DIM_PTRS(ckd, cs, 0);

  nj = F90_DPTR_EXTENT_G(bjd);
  nk = F90_DPTR_EXTENT_G(ckd);
  if (F90_DPTR_EXTENT_G(ajd) != nj || F90_DPTR_EXTENT_G(bkd) != nk)
    __fort_abort("MATMUL: nonconforming array shapes");

  I8(__fort_fills)(cb, cs, GET_DIST_ZED);

  if (~F90_FLAGS_G(bs) & __OFF_TEMPLATE) {
    I8(__fort_cycle_bounds)(bs);

    a0 =
        F90_LBASE_G(as) - 1 + F90_DPTR_SOFFSET_G(ajd) * F90_DPTR_LSTRIDE_G(ajd);
    b0 = F90_LBASE_G(bs) - 1 +
         F90_DPTR_SOFFSET_G(bjd) * F90_DPTR_LSTRIDE_G(bjd) +
         F90_DPTR_SOFFSET_G(bkd) * F90_DPTR_LSTRIDE_G(bkd);
    c0 =
        F90_LBASE_G(cs) - 1 + F90_DPTR_SOFFSET_G(ckd) * F90_DPTR_LSTRIDE_G(ckd);

    ajs = F90_DPTR_SSTRIDE_G(ajd) * F90_DPTR_LSTRIDE_G(ajd);
    bjs = F90_DPTR_SSTRIDE_G(bjd) * F90_DPTR_LSTRIDE_G(bjd);
    bks = F90_DPTR_SSTRIDE_G(bkd) * F90_DPTR_LSTRIDE_G(bkd);
    cks = F90_DPTR_SSTRIDE_G(ckd) * F90_DPTR_LSTRIDE_G(ckd);

    bjcl = DIST_DPTR_CL_G(bjd);
    bjcn = DIST_DPTR_CN_G(bjd);
    bjclof = DIST_DPTR_CLOF_G(bjd);
    bjn = 0;
    ajcn = 0;
    ajn = 0;

    while (bjn > 0 || bjcn > 0) {
      if (bjn == 0) {
        bjn = I8(__fort_block_bounds)(bs, 1, bjcl, &bjl, &bju);
        bj = b0 +
             (F90_DPTR_SSTRIDE_G(bjd) * bjl - bjclof) * F90_DPTR_LSTRIDE_G(bjd);
        bjcl += DIST_DPTR_CS_G(bjd);
        bjclof += DIST_DPTR_CLOS_G(bjd);
        --bjcn;
      }
      if (ajn == 0) {
        if (ajcn <= 0) {
          ajl = F90_DPTR_LBOUND_G(ajd) + bjl - F90_DPTR_LBOUND_G(bjd);
          aju = ajl + bjn - 1;
          ajcn = I8(__fort_cyclic_loop)(as, 1, ajl, aju, 1, &ajcl, &ajcu, &ajcs,
                                       &ajclof, &ajclos);
        }
        ajn = I8(__fort_block_bounds)(as, 1, ajcl, &ajl, &aju);
        aj = a0 +
             (F90_DPTR_SSTRIDE_G(ajd) * ajl - ajclof) * F90_DPTR_LSTRIDE_G(ajd);
        ajcl += ajcs;
        ajclof += ajclos;
        --ajcn;
      }
      nj = ajn < bjn ? ajn : bjn;

      bkcl = DIST_DPTR_CL_G(bkd);
      bkcn = DIST_DPTR_CN_G(bkd);
      bkclof = DIST_DPTR_CLOF_G(bkd);
      bkn = 0;
      ckcn = 0;
      ckn = 0;

      while (bkn > 0 || bkcn > 0) {
        if (bkn == 0) {
          bkn = I8(__fort_block_bounds)(bs, 2, bkcl, &bkl, &bku);
          bjk = bj +
                (F90_DPTR_SSTRIDE_G(bkd) * bkl - bkclof) *
                    F90_DPTR_LSTRIDE_G(bkd);
          bkcl += DIST_DPTR_CS_G(bkd);
          bkclof += DIST_DPTR_CLOS_G(bkd);
          --bkcn;
        }
        if (ckn == 0) {
          if (ckcn <= 0) {
            ckl = F90_DPTR_LBOUND_G(ckd) + bkl - F90_DPTR_LBOUND_G(bkd);
            cku = ckl + bkn - 1;
            ckcn = I8(__fort_cyclic_loop)(cs, 1, ckl, cku, 1, &ckcl, &ckcu,
                                         &ckcs, &ckclof, &ckclos);
          }
          ckn = I8(__fort_block_bounds)(cs, 1, ckcl, &ckl, &cku);
          ck = c0 +
               (F90_DPTR_SSTRIDE_G(ckd) * ckl - ckclof) *
                   F90_DPTR_LSTRIDE_G(ckd);
          ckcl += ckcs;
          ckclof += ckclos;
          --ckcn;
        }
        nk = ckn < bkn ? ckn : bkn;
        bkl += nk;
        bkn -= nk;
        ckn -= nk;
        while (--nk >= 0) {
          mmul(cb + ck * len, nj, ab, aj, ajs, bb, bjk, bjs);
          bjk += bks;
          ck += cks;
        }
      }
      bjl += nj;
      bjn -= nj;
      bj += nj * bjs;

      ajn -= nj;
      aj += nj * ajs;
    }
  }

  I8(__fort_reduce_section)(cb, kind, len, NULL, kind, len, F90_LSIZE_G(cs),
			     I8(__fort_g_sum)[kind], 1, bs);

  I8(__fort_replicate_result)(cb, kind, len, NULL, kind, len, F90_LSIZE_G(cs), 
                               bs);

  if (cs == cs1)
    I8(__fort_copy_out)(cb0, cb, cs0, cs, __INTENT_OUT);
  if (as == as1)
    I8(__fort_copy_out)(ab0, ab, as0, as, __INTENT_IN);
  if (bs == bs1)
    I8(__fort_copy_out)(bb0, bb, bs0, bs, __INTENT_IN);
}

static void I8(mmul_mxv)(char *cb0, char *ab0, char *bb0, F90_Desc *cs0,
                         F90_Desc *as0, F90_Desc *bs0)
{
  char *ab = 0, *bb = 0, *cb = 0;
  DECL_HDR_PTRS(as);
  DECL_HDR_PTRS(bs);
  DECL_HDR_PTRS(cs);
  DECL_HDR_VARS(as1);
  DECL_HDR_VARS(bs1);
  DECL_HDR_VARS(cs1);
  DECL_DIM_PTRS(aid);
  DECL_DIM_PTRS(ajd);
  DECL_DIM_PTRS(bjd);
  DECL_DIM_PTRS(cid);
  __INT_T flags, kind, len;
  mmul_fn mmul;
  __INT_T a0, aicl, aicn, aij, ail, aiclof, ain, ais, aiu, aj, ajcl, ajcn, ajl,
      ajclof, ajn, ajs, aju, b0, bj, bjcl, bjclof, bjclos, bjcn, bjcs, bjcu,
      bjl, bjn, bjs, bju, c0, ci, cicl, ciclof, ciclos, cicn, cics, cicu, cil,
      cin, cis, ciu, ni, nj;
  int copy_required;

#if defined(DEBUG)
  if (__fort_test & DEBUG_MMUL) {
    printf("%d mmul_mxv c\n", GET_DIST_LCPU);
    I8(__fort_describe)(cb0, cs0);
    printf("%d mmul_mxv a\n", GET_DIST_LCPU);
    I8(__fort_describe)(ab0, as0);
    printf("%d mmul_mxv b\n", GET_DIST_LCPU);
    I8(__fort_describe)(bb0, bs0);
    printf("\n");
  }
#endif

  kind = F90_KIND_G(as0);
  len = F90_LEN_G(as0);

  copy_required = I8(is_nonsequential_section)(as0, F90_RANK_G(as0));
  if (copy_required) {
    as = as1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_IN + __INHERIT +
             __TRANSCRIPTIVE_DIST_TARGET + __TRANSCRIPTIVE_DIST_FORMAT);
    ENTFTN(QOPY_IN, qopy_in)
    (&ab, (__POINT_T *)ABSENT, ab0, as, ab0, as0, &_2, &kind, &len, &flags, &_1,
     &_1); /* lb(1), lb(2) */
  } else {
    ab = ab0;
    as = as0;
  }

  copy_required = I8(is_nonsequential_section)(bs0, F90_RANK_G(bs0));
  if (copy_required) {
    bs = bs1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_IN +
             __PRESCRIPTIVE_ALIGN_TARGET);
    ENTFTN(QOPY_IN, qopy_in)
    (&bb, (__POINT_T *)ABSENT, bb0, bs, bb0, bs0, &_1, &kind, &len, &flags, as,
     &_1, &_0,      /* align-target, conform, collapse */
     &_2, &_1, &_0, /* axis, stride, offset */
     &_0,           /* single */
     &_1);          /* lb */
  } else {
    bb = bb0;
    bs = bs0;
  }

  copy_required = I8(is_nonsequential_section)(cs0, F90_RANK_G(cs0));
  if (copy_required) {
    cs = cs1;
    flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_OUT +
             __PRESCRIPTIVE_ALIGN_TARGET);
    ENTFTN(QOPY_IN, qopy_in)
    (&cb, (__POINT_T *)ABSENT, cb0, cs, cb0, cs0, &_1, &kind, &len, &flags, as,
     &_1, &_0,      /* align-target, conform, collapse */
     &_1, &_1, &_0, /* axis, stride, offset */
     &_0,           /* single */
     &_1);          /* lb */
  } else {
    cb = cb0;
    cs = cs0;
  }

  switch (kind) {
  case __REAL4:
    mmul = (mmul_fn)dotp_real4;
    break;
  case __REAL8:
    mmul = (mmul_fn)dotp_real8;
    break;
  case __CPLX8:
    mmul = (mmul_fn)mmul_cplx8;
    break;
  case __CPLX16:
    mmul = (mmul_fn)mmul_cplx16;
    break;
  case __INT1:
    mmul = (mmul_fn)dotp_int1;
    break;
  case __INT2:
    mmul = (mmul_fn)dotp_int2;
    break;
  case __INT4:
    mmul = (mmul_fn)dotp_int4;
    break;
  case __LOG1:
    mmul = (mmul_fn)dotp_log1;
    break;
  case __LOG2:
    mmul = (mmul_fn)dotp_log2;
    break;
  case __LOG4:
    mmul = (mmul_fn)dotp_log4;
    break;
  case __INT8:
    mmul = (mmul_fn)dotp_int8;
    break;
  case __LOG8:
    mmul = (mmul_fn)dotp_log8;
    break;
  default:
    __fort_abort("MATMUL: unimplemented for data type");
  }

  SET_DIM_PTRS(aid, as, 0);
  SET_DIM_PTRS(ajd, as, 1);
  SET_DIM_PTRS(bjd, bs, 0);
  SET_DIM_PTRS(cid, cs, 0);

  ni = F90_DPTR_EXTENT_G(aid);
  nj = F90_DPTR_EXTENT_G(bjd);
  if (F90_DPTR_EXTENT_G(cid) != ni || F90_DPTR_EXTENT_G(ajd) != nj)
    __fort_abort("MATMUL: nonconforming array shapes");

  I8(__fort_fills)(cb, cs, GET_DIST_ZED);

  if (~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
    I8(__fort_cycle_bounds)(as);

    a0 = F90_LBASE_G(as) - 1 +
         F90_DPTR_SOFFSET_G(aid) * F90_DPTR_LSTRIDE_G(aid) +
         F90_DPTR_SOFFSET_G(ajd) * F90_DPTR_LSTRIDE_G(ajd);
    b0 =
        F90_LBASE_G(bs) - 1 + F90_DPTR_SOFFSET_G(bjd) * F90_DPTR_LSTRIDE_G(bjd);
    c0 =
        F90_LBASE_G(cs) - 1 + F90_DPTR_SOFFSET_G(cid) * F90_DPTR_LSTRIDE_G(cid);

    ais = F90_DPTR_SSTRIDE_G(aid) * F90_DPTR_LSTRIDE_G(aid);
    ajs = F90_DPTR_SSTRIDE_G(ajd) * F90_DPTR_LSTRIDE_G(ajd);
    bjs = F90_DPTR_SSTRIDE_G(bjd) * F90_DPTR_LSTRIDE_G(bjd);
    cis = F90_DPTR_SSTRIDE_G(cid) * F90_DPTR_LSTRIDE_G(cid);

    ajcl = DIST_DPTR_CL_G(ajd);
    ajcn = DIST_DPTR_CN_G(ajd);
    ajclof = DIST_DPTR_CLOF_G(ajd);
    ajn = 0;
    bjcn = 0;
    bjn = 0;

    while (ajn > 0 || ajcn > 0) {
      if (ajn == 0) {
        ajn = I8(__fort_block_bounds)(as, 2, ajcl, &ajl, &aju);
        aj = a0 +
             (F90_DPTR_SSTRIDE_G(ajd) * ajl - ajclof) * F90_DPTR_LSTRIDE_G(ajd);
        ajcl += DIST_DPTR_CS_G(ajd);
        ajclof += DIST_DPTR_CLOS_G(ajd);
        --ajcn;
      }
      if (bjn == 0) {
        if (bjcn <= 0) {
          bjl = F90_DPTR_LBOUND_G(bjd) + ajl - F90_DPTR_LBOUND_G(ajd);
          bju = bjl + ajn - 1;
          bjcn = I8(__fort_cyclic_loop)(bs, 1, bjl, bju, 1, &bjcl, &bjcu, &bjcs,
                                       &bjclof, &bjclos);
        }
        bjn = I8(__fort_block_bounds)(bs, 1, bjcl, &bjl, &bju);
        bj = b0 +
             (F90_DPTR_SSTRIDE_G(bjd) * bjl - bjclof) * F90_DPTR_LBOUND_G(bjd);
        bjcl += bjcs;
        bjclof += bjclos;
        --bjcn;
      }
      nj = bjn < ajn ? bjn : ajn;

      aicl = DIST_DPTR_CL_G(aid);
      aicn = DIST_DPTR_CN_G(aid);
      aiclof = DIST_DPTR_CLOF_G(aid);
      ain = 0;
      cicn = 0;
      cin = 0;

      while (ain > 0 || aicn > 0) {
        if (ain == 0) {
          ain = I8(__fort_block_bounds)(as, 1, aicl, &ail, &aiu);
          aij =
              aj +
              (F90_DPTR_SSTRIDE_G(aid) * ail - aiclof) * F90_DPTR_LBOUND_G(aid);
          aicl += DIST_DPTR_CS_G(aid);
          aiclof += DIST_DPTR_CLOS_G(aid);
          --aicn;
        }
        if (cin == 0) {
          if (cicn <= 0) {
            cil = F90_DPTR_LBOUND_G(cid) + ail - F90_DPTR_LBOUND_G(aid);
            ciu = cil + ain - 1;
            cicn = I8(__fort_cyclic_loop)(cs, 1, cil, ciu, 1, &cicl, &cicu,
                                         &cics, &ciclof, &ciclos);
          }
          cin = I8(__fort_block_bounds)(cs, 1, cicl, &cil, &ciu);
          ci =
              c0 +
              (F90_DPTR_SSTRIDE_G(cid) * cil - ciclof) * F90_DPTR_LBOUND_G(cid);
          cicl += cics;
          ciclof += ciclos;
          --cicn;
        }
        ni = cin < ain ? cin : ain;
        ail += ni;
        ain -= ni;
        cin -= ni;
        while (--ni >= 0) {
          mmul(cb + ci * len, nj, ab, aij, ajs, bb, bj, bjs);
          aij += ais;
          ci += cis;
        }
      }
      ajl += nj;
      ajn -= nj;
      aj += nj * ajs;

      bjn -= nj;
      bj += nj * bjs;
    }
  }

  I8(__fort_reduce_section)(cb, kind, len, NULL, kind, len, F90_LSIZE_G(cs),
			     I8(__fort_g_sum)[kind], 2, as);

  I8(__fort_replicate_result)(cb, kind, len, NULL, kind, len, F90_LSIZE_G(cs), 
                               as);

  if (cs == cs1)
    I8(__fort_copy_out)(cb0, cb, cs0, cs, __INTENT_OUT);
  if (bs == bs1)
    I8(__fort_copy_out)(bb0, bb, bs0, bs, __INTENT_IN);
  if (as == as1)
    I8(__fort_copy_out)(ab0, ab, as0, as, __INTENT_IN);
}

void ENTFTN(MATMUL, matmul)(char *cb, char *ab, char *bb, F90_Desc *cs,
                            F90_Desc *as, F90_Desc *bs)
{
  if (F90_RANK_G(as) == 2 & F90_RANK_G(bs) == 2)
    I8(mmul_mxm)(cb, ab, bb, cs, as, bs);
  else if (F90_RANK_G(as) == 1 & F90_RANK_G(bs) == 2)
    I8(mmul_vxm)(cb, ab, bb, cs, as, bs);
  else if (F90_RANK_G(as) == 2 & F90_RANK_G(bs) == 1)
    I8(mmul_mxv)(cb, ab, bb, cs, as, bs);
  else
    __fort_abort("MATMUL: non-conforming array shapes");
}
