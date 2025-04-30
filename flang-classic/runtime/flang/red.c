/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * Intrinsic reduction functions
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

extern void (*__fort_scalar_copy[__NTYPES])(void *rp, const void *sp, int len);

void ENTFTN(QOPY_IN, qopy_in)(char **dptr, __POINT_T *doff, char *dbase,
                              F90_Desc *dd, char *ab, F90_Desc *ad,
                              __INT_T *p_rank, __INT_T *p_kind, __INT_T *p_len,
                              __INT_T *p_flags, ...);

#if !defined(DESC_I8)
void
__fort_red_unimplemented()
{
  char str[80];

  sprintf(str, "%s: unimplemented for data type", __fort_red_what);
  __fort_abort(str);
}

void
__fort_red_abort(const char *msg)
{
  char str[80];

  sprintf(str, "%s: %s", __fort_red_what, msg);
  __fort_abort(str);
}
#endif

/** \brief reduction, dim argument absent */
static void I8(red_scalar_loop)(red_parm *z, __INT_T aof, __INT_T ll, int dim)
{
  DECL_HDR_PTRS(as);
  DECL_HDR_PTRS(ms);
  DECL_DIM_PTRS(asd);
  DECL_DIM_PTRS(msd);
  char *ap;
  __LOG_T *mp;

  __INT_T abl, abn, abu, acl, acn, aclof, ahop, ao, extent, i, li, ls, mhop,
      mlow;

  as = z->as;
  SET_DIM_PTRS(asd, as, dim - 1);
  acn = DIST_DPTR_CN_G(asd);
  acl = DIST_DPTR_CL_G(asd);
  aclof = DIST_DPTR_CLOF_G(asd);
  ahop = F90_DPTR_SSTRIDE_G(asd) * F90_DPTR_LSTRIDE_G(asd);
  if (z->mask_present) {
    ms = z->ms;
    SET_DIM_PTRS(msd, ms, dim - 1);
    mlow = F90_DPTR_LBOUND_G(msd);
    mhop = F90_DPTR_SSTRIDE_G(msd) * F90_DPTR_LSTRIDE_G(msd);
  } else {
    mlow = mhop = 0;
    mp = z->mb;
  }

  extent = F90_DPTR_EXTENT_G(asd);
  if (extent < 0)
    extent = 0;
  ll *= extent;

  if (dim == 1 && acn > 1 && DIST_DPTR_BLOCK_G(asd) == DIST_DPTR_TSTRIDE_G(asd) &&
      (!z->mask_present || z->mask_stored_alike)) {

    /* cyclic(1) optimization */

    abl = acl - DIST_DPTR_TOFFSET_G(asd);
    if (DIST_DPTR_TSTRIDE_G(asd) != 1)
      abl /= DIST_DPTR_TSTRIDE_G(asd);
    ao = aof +
         (F90_DPTR_SSTRIDE_G(asd) * abl + F90_DPTR_SOFFSET_G(asd) - aclof) *
             F90_DPTR_LSTRIDE_G(asd);

    i = abl - F90_DPTR_LBOUND_G(asd); /* section ordinal index */
    li = ll + i + 1;                  /* linearized location */
    ls = DIST_DPTR_PSHAPE_G(asd);      /* location stride */

    if (z->mask_present)
      mp = (__LOG_T *)((char *)(z->mb) + (ao << z->lk_shift));

    ap = z->ab + ao * F90_LEN_G(as);
    if (z->l_fn_b) {
      z->l_fn_b(z->rb, acn, ap, ahop, mp, mhop, z->xb, li, ls, z->len, z->back);
    } else {
      z->l_fn(z->rb, acn, ap, ahop, mp, mhop, z->xb, li, ls, z->len);
    }

    return;
  }
  while (acn > 0) {
    abn = I8(__fort_block_bounds)(as, dim, acl, &abl, &abu);
    ao = aof +
         (F90_DPTR_SSTRIDE_G(asd) * abl + F90_DPTR_SOFFSET_G(asd) - aclof) *
             F90_DPTR_LSTRIDE_G(asd);

    i = abl - F90_DPTR_LBOUND_G(asd); /* ordinal index (zero-based) */
    li = ll + i + 1;                  /* linearized location */

    z->mi[dim - 1] = mlow + i; /* mask index */

    if (dim > 1) {
      while (abn > 0) {
        I8(red_scalar_loop)(z, ao, li, dim - 1);
        li++;
        z->mi[dim - 1]++;
        ao += ahop;
        abn--;
      }
    } else {
      if (z->mask_present) {
        if (z->mask_stored_alike)
          mp = (__LOG_T *)((char *)(z->mb) + (ao << z->lk_shift));
        else {
          mp = I8(__fort_local_address)(z->mb, ms, z->mi);
          if (mp == NULL)
            __fort_red_abort("mask misalignment");
        }
      }
      ap = z->ab + ao * F90_LEN_G(as);
      if (z->l_fn_b) {
        z->l_fn_b(z->rb, abn, ap, ahop, mp, mhop, z->xb, li, 1, z->len,
                  z->back);
      } else {
        z->l_fn(z->rb, abn, ap, ahop, mp, mhop, z->xb, li, 1, z->len);
      }
    }
    acl += DIST_DPTR_CS_G(asd);
    aclof += DIST_DPTR_CLOS_G(asd);
    --acn;
  }
}

void I8(__fort_red_scalar)(red_parm *z, char *rb, char *ab, char *mb,
                          F90_Desc *rs, F90_Desc *as, F90_Desc *ms, __INT_T *xb,
                          red_enum op)
{
  DECL_DIM_PTRS(asd);
  __INT_T ao, i, m, p, q;

  z->rb = rb;
  z->rs = rs;
  z->ab = ab;
  z->as = as;
  z->mb = (__LOG_T *)mb;
  z->ms = ms;
  z->xb = xb;
  z->dim = 0;

  I8(__fort_cycle_bounds)(as);

  __fort_scalar_copy[z->kind](rb, z->zb, z->len);

  if (xb != NULL) {
    for (i = F90_RANK_G(as); --i >= 0;)
      xb[i] = 0;
  }

  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (z->mask_present) {
    z->mask_stored_alike = I8(__fort_stored_alike)(as, ms);
    if (z->mask_stored_alike)
      z->mb += F90_LBASE_G(ms);
    for (i = F90_RANK_G(ms); --i >= 0;)
      z->mi[i] = F90_DIM_LBOUND_G(ms, i);
  } else if (!ISPRESENT(mb) || I8(__fort_fetch_log)(mb, ms))
    z->mb = GET_DIST_TRUE_LOG_ADDR;
  else
    return; /* scalar mask == .false. */

  if (~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
    z->ab += F90_LBASE_G(as) * F90_LEN_G(as);
    ao = -1;
    I8(red_scalar_loop)(z, ao, 0, F90_RANK_G(as));
  }

  I8(__fort_reduce_section)(rb, z->kind, z->len,
			     xb, __INT, sizeof(__INT_T), 1,
			     z->g_fn, -1, as);

  I8(__fort_replicate_result)(rb, z->kind, z->len,
			       xb, __INT, sizeof(__INT_T), 1, as);

  if (xb != NULL && (p = xb[0]) > 0) {
    for (i = 0; i < F90_RANK_G(as); i++) {
      SET_DIM_PTRS(asd, as, i);
      m = F90_DPTR_EXTENT_G(asd);
      q = (p - 1) / m;
      xb[i] = p - q * m;
      p = q;
    }
  }
}

/** \brief  SAME as previous, but allow any logical kind  for the mask */
void I8(__fort_red_scalarlk)(red_parm *z, char *rb, char *ab, char *mb,
                            F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                            __INT_T *xb, red_enum op)
{
  DECL_DIM_PTRS(asd);
  __INT_T ao, i, m, p, q;

  z->rb = rb;
  z->rs = rs;
  z->ab = ab;
  z->as = as;
  z->mb = (__LOG_T *)mb;
  z->ms = ms;
  z->xb = xb;
  z->dim = 0;

  I8(__fort_cycle_bounds)(as);

  __fort_scalar_copy[z->kind](rb, z->zb, z->len);

  if (xb != NULL) {
    for (i = F90_RANK_G(as); --i >= 0;)
      xb[i] = 0;
  }

  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (z->mask_present) {
    z->mask_stored_alike = I8(__fort_stored_alike)(as, ms);
    if (z->mask_stored_alike)
      z->mb = (__LOG_T *)((char *)(z->mb) + (F90_LBASE_G(ms) << z->lk_shift));
    for (i = F90_RANK_G(ms); --i >= 0;)
      z->mi[i] = F90_DIM_LBOUND_G(ms, i);
  } else if (!ISPRESENT(mb) || I8(__fort_fetch_log)(mb, ms))
    z->mb = GET_DIST_TRUE_LOG_ADDR;
  else
    return; /* scalar mask == .false. */

  if (~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
    z->ab += F90_LBASE_G(as) * F90_LEN_G(as);
    ao = -1;

    I8(red_scalar_loop)(z, ao, 0, F90_RANK_G(as));
  }

  I8(__fort_reduce_section)(rb, z->kind, z->len,
			     xb, __INT, sizeof(__INT_T), 1,
			     z->g_fn, -1, as);

  I8(__fort_replicate_result)(rb, z->kind, z->len,
			       xb, __INT, sizeof(__INT_T), 1, as);
  if (xb != NULL && (p = xb[0]) > 0) {
    for (i = 0; i < F90_RANK_G(as); i++) {
      SET_DIM_PTRS(asd, as, i);
      m = F90_DPTR_EXTENT_G(asd);
      q = (p - 1) / m;
      xb[i] = p - q * m;
      p = q;
    }
  }
}

/** \brief reduction, dim argument present */
static void I8(red_array_loop)(red_parm *z, __INT_T rof, __INT_T aof, int rdim,
                               int adim)
{
  DECL_HDR_PTRS(as);
  DECL_HDR_PTRS(rs);
  DECL_HDR_PTRS(ms);
  DECL_DIM_PTRS(asd);
  DECL_DIM_PTRS(rsd);
  DECL_DIM_PTRS(msd);
  char *ap, *rp;
  __LOG_T *mp;
  __INT4_T *lp;

  __INT_T abl, abn, abu, acl, acn, aclof, ahop, ao, i, li, ls, mhop, mlow, rbl,
      rbn, rbu, rcl, rclof, rhop, ro;

#if defined(DEBUG)
  if (__fort_test & DEBUG_REDU) {
    printf("%d red_array_loop rdim=%d rof=%d adim=%d aof=%d\n", GET_DIST_LCPU,
           rdim, rof, adim, aof);
  }
#endif

  if (rdim > 0) {
    rs = z->rs;
    SET_DIM_PTRS(rsd, rs, rdim - 1);
    rcl = DIST_DPTR_CL_G(rsd);
    rclof = DIST_DPTR_CLOF_G(rsd);
    rhop = F90_DPTR_SSTRIDE_G(rsd) * F90_DPTR_LSTRIDE_G(rsd);

    if (adim == z->dim)
      --adim;
  } else {
    rp = z->rb + rof * z->len;
    adim = z->dim;
  }

  as = z->as;
  SET_DIM_PTRS(asd, as, adim - 1);
  acn = DIST_DPTR_CN_G(asd);
  acl = DIST_DPTR_CL_G(asd);
  aclof = DIST_DPTR_CLOF_G(asd);
  ahop = F90_DPTR_SSTRIDE_G(asd) * F90_DPTR_LSTRIDE_G(asd);

  if (z->mask_present) {
    ms = z->ms;
    SET_DIM_PTRS(msd, ms, adim - 1);
    mlow = F90_DPTR_LBOUND_G(msd);
    mhop = F90_DPTR_SSTRIDE_G(msd) * F90_DPTR_LSTRIDE_G(msd);
  } else {
    mlow = mhop = 0;
    mp = z->mb;
  }

  if (rdim <= 0 && acn > 1 &&
      DIST_DPTR_BLOCK_G(asd) == DIST_DPTR_TSTRIDE_G(asd) &&
      (!z->mask_present || z->mask_stored_alike)) {

    /* cyclic(1) optimization */

    abl = acl - DIST_DPTR_TOFFSET_G(asd);
    if (DIST_DPTR_TSTRIDE_G(asd) != 1)
      abl /= DIST_DPTR_TSTRIDE_G(asd);
    ao = aof +
         (F90_DPTR_SSTRIDE_G(asd) * abl + F90_DPTR_SOFFSET_G(asd) - aclof) *
             F90_DPTR_LSTRIDE_G(asd);

    i = abl - F90_DPTR_LBOUND_G(asd); /* section ordinal index */
    li = i + 1;                       /* linearized location */
    ls = DIST_DPTR_PSHAPE_G(asd);      /* location stride */

    if (z->mask_present)
      mp = (__LOG_T *)((char *)(z->mb) + (ao << z->lk_shift));

    if (z->xb != NULL)
      lp = ((__INT4_T *)z->xb) + rof;
    else
      lp = NULL;

    ap = z->ab + ao * F90_LEN_G(as);
    if (z->l_fn_b) {
      z->l_fn_b(rp, acn, ap, ahop, mp, mhop, lp, li, ls, z->len, z->back);
    } else {
      z->l_fn(rp, acn, ap, ahop, mp, mhop, lp, li, ls, z->len);
    }

    return;
  }

  while (acn > 0) {
    abn = I8(__fort_block_bounds)(as, adim, acl, &abl, &abu);
    ao = aof +
         (F90_DPTR_SSTRIDE_G(asd) * abl + F90_DPTR_SOFFSET_G(asd) - aclof) *
             F90_DPTR_LSTRIDE_G(asd);

    i = abl - F90_DPTR_LBOUND_G(asd); /* ordinal index (zero-based) */
    li = i + 1;                       /* linearized location */
    z->mi[adim - 1] = mlow + i;       /* mask array index */

    if (rdim > 0) {

      /* this array dimension is not being reduced.  there is a
         result element corresponding to every element of this
         array dimension. */

      rbn = I8(__fort_block_bounds)(rs, rdim, rcl, &rbl, &rbu);
#if defined(DEBUG)
      if (rbn != abn)
        __fort_red_abort("result misalignment");
#endif
      ro = rof +
           (F90_DPTR_SSTRIDE_G(rsd) * rbl + F90_DPTR_SOFFSET_G(rsd) - rclof) *
               F90_DPTR_LSTRIDE_G(rsd);

      while (abn > 0) {
        I8(red_array_loop)(z, ro, ao, rdim - 1, adim - 1);
        ro += rhop;
        ao += ahop;
        z->mi[adim - 1]++;
        --abn;
      }
      rcl += DIST_DPTR_CS_G(rsd);
      rclof += DIST_DPTR_CLOS_G(rsd);
    } else {

      /* this is the array dimension being reduced */

      if (z->mask_present) {
        if (z->mask_stored_alike)
          mp = (__LOG_T *)((char *)(z->mb) + (ao << z->lk_shift));
        else {
          mp = I8(__fort_local_address)(z->mb, ms, z->mi);
          if (mp == NULL)
            __fort_red_abort("mask misalignment");
        }
      }

      if (z->xb != NULL)
        lp = ((__INT4_T *)z->xb) + rof;
      else
        lp = NULL;

      ap = z->ab + ao * F90_LEN_G(as);
      if (z->l_fn_b) {
        z->l_fn_b(rp, abn, ap, ahop, mp, mhop, lp, li, 1, z->len, z->back);
      } else {
        z->l_fn(rp, abn, ap, ahop, mp, mhop, lp, li, 1, z->len);
      }
    }
    acl += DIST_DPTR_CS_G(asd);
    aclof += DIST_DPTR_CLOS_G(asd);
    --acn;
  }
#if defined(DEBUG)
  if (__fort_test & DEBUG_REDU && rdim <= 0) {
    printf("%d red_array_loop rp=%x *rp=", GET_DIST_LCPU, rp);
    __fort_print_scalar(rp, z->kind);
    printf("\n");
  }
#endif
}

void I8(__fort_kred_scalarlk)(red_parm *z, char *rb, char *ab, char *mb,
                             F90_Desc *rs, F90_Desc *as, F90_Desc *ms,
                             __INT8_T *xb, red_enum op)
{
  DECL_DIM_PTRS(asd);
  __INT_T ao, i, m, p, q;

  z->rb = rb;
  z->rs = rs;
  z->ab = ab;
  z->as = as;
  z->mb = (__LOG_T *)mb;
  z->ms = ms;
  z->xb = (__INT_T *)xb;
  z->dim = 0;

  I8(__fort_cycle_bounds)(as);

  __fort_scalar_copy[z->kind](rb, z->zb, z->len);

  if (xb != NULL) {
    for (i = F90_RANK_G(as); --i >= 0;)
      xb[i] = 0;
  }

  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (z->mask_present) {
    z->mask_stored_alike = I8(__fort_stored_alike)(as, ms);
    if (z->mask_stored_alike)
      z->mb = (__LOG_T *)((char *)(z->mb) + (F90_LBASE_G(ms) << z->lk_shift));
    for (i = F90_RANK_G(ms); --i >= 0;)
      z->mi[i] = F90_DIM_LBOUND_G(ms, i);
  } else if (!ISPRESENT(mb) || I8(__fort_fetch_log)(mb, ms))
    z->mb = GET_DIST_TRUE_LOG_ADDR;
  else
    return; /* scalar mask == .false. */

  if (~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
    z->ab += F90_LBASE_G(as) * F90_LEN_G(as);
    ao = -1;
    I8(red_scalar_loop)(z, ao, 0, F90_RANK_G(as));
  }

  I8(__fort_reduce_section)(rb, z->kind, z->len,
			     xb, __INT, sizeof(__INT_T), 1,
			     z->g_fn, -1, as);

  I8(__fort_replicate_result)(rb, z->kind, z->len,
			       xb, __INT, sizeof(__INT_T), 1, as);

  if (xb != NULL && (p = xb[0]) > 0) {
    for (i = 0; i < F90_RANK_G(as); i++) {
      SET_DIM_PTRS(asd, as, i);
      m = F90_DPTR_EXTENT_G(asd);
      q = (p - 1) / m;
      xb[i] = p - q * m;
      p = q;
    }
  }
}

static void I8(kred_array_loop)(red_parm *z, __INT_T rof, __INT_T aof, int rdim,
                                int adim)
{
  DECL_HDR_PTRS(as);
  DECL_HDR_PTRS(rs);
  DECL_HDR_PTRS(ms);
  DECL_DIM_PTRS(asd);
  DECL_DIM_PTRS(rsd);
  DECL_DIM_PTRS(msd);
  char *ap, *rp;
  __LOG_T *mp;
  __INT_T *lp;

  __INT_T abl, abn, abu, acl, acn, aclof, ahop, ao, i, li, ls, mhop, mlow, rbl,
      rbn, rbu, rcl, rclof, rhop, ro;

#if defined(DEBUG)
  if (__fort_test & DEBUG_REDU) {
    printf("%d red_array_loop rdim=%d rof=%d adim=%d aof=%d\n", GET_DIST_LCPU,
           rdim, rof, adim, aof);
  }
#endif

  if (rdim > 0) {
    rs = z->rs;
    SET_DIM_PTRS(rsd, rs, rdim - 1);
    rcl = DIST_DPTR_CL_G(rsd);
    rclof = DIST_DPTR_CLOF_G(rsd);
    rhop = F90_DPTR_SSTRIDE_G(rsd) * F90_DPTR_LSTRIDE_G(rsd);

    if (adim == z->dim)
      --adim;
  } else {
    rp = z->rb + rof * z->len;
    adim = z->dim;
  }

  as = z->as;
  SET_DIM_PTRS(asd, as, adim - 1);
  acn = DIST_DPTR_CN_G(asd);
  acl = DIST_DPTR_CL_G(asd);
  aclof = DIST_DPTR_CLOF_G(asd);
  ahop = F90_DPTR_SSTRIDE_G(asd) * F90_DPTR_LSTRIDE_G(asd);

  if (z->mask_present) {
    ms = z->ms;
    SET_DIM_PTRS(msd, ms, adim - 1);
    mlow = F90_DPTR_LBOUND_G(msd);
    mhop = F90_DPTR_SSTRIDE_G(msd) * F90_DPTR_LSTRIDE_G(msd);
  } else {
    mlow = mhop = 0;
    mp = z->mb;
  }

  if (rdim <= 0 && acn > 1 &&
      DIST_DPTR_BLOCK_G(asd) == DIST_DPTR_TSTRIDE_G(asd) &&
      (!z->mask_present || z->mask_stored_alike)) {

    /* cyclic(1) optimization */

    abl = acl - DIST_DPTR_TOFFSET_G(asd);
    if (DIST_DPTR_TSTRIDE_G(asd) != 1)
      abl /= DIST_DPTR_TSTRIDE_G(asd);
    ao = aof +
         (F90_DPTR_SSTRIDE_G(asd) * abl + F90_DPTR_SOFFSET_G(asd) - aclof) *
             F90_DPTR_LSTRIDE_G(asd);

    i = abl - F90_DPTR_LBOUND_G(asd); /* section ordinal index */
    li = i + 1;                       /* linearized location */
    ls = DIST_DPTR_PSHAPE_G(asd);      /* location stride */

    if (z->mask_present)
      mp = (__LOG_T *)((char *)(z->mb) + (ao << z->lk_shift));

    if (z->xb != NULL)
      lp = (__INT_T *)((char *)z->xb + rof * sizeof(__INT8_T));
    else
      lp = NULL;

    ap = z->ab + ao * F90_LEN_G(as);
    if (z->l_fn_b) {
      z->l_fn_b(rp, acn, ap, ahop, mp, mhop, lp, li, ls, z->len, z->back);
    } else {
      z->l_fn(rp, acn, ap, ahop, mp, mhop, lp, li, ls, z->len);
    }

    return;
  }

  while (acn > 0) {
    abn = I8(__fort_block_bounds)(as, adim, acl, &abl, &abu);
    ao = aof +
         (F90_DPTR_SSTRIDE_G(asd) * abl + F90_DPTR_SOFFSET_G(asd) - aclof) *
             F90_DPTR_LSTRIDE_G(asd);

    i = abl - F90_DPTR_LBOUND_G(asd); /* ordinal index (zero-based) */
    li = i + 1;                       /* linearized location */
    z->mi[adim - 1] = mlow + i;       /* mask array index */

    if (rdim > 0) {

      /* this array dimension is not being reduced.  there is a
         result element corresponding to every element of this
         array dimension. */

      rbn = I8(__fort_block_bounds)(rs, rdim, rcl, &rbl, &rbu);
#if defined(DEBUG)
      if (rbn != abn)
        __fort_red_abort("result misalignment");
#endif
      ro = rof +
           (F90_DPTR_SSTRIDE_G(rsd) * rbl + F90_DPTR_SOFFSET_G(rsd) - rclof) *
               F90_DPTR_LSTRIDE_G(rsd);

      while (abn > 0) {
        I8(kred_array_loop)(z, ro, ao, rdim - 1, adim - 1);
        ro += rhop;
        ao += ahop;
        z->mi[adim - 1]++;
        --abn;
      }
      rcl += DIST_DPTR_CS_G(rsd);
      rclof += DIST_DPTR_CLOS_G(rsd);
    } else {

      /* this is the array dimension being reduced */

      if (z->mask_present) {
        if (z->mask_stored_alike)
          mp = (__LOG_T *)((char *)(z->mb) + (ao << z->lk_shift));
        else {
          mp = I8(__fort_local_address)(z->mb, ms, z->mi);
          if (mp == NULL)
            __fort_red_abort("mask misalignment");
        }
      }

      if (z->xb != NULL)
        lp = (__INT_T *)((char *)z->xb + rof * sizeof(__INT8_T));
      else
        lp = NULL;

      ap = z->ab + ao * F90_LEN_G(as);
      if (z->l_fn_b) {
        z->l_fn_b(rp, abn, ap, ahop, mp, mhop, lp, li, 1, z->len, z->back);
      } else {
        z->l_fn(rp, abn, ap, ahop, mp, mhop, lp, li, 1, z->len);
      }
    }
    acl += DIST_DPTR_CS_G(asd);
    aclof += DIST_DPTR_CLOS_G(asd);
    --acn;
  }
#if defined(DEBUG)
  if (__fort_test & DEBUG_REDU && rdim <= 0) {
    printf("%d red_array_loop rp=%x *rp=", GET_DIST_LCPU, rp);
    __fort_print_scalar(rp, z->kind);
    printf("\n");
  }
#endif
}

void I8(__fort_red_array)(red_parm *z, char *rb0, char *ab, char *mb, char *db,
                         F90_Desc *rs0, F90_Desc *as, F90_Desc *ms,
                         F90_Desc *ds, red_enum op)
{
  DECL_HDR_PTRS(rs);
  DECL_HDR_VARS(rs1);
  char *rb = 0, *xb, *zb;
  __INT_T flags, kind, len, rank, _1 = 1;
  int i, rc, rl;
  __INT_T ao, ro;

  z->dim = I8(__fort_fetch_int)(db, ds);

#if defined(DEBUG)
  if (__fort_test & DEBUG_REDU) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs0);
    printf("@%x = %s(a", rb0, __fort_red_what);
    I8(__fort_show_section)(as);
    printf("@%x, dim=%d, mask", ab, z->dim);
    I8(__fort_show_section)(ms);
    printf("@%x)\n", mb);
  }
#endif

  if (as == NULL || F90_TAG_G(as) != __DESC)
    __fort_red_abort("invalid array argument descriptor");
  if (z->dim < 1 || z->dim > F90_RANK_G(as))
    __fort_red_abort("invalid DIM argument");
  rank = F90_RANK_G(as) - 1;

  I8(__fort_cycle_bounds)(as);

  rs = rs0;
  rb = rb0;
  if (F90_TAG_G(rs0) == __DESC) {
    if ((op == __MINLOC || op == __MAXLOC || op == __FINDLOC) &&
        z->kind != __STR) {
      kind = __INT;
      len = sizeof(__INT_T);
    } else {
      kind = z->kind;
      len = z->len;
    }
    if (DIST_MAPPED_G(rs0) ||
        I8(is_nonsequential_section)(rs0, F90_RANK_G(rs0))) {
      rs = rs1;
      flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_OUT + __INHERIT +
               __TRANSCRIPTIVE_DIST_TARGET + __TRANSCRIPTIVE_DIST_FORMAT);
      ENTFTN(QOPY_IN, qopy_in)
      (&rb, (__POINT_T *)ABSENT, rb0, rs, rb0, rs0, &rank, &kind, &len, &flags,
       &_1, &_1, &_1, &_1, &_1, &_1, &_1); /* lb */
    }
    I8(__fort_cycle_bounds)(rs);
    ro = F90_LBASE_G(rs) - 1;
    rc = F90_LSIZE_G(rs);
    rl = F90_LEN_G(rs);
  } else {
#if defined(DEBUG)
    if (rank != 0)
      __fort_red_abort("result/array rank mismatch");
#endif
    ro = 0;
    rc = 1;
    rl = GET_DIST_SIZE_OF(F90_TAG_G(rs0));
  }

  if (op == __MINLOC || op == __MAXLOC || op == __FINDLOC) {
    if (rc > 0)
      memset(rb, '\0', rc * rl);
    xb = rb;
    rb = (char *)__fort_gmalloc(rc * F90_LEN_G(as));
  } else
    xb = NULL;

  z->rb = rb;
  z->rs = rs;
  z->ab = ab;
  z->as = as;
  z->mb = (__LOG_T *)mb;
  z->ms = ms;
  z->xb = (__INT_T *)xb;

  zb = z->zb;
#if defined(DEBUG)
  if (zb == NULL)
    __fort_red_abort("missing null constant (unimplemented)");
#endif
  switch (z->kind) {
  case __LOG1:
    for (i = 0; i < rc; ++i)
      ((__LOG1_T *)rb)[i] = *(__LOG1_T *)zb;
    break;
  case __LOG2:
    for (i = 0; i < rc; ++i)
      ((__LOG2_T *)rb)[i] = *(__LOG2_T *)zb;
    break;
  case __LOG4:
    for (i = 0; i < rc; ++i)
      ((__LOG4_T *)rb)[i] = *(__LOG4_T *)zb;
    break;
  case __LOG8:
    for (i = 0; i < rc; ++i)
      ((__LOG8_T *)rb)[i] = *(__LOG8_T *)zb;
    break;
  case __INT1:
    for (i = 0; i < rc; ++i)
      ((__INT1_T *)rb)[i] = *(__INT1_T *)zb;
    break;
  case __INT2:
    for (i = 0; i < rc; ++i)
      ((__INT2_T *)rb)[i] = *(__INT2_T *)zb;
    break;
  case __INT4:
    for (i = 0; i < rc; ++i)
      ((__INT4_T *)rb)[i] = *(__INT4_T *)zb;
    break;
  case __INT8:
    for (i = 0; i < rc; ++i)
      ((__INT8_T *)rb)[i] = *(__INT8_T *)zb;
    break;
  case __REAL4:
    for (i = 0; i < rc; ++i)
      ((__REAL4_T *)rb)[i] = *(__REAL4_T *)zb;
    break;
  case __REAL8:
    for (i = 0; i < rc; ++i)
      ((__REAL8_T *)rb)[i] = *(__REAL8_T *)zb;
    break;
  case __REAL16:
    for (i = 0; i < rc; ++i)
      ((__REAL16_T *)rb)[i] = *(__REAL16_T *)zb;
    break;
  case __CPLX8:
    for (i = 0; i < rc; ++i)
      ((__CPLX8_T *)rb)[i] = *(__CPLX8_T *)zb;
    break;
  case __CPLX16:
    for (i = 0; i < rc; ++i)
      ((__CPLX16_T *)rb)[i] = *(__CPLX16_T *)zb;
    break;
  case __CPLX32:
    for (i = 0; i < rc; ++i)
      ((__CPLX32_T *)rb)[i] = *(__CPLX32_T *)zb;
    break;
  case __STR:
    if (op == __FINDLOC) {
      for (i = 0; i < rc; ++i)
        memcpy(&rb[i * z->len], (char *)zb, z->len);
    } else {
      for (i = 0; i < rc; ++i)
        memset(&rb[i * z->len], *(__STR_T *)zb, z->len * sizeof(__STR_T));
    }
    break;
  default:
    __fort_red_abort("unsupported result type");
  }

  z->mask_present = (F90_TAG_G(ms) == __DESC && F90_RANK_G(ms) > 0);
  if (z->mask_present) {
    z->mask_stored_alike = I8(__fort_stored_alike)(as, ms);
    if (z->mask_stored_alike)
      z->mb = (__LOG_T *)((char *)z->mb + (F90_LBASE_G(ms) << z->lk_shift));
    for (i = F90_RANK_G(ms); --i >= 0;)
      z->mi[i] = F90_DIM_LBOUND_G(ms, i);
  } else if (!ISPRESENT(mb) || I8(__fort_fetch_log)(mb, ms))
    z->mb = GET_DIST_TRUE_LOG_ADDR;
  else
    z->mb = (__LOG_T *)GET_DIST_ZED; /* scalar mask == .false. */

  if (~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
    z->ab += F90_LBASE_G(as) * F90_LEN_G(as);
    ao = -1;
    I8(red_array_loop)(z, ro, ao, rank, F90_RANK_G(as));
  }

  I8(__fort_reduce_section)(rb, z->kind, z->len,
			     xb, __INT, sizeof(__INT_T), rc,
			     z->g_fn, z->dim, as);

  I8(__fort_replicate_result)(rb, z->kind, z->len,
			       xb, __INT, sizeof(__INT_T), rc, as);

  if (xb != NULL) {
    __fort_gfree(rb);
    rb = xb;
  }
  if (rs == rs1)
    I8(__fort_copy_out)(rb0, rb, rs0, rs, __INTENT_OUT);
}

/** \brief  SAME as previous, but allow any logical kind  for the mask */
void I8(__fort_red_arraylk)(red_parm *z, char *rb0, char *ab, char *mb, char *db,
                           F90_Desc *rs0, F90_Desc *as, F90_Desc *ms,
                           F90_Desc *ds, red_enum op)
{
  DECL_HDR_PTRS(rs);
  DECL_HDR_VARS(rs1);
  char *rb = 0, *xb, *zb;
  __INT_T flags, kind, len, rank, _1 = 1;
  int i, rc, rl;
  __INT_T ao, ro;

  z->dim = I8(__fort_fetch_int)(db, ds);

#if defined(DEBUG)
  if (__fort_test & DEBUG_REDU) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs0);
    printf("@%x = %s(a", rb0, __fort_red_what);
    I8(__fort_show_section)(as);
    printf("@%x, dim=%d, mask", ab, z->dim);
    I8(__fort_show_section)(ms);
    printf("@%x)\n", mb);
  }
#endif

  if (as == NULL || F90_TAG_G(as) != __DESC)
    __fort_red_abort("invalid array argument descriptor");
  if (z->dim < 1 || z->dim > F90_RANK_G(as))
    __fort_red_abort("invalid DIM argument");
  rank = F90_RANK_G(as) - 1;

  I8(__fort_cycle_bounds)(as);

  rs = rs0;
  rb = rb0;
  if (F90_TAG_G(rs0) == __DESC) {
    if ((op == __MINLOC || op == __MAXLOC || op == __FINDLOC) &&
        z->kind != __STR) {
      kind = __INT;
      len = sizeof(__INT_T);
    } else {
      kind = z->kind;
      len = z->len;
    }
    if (DIST_MAPPED_G(rs0) ||
        I8(is_nonsequential_section)(rs0, F90_RANK_G(rs0))) {
      rs = rs1;
      flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_OUT + __INHERIT +
               __TRANSCRIPTIVE_DIST_TARGET + __TRANSCRIPTIVE_DIST_FORMAT);
      ENTFTN(QOPY_IN, qopy_in)
      (&rb, (__POINT_T *)ABSENT, rb0, rs, rb0, rs0, &rank, &kind, &len, &flags,
       &_1, &_1, &_1, &_1, &_1, &_1, &_1); /* lb */
    }
    I8(__fort_cycle_bounds)(rs);
    ro = F90_LBASE_G(rs) - 1;
    rc = F90_LSIZE_G(rs);
    rl = F90_LEN_G(rs);
  } else {
#if defined(DEBUG)
    if (rank != 0)
      __fort_red_abort("result/array rank mismatch");
#endif
    rank = 0;
    ro = 0;
    rc = 1;
    rl = GET_DIST_SIZE_OF(F90_TAG_G(rs0));
  }

  if (op == __MINLOC || op == __MAXLOC || op == __FINDLOC) {
    if (rc > 0)
      memset(rb, '\0', rc * rl);
    xb = rb;
    rb = (char *)__fort_gmalloc(rc * F90_LEN_G(as));
  } else
    xb = NULL;

  z->rb = rb;
  z->rs = rs;
  z->ab = ab;
  z->as = as;
  z->mb = (__LOG_T *)mb;
  z->ms = ms;
  z->xb = (__INT_T *)xb;

  zb = z->zb;
#if defined(DEBUG)
  if (zb == NULL)
    __fort_red_abort("missing null constant (unimplemented)");
#endif
  switch (z->kind) {
  case __LOG1:
    for (i = 0; i < rc; ++i)
      ((__LOG1_T *)rb)[i] = *(__LOG1_T *)zb;
    break;
  case __LOG2:
    for (i = 0; i < rc; ++i)
      ((__LOG2_T *)rb)[i] = *(__LOG2_T *)zb;
    break;
  case __LOG4:
    for (i = 0; i < rc; ++i)
      ((__LOG4_T *)rb)[i] = *(__LOG4_T *)zb;
    break;
  case __LOG8:
    for (i = 0; i < rc; ++i)
      ((__LOG8_T *)rb)[i] = *(__LOG8_T *)zb;
    break;
  case __INT1:
    for (i = 0; i < rc; ++i)
      ((__INT1_T *)rb)[i] = *(__INT1_T *)zb;
    break;
  case __INT2:
    for (i = 0; i < rc; ++i)
      ((__INT2_T *)rb)[i] = *(__INT2_T *)zb;
    break;
  case __INT4:
    for (i = 0; i < rc; ++i)
      ((__INT4_T *)rb)[i] = *(__INT4_T *)zb;
    break;
  case __INT8:
    for (i = 0; i < rc; ++i)
      ((__INT8_T *)rb)[i] = *(__INT8_T *)zb;
    break;
  case __REAL4:
    for (i = 0; i < rc; ++i)
      ((__REAL4_T *)rb)[i] = *(__REAL4_T *)zb;
    break;
  case __REAL8:
    for (i = 0; i < rc; ++i)
      ((__REAL8_T *)rb)[i] = *(__REAL8_T *)zb;
    break;
  case __REAL16:
    for (i = 0; i < rc; ++i)
      ((__REAL16_T *)rb)[i] = *(__REAL16_T *)zb;
    break;
  case __CPLX8:
    for (i = 0; i < rc; ++i)
      ((__CPLX8_T *)rb)[i] = *(__CPLX8_T *)zb;
    break;
  case __CPLX16:
    for (i = 0; i < rc; ++i)
      ((__CPLX16_T *)rb)[i] = *(__CPLX16_T *)zb;
    break;
  case __CPLX32:
    for (i = 0; i < rc; ++i)
      ((__CPLX32_T *)rb)[i] = *(__CPLX32_T *)zb;
    break;
  case __STR:
    if (op == __FINDLOC) {
      for (i = 0; i < rc; ++i)
        memcpy(&rb[i * z->len], (char *)zb, z->len);
    } else {
      for (i = 0; i < rc; ++i)
        memset(&rb[i * z->len], *(__STR_T *)zb, z->len * sizeof(__STR_T));
    }
    break;
  default:
    __fort_red_abort("unsupported result type");
  }

  if (z->mask_present) {
    z->mask_stored_alike = I8(__fort_stored_alike)(as, ms);
    if (z->mask_stored_alike)
      z->mb = (__LOG_T *)((char *)z->mb + (F90_LBASE_G(ms) << z->lk_shift));
    for (i = F90_RANK_G(ms); --i >= 0;)
      z->mi[i] = F90_DIM_LBOUND_G(ms, i);
  } else if (!ISPRESENT(mb) || I8(__fort_fetch_log)(mb, ms))
    z->mb = GET_DIST_TRUE_LOG_ADDR;
  else
    z->mb = (__LOG_T *)GET_DIST_ZED; /* scalar mask == .false. */

  if (~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
    z->ab += F90_LBASE_G(as) * F90_LEN_G(as);
    ao = -1;
    I8(red_array_loop)(z, ro, ao, rank, F90_RANK_G(as));
  }

  I8(__fort_reduce_section)(rb, z->kind, z->len,
			     xb, __INT, sizeof(__INT_T), rc,
			     z->g_fn, z->dim, as);

  I8(__fort_replicate_result)(rb, z->kind, z->len,
			       xb, __INT, sizeof(__INT_T), rc, as);

  if (xb != NULL) {
    __fort_gfree(rb);
    rb = xb;
  }
  if (rs == rs1)
    I8(__fort_copy_out)(rb0, rb, rs0, rs, __INTENT_OUT);
}

void I8(__fort_kred_arraylk)(red_parm *z, char *rb0, char *ab, char *mb,
                            char *db, F90_Desc *rs0, F90_Desc *as, F90_Desc *ms,
                            F90_Desc *ds, red_enum op)
{
  DECL_HDR_PTRS(rs);
  DECL_HDR_VARS(rs1);
  char *rb = 0, *xb, *zb;
  __INT_T flags, kind, len, rank, _1 = 1;
  int i, rc, rl;
  __INT_T ao, ro;

  z->dim = I8(__fort_fetch_int)(db, ds);

#if defined(DEBUG)
  if (__fort_test & DEBUG_REDU) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs0);
    printf("@%x = %s(a", rb0, __fort_red_what);
    I8(__fort_show_section)(as);
    printf("@%x, dim=%d, mask", ab, z->dim);
    I8(__fort_show_section)(ms);
    printf("@%x)\n", mb);
  }
#endif

  if (as == NULL || F90_TAG_G(as) != __DESC)
    __fort_red_abort("invalid array argument descriptor");
  if (z->dim < 1 || z->dim > F90_RANK_G(as))
    __fort_red_abort("invalid DIM argument");
  rank = F90_RANK_G(as) - 1;

  I8(__fort_cycle_bounds)(as);

  rs = rs0;
  rb = rb0;
  if (F90_TAG_G(rs0) == __DESC) {
    if ((op == __MINLOC || op == __MAXLOC || op == __FINDLOC) &&
        z->kind != __STR) {
      kind = __INT8;
      len = sizeof(__INT8_T);
    } else {
      kind = z->kind;
      len = z->len;
    }
    if (DIST_MAPPED_G(rs0) ||
        I8(is_nonsequential_section)(rs0, F90_RANK_G(rs0))) {
      rs = rs1;
      flags = (__ASSUMED_SHAPE + __ASSUMED_OVERLAPS + __INTENT_OUT + __INHERIT +
               __TRANSCRIPTIVE_DIST_TARGET + __TRANSCRIPTIVE_DIST_FORMAT);
      ENTFTN(QOPY_IN, qopy_in)
      (&rb, (__POINT_T *)ABSENT, rb0, rs, rb0, rs0, &rank, &kind, &len, &flags,
       &_1, &_1, &_1, &_1, &_1, &_1, &_1); /* lb */
    }
    I8(__fort_cycle_bounds)(rs);
    ro = F90_LBASE_G(rs) - 1;
    rc = F90_LSIZE_G(rs);
    rl = F90_LEN_G(rs);
  } else {
#if defined(DEBUG)
    if (rank != 0)
      __fort_red_abort("result/array rank mismatch");
#endif
    rank = 0;
    ro = 0;
    rc = 1;
    rl = GET_DIST_SIZE_OF(F90_TAG_G(rs0)); /* same as sizeof(__INT8_T) */
  }

  if (op == __MINLOC || op == __MAXLOC || op == __FINDLOC) {
    if (rc > 0)
      memset(rb, '\0', rc * rl);
    xb = rb;
    rb = (char *)__fort_gmalloc(rc * F90_LEN_G(as));
  } else
    xb = NULL;

  z->rb = rb;
  z->rs = rs;
  z->ab = ab;
  z->as = as;
  z->mb = (__LOG_T *)mb;
  z->ms = ms;
  z->xb = (__INT_T *)xb;

  zb = z->zb;
#if defined(DEBUG)
  if (zb == NULL)
    __fort_red_abort("missing null constant (unimplemented)");
#endif
  switch (z->kind) {
  case __LOG1:
    for (i = 0; i < rc; ++i)
      ((__LOG1_T *)rb)[i] = *(__LOG1_T *)zb;
    break;
  case __LOG2:
    for (i = 0; i < rc; ++i)
      ((__LOG2_T *)rb)[i] = *(__LOG2_T *)zb;
    break;
  case __LOG4:
    for (i = 0; i < rc; ++i)
      ((__LOG4_T *)rb)[i] = *(__LOG4_T *)zb;
    break;
  case __LOG8:
    for (i = 0; i < rc; ++i)
      ((__LOG8_T *)rb)[i] = *(__LOG8_T *)zb;
    break;
  case __INT1:
    for (i = 0; i < rc; ++i)
      ((__INT1_T *)rb)[i] = *(__INT1_T *)zb;
    break;
  case __INT2:
    for (i = 0; i < rc; ++i)
      ((__INT2_T *)rb)[i] = *(__INT2_T *)zb;
    break;
  case __INT4:
    for (i = 0; i < rc; ++i)
      ((__INT4_T *)rb)[i] = *(__INT4_T *)zb;
    break;
  case __INT8:
    for (i = 0; i < rc; ++i)
      ((__INT8_T *)rb)[i] = *(__INT8_T *)zb;
    break;
  case __REAL4:
    for (i = 0; i < rc; ++i)
      ((__REAL4_T *)rb)[i] = *(__REAL4_T *)zb;
    break;
  case __REAL8:
    for (i = 0; i < rc; ++i)
      ((__REAL8_T *)rb)[i] = *(__REAL8_T *)zb;
    break;
  case __REAL16:
    for (i = 0; i < rc; ++i)
      ((__REAL16_T *)rb)[i] = *(__REAL16_T *)zb;
    break;
  case __CPLX8:
    for (i = 0; i < rc; ++i)
      ((__CPLX8_T *)rb)[i] = *(__CPLX8_T *)zb;
    break;
  case __CPLX16:
    for (i = 0; i < rc; ++i)
      ((__CPLX16_T *)rb)[i] = *(__CPLX16_T *)zb;
    break;
  case __CPLX32:
    for (i = 0; i < rc; ++i)
      ((__CPLX32_T *)rb)[i] = *(__CPLX32_T *)zb;
    break;
  case __STR:
    if (op == __FINDLOC) {
      for (i = 0; i < rc; ++i)
        memcpy(&rb[i * z->len], (char *)zb, z->len);
    } else {
      for (i = 0; i < rc; ++i)
        memset(&rb[i * z->len], *(__STR_T *)zb, z->len * sizeof(__STR_T));
    }
    break;
  default:
    __fort_red_abort("unsupported result type");
  }

  if (z->mask_present) {
    z->mask_stored_alike = I8(__fort_stored_alike)(as, ms);
    if (z->mask_stored_alike)
      z->mb = (__LOG_T *)((char *)z->mb + (F90_LBASE_G(ms) << z->lk_shift));
    for (i = F90_RANK_G(ms); --i >= 0;)
      z->mi[i] = F90_DIM_LBOUND_G(ms, i);
  } else if (!ISPRESENT(mb) || I8(__fort_fetch_log)(mb, ms))
    z->mb = GET_DIST_TRUE_LOG_ADDR;
  else
    z->mb = (__LOG_T *)GET_DIST_ZED; /* scalar mask == .false. */

  if (~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
    z->ab += F90_LBASE_G(as) * F90_LEN_G(as);
    ao = -1;
    I8(kred_array_loop)(z, ro, ao, rank, F90_RANK_G(as));
  }

  I8(__fort_reduce_section)(rb, z->kind, z->len,
			     xb, __INT, sizeof(__INT_T), rc,
			     z->g_fn, z->dim, as);

  I8(__fort_replicate_result)(rb, z->kind, z->len,
			       xb, __INT, sizeof(__INT_T), rc, as);

  if (xb != NULL) {
    __fort_gfree(rb);
    rb = xb;
  }
  if (rs == rs1)
    I8(__fort_copy_out)(rb0, rb, rs0, rs, __INTENT_OUT);
}

/** \brief set up result descriptor for reduction intrinsic -- used when the
 * dim arg is variable.  result dimensions are aligned with the
 * corresponding source dimensions and the result array becomes
 * replicated over the reduced dimension.  lbounds are set to 1 and
 * overlap allowances are set to 0.
 */
void ENTFTN(REDUCE_DESCRIPTOR,
            reduce_descriptor)(F90_Desc *rd,   /* result descriptor */
                               __INT_T *kindb, /* result kind */
                               __INT_T *lenb,  /* result data item length */
                               F90_Desc *ad,   /* array descriptor */
                               __INT_T *dimb)  /* dimension */
{
  DECL_DIM_PTRS(add);
  DECL_HDR_PTRS(td);
  dtype kind;
  __INT_T dim, extent, len, m, offset, rx, ax, tx;

#if defined(DEBUG)
  if (F90_TAG_G(ad) != __DESC)
    __fort_abort("reduction intrinsic: invalid array arg");
#endif

  kind = (dtype)*kindb;
  len = *lenb;
  dim = *dimb;
  if (dim < 1 || dim > F90_RANK_G(ad))
    __fort_abort("reduction intrinsic: invalid dim");

  td = DIST_ALIGN_TARGET_G(ad);
  __DIST_INIT_DESCRIPTOR(rd, F90_RANK_G(ad) - 1, kind, len, F90_FLAGS_G(ad), td);
  for (rx = ax = 1; ax <= F90_RANK_G(ad); ++ax) {
    if (ax == dim)
      continue;
    SET_DIM_PTRS(add, ad, ax - 1);
    extent = F90_DPTR_EXTENT_G(add);
    offset = DIST_DPTR_TSTRIDE_G(add) * (F90_DPTR_LBOUND_G(add) - 1) +
             DIST_DPTR_TOFFSET_G(add);

    /* 
     * added gen_block argument to __fort_set_alignment ...
     * Should last arg be &DIST_DIM_GEN_BLOCK_G(td,(DIST_DPTR_TAXIS_G(add))-1) or
     * &DIST_DPTR_GEN_BLOCK_G(add) ???
     */
    I8(__fort_set_alignment)(rd, rx, 1, extent, DIST_DPTR_TAXIS_G(add),
			        DIST_DPTR_TSTRIDE_G(add), offset,
			        &DIST_DIM_GEN_BLOCK_G(td,(DIST_DPTR_TAXIS_G(add))-1));
    __DIST_SET_ALLOCATION(rd, rx, 0, 0);
    ++rx;
  }
  m = DIST_SINGLE_G(ad);
  for (tx = 1; m > 0; ++tx, m >>= 1) {
    if (m & 1)
      I8(__fort_set_single)(rd, td, tx, DIST_INFO_G(ad, tx - 1), __SINGLE);
  }
  I8(__fort_finish_descriptor)(rd);
}

void *I8(__fort_create_conforming_mask_array)(const char *what, char *ab,
                                              char *mb, F90_Desc *as,
                                              F90_Desc *ms, F90_Desc *new_ms)
{

  /* Create a conforming mask array. Returns a pointer to the
   * array and assigns a new descriptor in ms. Caller responsible
   * for the __fort_gfree() ...
   */

  __INT_T mask_kind;
  __INT_T mask_len;
  __INT_T i, _255 = 255;
  void *mask_array;

  if (!ISSCALAR(ms)) {
    __fort_abort("__fort_create_conforming_mask_array: bad mask descriptor");
  }

  mask_kind = F90_TAG_G(ms);

  switch (mask_kind) {

  case __LOG1:
    mask_len = sizeof(__LOG1_T);
    break;

  case __LOG2:

    mask_len = sizeof(__LOG2_T);
    break;

  case __LOG4:
    mask_len = sizeof(__LOG4_T);
    break;

  case __LOG8:
    mask_len = sizeof(__LOG8_T);
    break;

  default:
    printf("%d %s: bad type for mask loc=1\n", GET_DIST_LCPU, what);
    __fort_abort((char *)0);
  }

  ENTFTN(INSTANCE, instance)
  (new_ms, as, &mask_kind, &mask_len, &_255); /*no overlaps*/
  mask_array = (void *)__fort_gmalloc(F90_GSIZE_G(new_ms) * mask_len);

  switch (mask_kind) {

  case __LOG1:
    for (i = 0; i < F90_LSIZE_G(new_ms); ++i)
      *((__LOG1_T *)mask_array + i) = *((__LOG1_T *)mb);
    break;

  case __LOG2:
    for (i = 0; i < F90_LSIZE_G(new_ms); ++i)
      *((__LOG2_T *)mask_array + i) = *((__LOG2_T *)mb);
    break;

  case __LOG4:
    for (i = 0; i < F90_LSIZE_G(new_ms); ++i)
      *((__LOG4_T *)mask_array + i) = *((__LOG4_T *)mb);
    break;

  case __LOG8:
    for (i = 0; i < F90_LSIZE_G(new_ms); ++i)
      *((__LOG8_T *)mask_array + i) = *((__LOG8_T *)mb);
    break;

  default:
    printf("%d %s: bad type for mask loc=2\n", GET_DIST_LCPU, what);
    __fort_abort((char *)0);
  }

  return mask_array;
}
