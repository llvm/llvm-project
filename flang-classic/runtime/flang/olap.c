/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* olap.c -- overlap shift communication */

#include "stdioInterf.h"
#include "fioMacros.h"

#include "fort_vars.h"
extern void (*__fort_scalar_copy[__NTYPES])(void *rp, const void *sp, int size);

/* overlap shift communication schedule */

typedef struct {
  sked sked;
  chdr *ch1[MAXDIMS], *ch2[MAXDIMS], *ch3[MAXDIMS];
  double boundary[2];
  enum { __SHIFT, __EOSHIFT, __CSHIFT } style;
  int rank;
  int nsh[MAXDIMS];       /* negative shift amounts */
  int psh[MAXDIMS];       /* positive shift amounts */
  int span;               /* highest span-able dimension */
  int dim;                /* dimension being shifted */
  int dir;                /* shift direction */
  int extent;             /* dimension extent */
  int pcoord;             /* my processor coordinate */
  int plow;               /* low processor number */
  int pshape;             /* processor shape */
  int pstride;            /* processor stride */
  int fullcycle[MAXDIMS]; /* optimizable cyclic case if != 0 */
#if defined(DEBUG)
  char *base, *limit;
  int lb[MAXDIMS];
  int ub[MAXDIMS];
#endif
} olap_sked;

/* ENTFTN(comm_start) function: adjust base addresses and call doit */

static void I8(olap_start)(void *op, char *rb, char *sb, F90_Desc *rs,
                           F90_Desc *ss)
{
  olap_sked *o = (olap_sked *)op;
  __INT_T i;

#if defined(DEBUG)
  if (rb != sb || rs != ss)
    __fort_abort("olap_start: source/dest not same");
#endif
  for (i = 0; i < o->rank; ++i) {
    if (o->ch1[i] != NULL) {
      __fort_adjbase(o->ch1[i], rb, rb, F90_KIND_G(rs), F90_LEN_G(rs));
      __fort_doit(o->ch1[i]);
    }
    if (o->ch2[i] != NULL) {
      __fort_adjbase(o->ch2[i], (void *)o->boundary, rb, F90_KIND_G(rs),
                    F90_LEN_G(rs));
      __fort_doit(o->ch2[i]);
    }
    if (o->ch3[i] != NULL) {
      __fort_adjbase(o->ch3[i], rb, rb, F90_KIND_G(rs), F90_LEN_G(rs));
      __fort_doit(o->ch3[i]);
    }
  }
}

/* olap_free function: free channels and schedule structure */

static void
olap_free(void *op)
{
  olap_sked *o = (olap_sked *)op;
  int i;

  for (i = 0; i < o->rank; ++i) {
    __fort_frechn(o->ch1[i]);
    __fort_frechn(o->ch2[i]);
    __fort_frechn(o->ch3[i]);
  }
  __fort_free(o);
}

/* shift local data out to neighbors */

static void I8(olap_send)(char *src, F90_Desc *as, olap_sked *o, int len)
{
  DECL_DIM_PTRS(dd);
  char *adr;
  __INT_T ak, bl, bn, bu, cl, cn, cnt, i, l, clof, n, nl, nu, pc, pstep, str,
      tl, tu, u, wrap;

  SET_DIM_PTRS(dd, as, o->dim - 1);

  ak = F90_DPTR_LSTRIDE_G(dd) * F90_LEN_G(as);

  pstep = o->dir;
  if (DIST_DPTR_TSTRIDE_G(dd) < 0)
    pstep = -pstep;

  cl = DIST_DPTR_CL_G(dd);
  cn = DIST_DPTR_CN_G(dd);
  clof = DIST_DPTR_CLOF_G(dd) + DIST_DPTR_LAB_G(dd);

  for (; cn > 0; --cn, cl += DIST_DPTR_CS_G(dd), clof += DIST_DPTR_CLOS_G(dd)) {
    bn = I8(__fort_block_bounds)(as, o->dim, cl, &bl, &bu);

    pc = o->pcoord;

    do {

      /* find next neighbor's processor coordinate */

      pc += pstep;
      if (pc < 0)
        pc += o->pshape;
      else if (pc >= o->pshape)
        pc -= o->pshape;

      wrap = (pc - o->pcoord) * pstep <= 0;

      if (wrap && o->style != __CSHIFT)
        break;

      /* neighbor's template block bounds */

      if (DIST_DPTR_GEN_BLOCK_G(dd)) {

        /*
         * get neighbor's block bounds from gen_block
         */

        __INT_T *gb, *tgb;
        __INT_T i, direction, tExtent, aExtent;

        nl = DIST_DPTR_TLB_G(dd);
        tExtent = (DIST_DPTR_TUB_G(dd) - nl) + 1;
        aExtent = F90_DPTR_EXTENT_G(dd);

        if (tExtent != aExtent) {

          /* recompute gen_block */

          gb = tgb = I8(__fort_new_gen_block)(as, o->dim - 1);

        } else {
          tgb = 0;
          gb = DIST_DPTR_GEN_BLOCK_G(dd);
        }

        direction = (DIST_DPTR_TSTRIDE_G(dd) < 0);

        if (!direction)
          i = 0;
        else {
          i = (DIST_DPTR_PSHAPE_G(dd)) - 1;
          gb += i;
        }
        nu = *gb + (nl - 1);
        for (; i != pc;) {
          if (!direction) {
            ++gb;
            ++i;
          } else {
            --i;
            --gb;
          }
          nl = nu + 1;
          nu += *gb;
        }

        if (tgb)
          __fort_free(tgb);

        goto check_nl_and_nu;
      }

      tl = DIST_DPTR_TLB_G(dd) + pc * DIST_DPTR_BLOCK_G(dd);
      tu = tl + DIST_DPTR_BLOCK_G(dd) - 1;

      /* neighbor's owned bounds */

      nl = tl - DIST_DPTR_TOFFSET_G(dd);
      nu = tu - DIST_DPTR_TOFFSET_G(dd);
      if (DIST_DPTR_TSTRIDE_G(dd) != 1) {
        if (DIST_DPTR_TSTRIDE_G(dd) < 0)
          i = nl, nl = nu, nu = i;
        if (DIST_DPTR_TSTRIDE_G(dd) == -1) {
          nl = -nl;
          nu = -nu;
        } else {
          nl /= DIST_DPTR_TSTRIDE_G(dd);
          nu /= DIST_DPTR_TSTRIDE_G(dd);
        }
      }
      if (nl < F90_DPTR_LBOUND_G(dd))
        nl = F90_DPTR_LBOUND_G(dd);
      if (nu > DPTR_UBOUND_G(dd))
        nu = DPTR_UBOUND_G(dd);

    check_nl_and_nu:

      if (nl > nu)
        continue; /* neighbor has no data */

      /* find overlap bounds */

      if (o->dir < 0) { /* negative shift */
        l = nu + 1;
        u = nu + o->nsh[o->dim - 1];
        if (l > bu) {
          l -= o->extent;
          u -= o->extent;
        }
      } else { /* positive shift */
        l = nl - o->psh[o->dim - 1];
        u = nl - 1;
        if (u < bl) {
          l += o->extent;
          u += o->extent;
        }
      }
      if (l < bl)
        l = bl;
      if (u > bu)
        u = bu;

      n = (u - l + 1);
      if (n <= 0)
        break; /* no more overlaps */

      /* move the data */

      adr = src + (l - clof) * ak;
      cnt = len;
      str = ak;
      if (o->dim <= o->span) {
        cnt *= n;
        str *= n;
        n = 1;
      } else
        u = l;

      for (; n > 0; --n, adr += str) {
#if defined(DEBUG)
        o->lb[o->dim - 1] = l++;
        o->ub[o->dim - 1] = u++;
        if (__fort_test & DEBUG_OLAP) {
          printf("%d olap_send %d ", GET_DIST_LCPU, o->plow + pc * o->pstride);
          for (i = 0; i < F90_RANK_G(as); ++i)
            printf("%c%d:%d", i == 0 ? '(' : ',', o->lb[i], o->ub[i]);
          printf(")[%d]@%x ", cnt, adr);
          __fort_show_scalar(adr, F90_KIND_G(as));
          printf("\n");
        }
        if (adr < o->base || adr + cnt * F90_LEN_G(as) > o->limit)
          __fort_abort("olap_send: bad address");
#endif
        __fort_sendl(o->ch1[o->dim - 1], pc, adr, cnt, 1, F90_KIND_G(as),
                    F90_LEN_G(as));
      }
    } while (pc != o->pcoord);
  }
}

/* receive shifted data from neighbors */

static void I8(olap_recv)(char *dst, F90_Desc *as, olap_sked *o, int len)
{
  DECL_DIM_PTRS(dd);
  char *adr;
  __INT_T ak, bl, bn, bu, cl, cn, cnt, i, l, clof, n, nl, nu, pc, pstep, str,
      tl, tu, u, wrap;

  SET_DIM_PTRS(dd, as, o->dim - 1);

  ak = F90_DPTR_LSTRIDE_G(dd) * F90_LEN_G(as);

  pstep = o->dir;
  if (DIST_DPTR_TSTRIDE_G(dd) < 0)
    pstep = -pstep;

  cl = DIST_DPTR_CL_G(dd);
  cn = DIST_DPTR_CN_G(dd);
  clof = DIST_DPTR_CLOF_G(dd) + DIST_DPTR_LAB_G(dd);

  for (; cn > 0; --cn, cl += DIST_DPTR_CS_G(dd), clof += DIST_DPTR_CLOS_G(dd)) {
    bn = I8(__fort_block_bounds)(as, o->dim, cl, &bl, &bu);

    pc = o->pcoord;

    do {

      /* find next neighbor's processor coordinate */

      pc -= pstep;
      if (pc < 0)
        pc += o->pshape;
      else if (pc >= o->pshape)
        pc -= o->pshape;

      wrap = (pc - o->pcoord) * pstep >= 0;

      if (wrap && o->style == __SHIFT)
        break;

      if (DIST_DPTR_GEN_BLOCK_G(dd)) {

        /* 
         * get neighbor's block bounds from gen_block
         */

        __INT_T *gb, *tgb;
        __INT_T i, direction, tExtent, aExtent;

        nl = DIST_DPTR_TLB_G(dd);
        tExtent = (DIST_DPTR_TUB_G(dd) - nl) + 1;
        aExtent = F90_DPTR_EXTENT_G(dd);

        if (tExtent != aExtent) {

          /* recompute gen_block */

          gb = tgb = I8(__fort_new_gen_block)(as, o->dim - 1);

        } else {
          tgb = 0;
          gb = DIST_DPTR_GEN_BLOCK_G(dd);
        }

        direction = (DIST_DPTR_TSTRIDE_G(dd) < 0);

        if (!direction)
          i = 0;
        else {
          i = (DIST_DPTR_PSHAPE_G(dd)) - 1;
          gb += i;
        }
        nu = *gb + (nl - 1);
        for (; i != pc;) {
          if (!direction) {
            ++i;
            ++gb;
          } else {
            --i;
            --gb;
          }
          nl = nu + 1;
          nu += *gb;
        }

        if (tgb)
          __fort_free(tgb);

        goto check_nl_and_nu;
      }

      /* neighbor's template block bounds */

      tl = DIST_DPTR_TLB_G(dd) + pc * DIST_DPTR_BLOCK_G(dd);
      tu = tl + DIST_DPTR_BLOCK_G(dd) - 1;

      /* neighbor's owned bounds */

      nl = tl - DIST_DPTR_TOFFSET_G(dd);
      nu = tu - DIST_DPTR_TOFFSET_G(dd);
      if (DIST_DPTR_TSTRIDE_G(dd) != 1) {
        if (DIST_DPTR_TSTRIDE_G(dd) < 0)
          i = nl, nl = nu, nu = i;
        if (DIST_DPTR_TSTRIDE_G(dd) == -1) {
          nl = -nl;
          nu = -nu;
        } else {
          nl /= DIST_DPTR_TSTRIDE_G(dd);
          nu /= DIST_DPTR_TSTRIDE_G(dd);
        }
      }
      if (nl < F90_DPTR_LBOUND_G(dd))
        nl = F90_DPTR_LBOUND_G(dd);
      if (nu > DPTR_UBOUND_G(dd))
        nu = DPTR_UBOUND_G(dd);

    check_nl_and_nu:

      if (nl > nu)
        continue; /* neighbor has no data */

      /* find overlap bounds */

      if (o->dir < 0) { /* negative shift */
        l = bu + 1;
        u = bu + o->nsh[o->dim - 1];
        if (l > nu) {
          nl += o->extent;
          nu += o->extent;
        }
      } else { /* positive shift */
        l = bl - o->psh[o->dim - 1];
        u = bl - 1;
        if (u < nl) {
          nl -= o->extent;
          nu -= o->extent;
        }
      }
      if (l < nl)
        l = nl;
      if (u > nu)
        u = nu;

      n = (u - l + 1);
      if (n <= 0)
        break; /* no more overlaps */

      /* move the data */

      adr = dst + (l - clof) * ak;
      cnt = len;
      str = ak;
      if (o->dim <= o->span) {
        cnt *= n;
        str *= n;
        n = 1;
      } else
        u = l;

      for (; n > 0; --n, adr += str) {
#if defined(DEBUG)
        o->lb[o->dim - 1] = l++;
        o->ub[o->dim - 1] = u++;
        if (__fort_test & DEBUG_OLAP) {
          if (wrap && o->style == __EOSHIFT)
            printf("%d olap_fill ", GET_DIST_LCPU);
          else
            printf("%d olap_recv %d ", GET_DIST_LCPU,
                   o->plow + pc * o->pstride);
          for (i = 0; i < F90_RANK_G(as); ++i)
            printf("%c%d:%d", i == 0 ? '(' : ',', o->lb[i], o->ub[i]);
          printf(")[%d]@%x ", cnt, adr);
          __fort_show_scalar(adr, F90_KIND_G(as));
          printf("\n");
        }
        if (adr < o->base || adr + cnt * F90_LEN_G(as) > o->limit)
          __fort_abort("olap_recv: bad address");
#endif
        if (wrap && o->style == __EOSHIFT) {
          __fort_sendl(o->ch2[o->dim - 1], 0, o->boundary, cnt, 0,
                      F90_KIND_G(as), F90_LEN_G(as));
          __fort_recvl(o->ch2[o->dim - 1], 0, adr, cnt, 1, F90_KIND_G(as),
                      F90_LEN_G(as));
        } else
          __fort_recvl(o->ch1[o->dim - 1], pc, adr, cnt, 1, F90_KIND_G(as),
                      F90_LEN_G(as));
      }
    } while (pc != o->pcoord);
  }
}

/* replicate data locally when shift amount > array extent */

static void I8(olap_copy)(char *adr0, F90_Desc *as, olap_sked *o, int len)
{
  DECL_DIM_PTRS(dd);
  char *adr1, *adr2;
  __INT_T ak, m, n, nl, nu, l, u;
#if defined(DEBUG)
  __INT_T i;
#endif

  SET_DIM_PTRS(dd, as, o->dim - 1);

  ak = F90_DPTR_LSTRIDE_G(dd) * F90_LEN_G(as);

  if (DIST_DPTR_OLB_G(dd) > DIST_DPTR_OUB_G(dd))
    return; /* no local data */

  if (o->dir < 0) {
    l = nl = DIST_DPTR_UAB_G(dd) - DIST_DPTR_PO_G(dd) + 1;
    m = o->nsh[o->dim - 1];
  } else {
    u = nu = DIST_DPTR_LAB_G(dd) + DIST_DPTR_NO_G(dd) - 1;
    m = o->psh[o->dim - 1];
  }

  for (m -= o->extent; m > 0; m -= o->extent) {

    n = Min(m, o->extent);

    if (o->dir < 0) {
      nu = nl + n - 1;
      l += o->extent;
      u = l + n - 1;
    } else {
      nl = nu - n + 1;
      u -= o->extent;
      l = u - n + 1;
    }

    n *= len;
    adr1 = adr0 + (nl - DIST_DPTR_LAB_G(dd)) * ak;
    adr2 = adr0 + (l - DIST_DPTR_LAB_G(dd)) * ak;
#if defined(DEBUG)
    o->lb[o->dim - 1] = l;
    o->ub[o->dim - 1] = u;
    if (__fort_test & DEBUG_OLAP) {
      printf("%d olap_copy ", GET_DIST_LCPU);
      for (i = 0; i < F90_RANK_G(as); ++i)
        printf("%c%d:%d", i == 0 ? '(' : ',', o->lb[i], o->ub[i]);
      printf(")[%d]@%x = ", n, adr2);
      o->lb[o->dim - 1] = nl;
      o->ub[o->dim - 1] = nu;
      for (i = 0; i < F90_RANK_G(as); ++i)
        printf("%c%d:%d", i == 0 ? '(' : ',', o->lb[i], o->ub[i]);
      printf(")@%x ", adr1);
      __fort_show_scalar(adr1, F90_KIND_G(as));
      printf("\n");
    }
    if (adr1 < o->base || adr1 + n * F90_LEN_G(as) > o->limit ||
        adr2 < o->base || adr2 + n * F90_LEN_G(as) > o->limit)
      __fort_abort("olap_recv: bad address");
#endif
    __fort_sendl(o->ch3[o->dim - 1], 0, adr1, n, 1, F90_KIND_G(as),
                F90_LEN_G(as));
    __fort_recvl(o->ch3[o->dim - 1], 0, adr2, n, 1, F90_KIND_G(as),
                F90_LEN_G(as));
  }
}

static void I8(olap_loop)(char *adr0, F90_Desc *as, olap_sked *o, int len,
                          int dim)
{
  DECL_DIM_PTRS(dd);
  char *adr1;
  __INT_T ak, bl, bn, bu, cl, cn, clof;

  if (dim == o->dim)
    --dim;

  if (dim < 1) {
    I8(olap_send)(adr0, as, o, len);
    I8(olap_recv)(adr0, as, o, len);
    I8(olap_copy)(adr0, as, o, len);
    return;
  }

  SET_DIM_PTRS(dd, as, dim - 1);

  ak = F90_DPTR_LSTRIDE_G(dd) * F90_LEN_G(as);

  cl = DIST_DPTR_CL_G(dd);
  cn = DIST_DPTR_CN_G(dd);
  clof = DIST_DPTR_CLOF_G(dd) + DIST_DPTR_LAB_G(dd);

  for (; cn > 0; --cn, cl += DIST_DPTR_CS_G(dd), clof += DIST_DPTR_CLOS_G(dd)) {

    if (cn > 1 && /* optimize cyclic cases */
        (DIST_DPTR_CLOS_G(dd) == 0 || (dim < o->dim && o->fullcycle[dim - 1]))) {
      bl = DIST_DPTR_LAB_G(dd) + DIST_DPTR_NO_G(dd);
      bu = DIST_DPTR_UAB_G(dd) - DIST_DPTR_PO_G(dd);
      bn = bu - bl + 1;
      cn = 1;                    /* short-circuit cyclic loop */
      clof = DIST_DPTR_LAB_G(dd); /* cyclic indices localized */
    } else
      bn = I8(__fort_block_bounds)(as, dim, cl, &bl, &bu);

    if (bn <= 0)
      continue; /* no local data */

    if (dim < o->dim) { /* include data shifted earlier */
      if (dim > o->span) {
        bl -= o->psh[dim - 1];
        bu += o->nsh[dim - 1];
      } else { /* span this dimension */
        bl = DIST_DPTR_LAB_G(dd);
        bu = DIST_DPTR_UAB_G(dd);
        cn = 1;                    /* short-circuit cyclic loop */
        clof = DIST_DPTR_LAB_G(dd); /* cyclic indices localized */
      }
      bn = bu - bl + 1;
    }
    adr1 = adr0 + (bl - clof) * ak;
    if (dim > o->span) {
      for (; bn > 0; --bn, adr1 += ak) {
#if defined(DEBUG)
        o->lb[dim - 1] = o->ub[dim - 1] = bl++;
#endif
        I8(olap_loop)(adr1, as, o, len, dim - 1);
      }
    } else {
#if defined(DEBUG)
      o->lb[dim - 1] = bl;
      o->ub[dim - 1] = bu;
#endif
      I8(olap_loop)(adr1, as, o, len * bn, dim - 1);
    }
  }
}

static sked *I8(olap_shift)(char *ab, F90_Desc *as, olap_sked *o)
{
  DECL_DIM_PTRS(dd);
  __INT_T i, abstr, lcpu, maxb, maxc;

  I8(__fort_cycle_bounds)(as);

  o->sked.tag = __SKED;
  o->sked.start = I8(olap_start);
  o->sked.free = olap_free;
  o->sked.arg = o;

  o->rank = F90_RANK_G(as);
  o->span = 0;

#if defined(DEBUG)
  o->base = ab;
  o->limit = ab + F90_LSIZE_G(as) * F90_LEN_G(as);
#endif

  lcpu = GET_DIST_LCPU;

  for (i = 0; i < o->rank; ++i) {
    SET_DIM_PTRS(dd, as, i);

    if (DIST_DPTR_GEN_BLOCK_G(dd)) {

      if (strncmp(GET_DIST_TRANSNAM, "smp", 3) == 0) {

        /* 
         * find max block based on gen_block.
         * (max block is defined as the largest element in the
         *  genblock array when using -SMP or using T3E) ...
         */
        __INT_T *gb, *tgb;
        __INT_T i, tExtent, aExtent, pshape;

        tExtent = (DIST_DPTR_TUB_G(dd) - DIST_DPTR_TLB_G(dd)) + 1;
        aExtent = F90_DPTR_EXTENT_G(dd);

        if (tExtent != aExtent) {

          /* recompute gen_block */

          gb = tgb = I8(__fort_new_gen_block)(as, o->dim - 1);

        } else {
          tgb = 0;
          gb = DIST_DPTR_GEN_BLOCK_G(dd);
        }

        gb = DIST_DPTR_GEN_BLOCK_G(dd);
        maxb = *gb;
        pshape = DIST_DPTR_PSHAPE_G(dd);
        for (i = 0; i < pshape; i++) {
          ++gb;
          if (*gb > maxb) {
            maxb = *gb;
          }
        }

        if (tgb)
          __fort_free(tgb);

        abstr = Abs(DIST_DPTR_TSTRIDE_G(dd));
        maxc = DIST_DPTR_CYCLE_G(dd) + abstr - 1;

      } else {

        /* asymmetrical memory allocation, no optimization
         * can be performed.
         */

        o->fullcycle[i] = 0;
        continue;
      }

    } else {

      abstr = Abs(DIST_DPTR_TSTRIDE_G(dd));
      maxb = DIST_DPTR_BLOCK_G(dd) + abstr - 1;
      maxc = DIST_DPTR_CYCLE_G(dd) + abstr - 1;
      if (abstr != 1) {
        maxb /= abstr;
        maxc /= abstr;
      }
    }
    o->fullcycle[i] = (maxb + o->nsh[i] + o->psh[i] >= maxc);
  }

  for (i = 0; i < o->rank; ++i) {
    o->dim = i + 1;
    o->ch1[i] = o->ch2[i] = o->ch3[i] = NULL;

    SET_DIM_PTRS(dd, as, i);

#if defined(DEBUG)
    if (__fort_test & DEBUG_OLAP)
      printf("%d olap dim=%d nsh=%d psh=%d po=%d no=%d\n", GET_DIST_LCPU,
             o->dim, o->nsh[i], o->psh[i], DIST_DPTR_PO_G(dd),
             DIST_DPTR_NO_G(dd));
#endif

    /* A lower dimension (below the one being shifted) can be
       spanned (transferred all at once) if consecutive elements
       are contiguous and the shift amounts for that dimension and
       all lower dimensions are equal to their respective overlap
       allowances. */

    if (o->span == i && DIST_DPTR_COFSTR_G(dd) == 0 &&
        o->psh[i] == DIST_DPTR_NO_G(dd) && o->nsh[i] == DIST_DPTR_PO_G(dd))
      o->span = o->dim;

    if (o->nsh[i] == 0 && o->psh[i] == 0)
      continue; /* nothing to do */

    if (o->nsh[i] > DIST_DPTR_PO_G(dd) || o->psh[i] > DIST_DPTR_NO_G(dd))
      __fort_abort("olap_shift: shift amount exceeds allowance");

    if (DIST_DPTR_PAXIS_G(dd) > 0) {
      o->pcoord = DIST_DPTR_PCOORD_G(dd);
      o->pshape = DIST_DPTR_PSHAPE_G(dd);
      o->pstride = DIST_DPTR_PSTRIDE_G(dd);
    } else {
      o->pcoord = 0;
      o->pshape = 1;
      o->pstride = 1;
    }
    o->plow = lcpu - o->pcoord * o->pstride;

    o->ch1[i] = __fort_chn_1to1(NULL, 1, o->plow, &o->pshape, &o->pstride, 1,
                               o->plow, &o->pshape, &o->pstride);

    /* channel for filling boundary values */

    if (o->style == __EOSHIFT)
      o->ch2[i] =
          __fort_chn_1to1(NULL, 0, lcpu, NULL, NULL, 0, lcpu, NULL, NULL);

    /* channel for replicating data when shift > array extent */

    o->extent = F90_DPTR_EXTENT_G(dd);
    if (o->extent < 0)
      o->extent = 0;
    if (o->nsh[i] > o->extent || o->psh[i] > o->extent)
      o->ch3[i] =
          __fort_chn_1to1(NULL, 0, lcpu, NULL, NULL, 0, lcpu, NULL, NULL);

    /* negative (left) shift */

    if (o->nsh[i] > 0 && ~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
      o->dir = -1;
      I8(olap_loop)(ab, as, o, 1, o->rank);
    }

    /* positive (right) shift */

    if (o->psh[i] > 0 && ~F90_FLAGS_G(as) & __OFF_TEMPLATE) {
      o->dir = 1;
      I8(olap_loop)(ab, as, o, 1, o->rank);
    }

    __fort_chn_prune(o->ch1[i]);
    __fort_chn_prune(o->ch2[i]);
    __fort_chn_prune(o->ch3[i]);

    __fort_setbase(o->ch1[i], ab, ab, F90_KIND_G(as), F90_LEN_G(as));
    __fort_setbase(o->ch2[i], (void *)o->boundary, ab, F90_KIND_G(as),
                  F90_LEN_G(as));
    __fort_setbase(o->ch3[i], ab, ab, F90_KIND_G(as), F90_LEN_G(as));
  }
  return &o->sked;
}

sked *ENTFTN(OLAP_SHIFT, olap_shift)(void *ab, F90_Desc *as, ...)
{
  olap_sked *o;
  int i;
  va_list va;

  if (!ISPRESENT(ab))
    __fort_abort("olap_shift: array absent or not allocated");

#if defined(DEBUG)
  if (as == NULL || F90_TAG_G(as) != __DESC)
    __fort_abort("olap_shift: invalid descriptor");
#endif

  o = (olap_sked *)__fort_malloc(sizeof(olap_sked));

  va_start(va, as);
  for (i = 0; i < F90_RANK_G(as); ++i) {
    o->nsh[i] = *va_arg(va, __INT_T *);
    o->psh[i] = *va_arg(va, __INT_T *);
  }
  va_end(va);
  o->style = __SHIFT;
  return I8(olap_shift)(ab, as, o);
}

sked *ENTFTN(OLAP_CSHIFT, olap_cshift)(void *ab, F90_Desc *as, ...)
{
  olap_sked *o;
  int i;
  va_list va;

  if (!ISPRESENT(ab))
    __fort_abort("olap_cshift: array absent or not allocated");

#if defined(DEBUG)
  if (as == NULL || F90_TAG_G(as) != __DESC)
    __fort_abort("olap_cshift: invalid descriptor");
#endif

  o = (olap_sked *)__fort_malloc(sizeof(olap_sked));

  va_start(va, as);
  for (i = 0; i < F90_RANK_G(as); ++i) {
    o->nsh[i] = *va_arg(va, __INT_T *);
    o->psh[i] = *va_arg(va, __INT_T *);
  }
  va_end(va);
  o->style = __CSHIFT;
  return I8(olap_shift)(ab, as, o);
}

sked *ENTFTN(OLAP_EOSHIFT, olap_eoshift)(void *ab, F90_Desc *as, void *boundary,
                                         ...)
{
  olap_sked *o;
  int i;
  va_list va;

  if (!ISPRESENT(ab))
    __fort_abort("olap_eoshift: array absent or not allocated");

#if defined(DEBUG)
  if (as == NULL || F90_TAG_G(as) != __DESC)
    __fort_abort("olap_eoshift: invalid descriptor");
#endif

  o = (olap_sked *)__fort_malloc(sizeof(olap_sked));

  va_start(va, boundary);
  for (i = 0; i < F90_RANK_G(as); ++i) {
    o->nsh[i] = *va_arg(va, __INT_T *);
    o->psh[i] = *va_arg(va, __INT_T *);
  }
  va_end(va);
  o->style = __EOSHIFT;
  if (!ISPRESENT(boundary))
    boundary = GET_DIST_ZED;
  __fort_scalar_copy[F90_KIND_G(as)](o->boundary, boundary, F90_LEN_G(as));
  return I8(olap_shift)(ab, as, o);
}

sked *ENTFTN(COMM_SHIFT, comm_shift)(void *ab, F90_Desc *as, ...)
{
  olap_sked *o;
  int i;
  va_list va;

  if (!ISPRESENT(ab))
    __fort_abort("comm_shift: array absent or not allocated");

#if defined(DEBUG)
  if (as == NULL || F90_TAG_G(as) != __DESC)
    __fort_abort("comm_shift: invalid descriptor");
#endif

  o = (olap_sked *)__fort_malloc(sizeof(olap_sked));

  va_start(va, as);
  for (i = 0; i < F90_RANK_G(as); ++i) {
    o->nsh[i] = *va_arg(va, __INT_T *);
    o->psh[i] = *va_arg(va, __INT_T *);
  }
  va_end(va);
  o->style = __CSHIFT;
  return I8(olap_shift)(ab, as, o);
}
