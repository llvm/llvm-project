/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* FIXME: how much (if any) of this is used/needed */

/** \file
 * \brief Scatter reduction routines
 *
 * call ENTFTN(xxx_scatter)(rb, rs, mb, ms, bb, bs, ...)
 * where xxx = all, any, count, parity
 *
 * call ENTFTN(yyy_scatter)(rb, rs, ab, as, bb, bs, mb, ms, ...)
 * where yyy = copy, iall, iany, iparity, maxval, minval, product, sum
 *
 * rb = result base address
 * rs = result descriptor
 * ab = array base address
 * as = array descriptor
 * bb = base base address
 * bs = base descriptor
 * mb = mask base address
 * ms = mask descriptor
 *
 * ... = { xb, xs, }* = base address and descriptors for the indexes,
 * one pair for each result dimension.
 *
 * indexes must not be scalar, instead the result section may have a
 * scalar subscript.  Index values must be within the bounds of the
 * corresponding result dimension.
 *
 * indexes must conform with array.  Each dimension of every index
 * must be either 1) replicated or 2) aligned with the corresponding
 * dimension of array.
 *
 * mask must conform with array, may be scalar, and is assumed to be
 * of type logical*4.  Each dimension of mask must be either
 * replicated or aligned with the corresponding dimension of array.
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "fort_vars.h"
#include "scatter.h"

extern double __fort_second();

/* un-permuted axis map */

static __INT_T id_map[MAXDIMS] = {1, 2, 3, 4, 5, 6, 7};

static void
gathscat_abort(const char *what, const char *msg)
{
  char str[80];
  sprintf(str, "%s: %s", what, msg);
  __fort_abort(str);
}

/* gather-scatter schedule structure */

typedef struct {
  sked sked;            /* schedule header */
  const char *what;     /* "GATHER"/"XXX_SCATTER" */
  gathscatfn_t gathscatfn; /* local gather-scatter-reduction function */
  scatterfn_t scatterfn;   /* local scatter-reduction function */
  chdr *repchn;         /* replication channel */
  int *countbuf, *offsetbuf;
  int *countr; /* incoming counts per target */
  int *counts; /* outgoing counts per target */
  int *goff;   /* gather offsets */
  int *soff;   /* scatter offsets */
  int lclcnt;  /* number of local elements */
  int maxcnt;  /* maximum send/receive count */
} gathscat_sked;

/* ENTFTN(comm_start) function: gather, transfer, scatter */

static void I8(gathscat_start)(void *skp, char *rb, char *sb, F90_Desc *rd,
                               F90_Desc *sd)
{
  gathscat_sked *sk = (gathscat_sked *)skp;
  char *rp, *sp, *bufr, *bufs;
  int cpu, j, k, lcpu, m, nr, ns, tcpus;

  double t;
  if (__fort_test & DEBUG_TIME)
    t = __fort_second();

  rp = rb + DIST_SCOFF_G(rd) * F90_LEN_G(rd);
  sp = sb + DIST_SCOFF_G(sd) * F90_LEN_G(sd);

  /* handle local elements first */

  j = k = sk->lclcnt;
  if (k > 0) {

    /* 
     * make sure base type of object hasn't changed for local gather...
     * This can occur when we share schedules across objects ...
     */
    /* Couldn't find where gathscatfn is set to local_gathscat_WRAPPER.
       Possible dead code? Work around the compiler warning with an
       explicit cast for now. */
    if (sk->gathscatfn == (gathscatfn_t)local_gathscat_WRAPPER) {

      local_gathscat_WRAPPER(k, rp, sk->soff, sp, sk->goff, F90_KIND_G(rd));

    } else {
      sk->gathscatfn(k, rp, sk->soff, sp, sk->goff);
    }
  }
  /* now non-local elements */

  lcpu = GET_DIST_LCPU;
  if (sk->maxcnt > 0) {

    /* allocate buffers for aggregated elements */

    bufr = (char *)__fort_gmalloc(2 * sk->maxcnt * F90_LEN_G(sd));
    bufs = bufr + sk->maxcnt * F90_LEN_G(sd);

    /* exchange targets are chosen by xor'ing this processor number
       with all masks in the range 1 .. 2**log2(__fort_tcpus).  Target
       values >= tcpus are skipped (which occurs only if tcpus is not
       a power of two.) */

    tcpus = GET_DIST_TCPUS;
    for (m = 1; m < __fort_np2; ++m) {

      /* identify target */

      cpu = lcpu ^ m;
      if (cpu >= tcpus)
        continue;

      nr = sk->countr[cpu];
      ns = sk->counts[cpu];

      /* exchange elements */

      if (ns > 0)
        local_gather_WRAPPER(ns, bufs, sp, sk->goff + j, F90_KIND_G(sd));

      if (cpu < lcpu) {
        if (nr > 0)
          __fort_rrecvl(cpu, bufr, nr, 1, F90_KIND_G(sd), F90_LEN_G(sd));
        if (ns > 0)
          __fort_rsendl(cpu, bufs, ns, 1, F90_KIND_G(sd), F90_LEN_G(sd));
      } else {
        if (ns > 0)
          __fort_rsendl(cpu, bufs, ns, 1, F90_KIND_G(sd), F90_LEN_G(sd));
        if (nr > 0)
          __fort_rrecvl(cpu, bufr, nr, 1, F90_KIND_G(sd), F90_LEN_G(sd));
      }

      if (nr > 0) {

        /* 
         * Added call to wrapper routine for local scatter
         * in order to handle cases where the schedule is
         * shared between objects of different types...
         */
        /* Couldn't find where gathscatfn is set to local_gathscat_WRAPPER.
           Possible dead code? Work around the compiler warning with an
           explicit cast for now. */
        if (sk->scatterfn == (scatterfn_t)local_scatter_WRAPPER) {
          local_scatter_WRAPPER(nr, rp, sk->soff + k, bufr, F90_KIND_G(rd));
        } else {
          sk->scatterfn(nr, rp, sk->soff + k, bufr);
        }
      }

      j += ns;
      k += nr;
    }

    __fort_gfree(bufr);
  }

  /* replicate result if needed */

  if (sk->repchn) {
    __fort_adjbase(sk->repchn, rp, rp, F90_KIND_G(rd), F90_LEN_G(rd));
    __fort_doit(sk->repchn);
  }

  if (__fort_test & DEBUG_TIME) {
    t = __fort_second() - t;
    printf("%d %s execute %.6f\n", lcpu, sk->what, t);
  }
}

/* ENTFTN(comm_free) function: free channels, vectors, schedule structures */

static void
gathscat_free(void *skp)
{
  gathscat_sked *sk = (gathscat_sked *)skp;
  if (sk->repchn)
    __fort_frechn(sk->repchn);
  if (sk->countbuf)
    __fort_gfree(sk->countbuf);
  if (sk->offsetbuf)
    __fort_free(sk->offsetbuf);
  __fort_free(sk);
}

/* local setup for each element of unvectored array */

static void I8(gathscat_element)(gathscat_parm *z, __INT_T uoff, __INT_T xoff[])
{
  gathscat_dim *zd;
  DECL_HDR_PTRS(ud);
  DECL_HDR_PTRS(vd);
  int cpu, k, ux, vx;
  __INT_T roff;
  __INT_T vi[MAXDIMS];

  ud = z->ud;
  vd = z->vd;

  /* construct vectored array index tuple */

  for (vx = F90_RANK_G(vd); --vx >= 0;) {
    zd = &z->dim[vx];
    if (z->indirect >> vx & 1)
      vi[vx] = zd->xb[xoff[vx]];
    else {
      ux = *zd->xmap;
      vi[vx] = z->ui[ux - 1];
    }
  }

#if defined(DEBUG)
  if (__fort_test & DEBUG_SCAT) {
    for (vx = 0; vx < F90_RANK_G(vd); ++vx) {
      zd = &z->dim[vx];
      if (z->indirect >> vx & 1)
        printf("%d vx %d xoff[vx] %d vi[vx] %d\n", GET_DIST_LCPU, vx, xoff[vx],
               vi[vx]);
      else
        printf("%d vx %d ux %d vi[vx] %d\n", GET_DIST_LCPU, vx, *zd->xmap,
               vi[vx]);
    }
  }
#endif

  k = ++z->outgoing; /* count outgoing elements */

  if (z->communicate | z->replicate) {

    /* identify target processor and element offset, count
       elements for target processor */

    I8(__fort_localize)(vd, vi, &cpu, &roff);
    cpu += z->group_offset;
    ++z->counts[cpu];

    /* link to target cpu's list.  using a linked list reverses
       the order of elements processed */

    z->next[k - 1] = z->head[cpu];
    z->head[cpu] = k;
  } else {
    cpu = GET_DIST_LCPU;
    roff = I8(__fort_local_offset)(vd, vi);
  }

  /* store offsets in local and remote arrays */

  z->loff[k - 1] = uoff - DIST_SCOFF_G(ud);
  z->roff[k - 1] = roff;

#if defined(DEBUG)
  if (__fort_test & DEBUG_SCAT) {
    printf("%d %s remote cpu %d r", GET_DIST_LCPU, z->what, cpu);
    if (z->dir == __SCATTER) {
      I8(__fort_show_index)(F90_RANK_G(vd), vi);
      printf("@%d = a", roff);
    }
    I8(__fort_show_index)(F90_RANK_G(ud), z->ui);
    printf("@%d ", uoff);
    if (z->dir == __GATHER) {
      printf("= a");
      I8(__fort_show_index)(F90_RANK_G(vd), vi);
      printf("@%d", roff);
    } else
      __fort_print_scalar(z->ub + uoff * F90_LEN_G(ud), F90_KIND_G(ud));
    printf("\n");
  }
#endif
}

/* loop over local elements of unvectored array, checking mask */

static void
    I8(gathscat_mask_loop)(gathscat_parm *z, /* parameters */
                           int uoff0,      /* unvectored array element offset */
                           __INT_T xoff[], /* index array element offsets */
                           int dim)        /* unvectored array/mask dimension */
{
  DECL_HDR_PTRS(ud);
  DECL_DIM_PTRS(udd);
  DECL_F90_DIM_PTR(xdd);
  DECL_DIST_DIM_PTR(xdd);
  xstuff *x;
  __LOG_T *mb;
  __LOG_T mask_log;
  __INT_T n, ubl, ubu, ubn, ucl, ucn, uclof, uoff;

  ud = z->ud;
  SET_DIM_PTRS(udd, ud, dim - 1);
  ucn = DIST_DPTR_CN_G(udd);
  ucl = DIST_DPTR_CL_G(udd);
  uclof = DIST_DPTR_CLOF_G(udd);
  ubn = 0;

  mb = (__LOG_T *)z->mb;

  for (x = z->xhead[dim - 1]; x; x = x->next) {
    if (z->aligned_x_u >> x->vx & 1) {
      F90_DIM_NAME(xdd) = x->F90_DIM_NAME(xdd);
      DIST_DIM_NAME(xdd) = x->DIST_DIM_NAME(xdd);
      x->cn = DIST_DPTR_CN_G(xdd);
      x->cl = DIST_DPTR_CL_G(xdd);
      x->cs = DIST_DPTR_CS_G(xdd);
      x->clof = DIST_DPTR_CLOF_G(xdd);
      x->clos = DIST_DPTR_CLOS_G(xdd);
    } else
      x->cn = 0;
    x->bn = 0;
    x->off0 = xoff[x->vx]; /* save initial offset */
  }

  mask_log = GET_DIST_MASK_LOG;
  while (ubn > 0 || ucn > 0) {
    if (ubn == 0) {
      ubn = I8(__fort_block_bounds)(ud, dim, ucl, &ubl, &ubu);
      uoff = uoff0 +
             (F90_DPTR_SSTRIDE_G(udd) * ubl - uclof) * F90_DPTR_LSTRIDE_G(udd);
      ucl += DIST_DPTR_CS_G(udd);
      uclof += DIST_DPTR_CLOS_G(udd);
      --ucn;
      z->ui[dim - 1] = ubl; /* unvectored array index */
    }
    n = ubn;
    for (x = z->xhead[dim - 1]; x; x = x->next) {
      if (x->bn == 0) { /* block exhausted */
        __INT_T xbl, xbu;

        F90_DIM_NAME(xdd) = x->F90_DIM_NAME(xdd);
        DIST_DIM_NAME(xdd) = x->DIST_DIM_NAME(xdd);
        if (z->aligned_x_u >> x->vx & 1) {

          /* mask/index aligned with unvectored. local
             blocks should synchronize with unvectored array */

          if (x->cn <= 0)
            gathscat_abort(z->what, "index misalignment");
          x->bn = I8(__fort_block_bounds)(x->xd, x->xx + 1, x->cl, &xbl, &xbu);
        } else {

          /* mask/index not aligned with unvectored (but
             local wherever unvectored is local). need to
             set up loop bounds corresponding to each new
             unvectored block */

          __INT_T xl = F90_DPTR_LBOUND_G(xdd) + ubl - F90_DPTR_LBOUND_G(udd);
          __INT_T xu = xl + ubn - 1;

          if (x->cn <= 0) {
            __INT_T xcu;
            x->cn = I8(__fort_cyclic_loop)(x->xd, x->xx + 1, xl, xu, 1, &x->cl,
                                          &xcu, &x->cs, &x->clof, &x->clos);
          }
          x->bn = I8(__fort_block_loop)(x->xd, x->xx + 1, xl, xu, 1, x->cl, &xbl,
                                       &xbu);
        }
        xoff[x->vx] =
            x->off0 +
            (F90_DPTR_SSTRIDE_G(xdd) * xbl - x->clof) * F90_DPTR_LSTRIDE_G(xdd);
        x->cl += x->cs;
        x->clof += x->clos;
        --x->cn;
      }
      if (x->bn < n)
        n = x->bn;
    }
    ubl += n;
    ubn -= n;
    for (x = z->xhead[dim - 1]; x; x = x->next)
      x->bn -= n;
    while (--n >= 0) {
      if (dim > 1)
        I8(gathscat_mask_loop)(z, uoff, xoff, dim - 1);
      else if (mb[xoff[MAXDIMS]] & mask_log)
        I8(gathscat_element)(z, uoff, xoff);
      uoff += F90_DPTR_SSTRIDE_G(udd) * F90_DPTR_LSTRIDE_G(udd);
      for (x = z->xhead[dim - 1]; x; x = x->next)
        xoff[x->vx] += x->str;
      ++z->ui[dim - 1];
    }
  }
  for (x = z->xhead[dim - 1]; x; x = x->next)
    xoff[x->vx] = x->off0; /* restore offset */
}

/* loop over local elements of unvectored array, no mask */

static void I8(gathscat_loop)(gathscat_parm *z, /* parameters */
                              int uoff0, /* unvectored array element offset */
                              __INT_T xoff[], /* index array element offsets */
                              int dim)        /* unvectored array dimension */
{
  DECL_HDR_PTRS(ud);
  DECL_DIM_PTRS(udd);
  DECL_F90_DIM_PTR(xdd);
  DECL_DIST_DIM_PTR(xdd);
  xstuff *x;
  __INT_T n, ubl, ubu, ubn, ucl, ucn, uclof, uoff;

  ud = z->ud;
  SET_DIM_PTRS(udd, ud, dim - 1);
  ucn = DIST_DPTR_CN_G(udd);
  ucl = DIST_DPTR_CL_G(udd);
  uclof = DIST_DPTR_CLOF_G(udd);
  ubn = 0;

  for (x = z->xhead[dim - 1]; x; x = x->next) {
    if (z->aligned_x_u >> x->vx & 1) {
      F90_DIM_NAME(xdd) = x->F90_DIM_NAME(xdd);
      DIST_DIM_NAME(xdd) = x->DIST_DIM_NAME(xdd);
      x->cn = DIST_DPTR_CN_G(xdd);
      x->cl = DIST_DPTR_CL_G(xdd);
      x->cs = DIST_DPTR_CS_G(xdd);
      x->clof = DIST_DPTR_CLOF_G(xdd);
      x->clos = DIST_DPTR_CLOS_G(xdd);
    } else
      x->cn = 0;
    x->bn = 0;
    x->off0 = xoff[x->vx]; /* save initial offset */
  }

  while (ubn > 0 || ucn > 0) {
    if (ubn == 0) {
      ubn = I8(__fort_block_bounds)(ud, dim, ucl, &ubl, &ubu);
      uoff = uoff0 +
             (F90_DPTR_SSTRIDE_G(udd) * ubl - uclof) * F90_DPTR_LSTRIDE_G(udd);
      ucl += DIST_DPTR_CS_G(udd);
      uclof += DIST_DPTR_CLOS_G(udd);
      --ucn;
      z->ui[dim - 1] = ubl; /* unvectored array index */
    }
    n = ubn;
    for (x = z->xhead[dim - 1]; x; x = x->next) {
      if (x->bn == 0) { /* block exhausted */
        __INT_T xbl, xbu;

        F90_DIM_NAME(xdd) = x->F90_DIM_NAME(xdd);
        DIST_DIM_NAME(xdd) = x->DIST_DIM_NAME(xdd);
        if (z->aligned_x_u >> x->vx & 1) {

          /* index aligned with unvectored. local blocks
             should synchronize with unvectored array */

          if (x->cn <= 0)
            gathscat_abort(z->what, "index misalignment");
          x->bn = I8(__fort_block_bounds)(x->xd, x->xx + 1, x->cl, &xbl, &xbu);
        } else {

          /* index not aligned with unvectored (but local
             wherever unvectored is local). need to set up
             loop bounds corresponding to each new
             unvectored block */

          __INT_T xl = F90_DPTR_LBOUND_G(xdd) + ubl - F90_DPTR_LBOUND_G(udd);
          __INT_T xu = xl + ubn - 1;

          if (x->cn <= 0) {
            __INT_T xcu;
            x->cn = I8(__fort_cyclic_loop)(x->xd, x->xx + 1, xl, xu, 1, &x->cl,
                                          &xcu, &x->cs, &x->clof, &x->clos);
          }
          x->bn = I8(__fort_block_loop)(x->xd, x->xx + 1, xl, xu, 1, x->cl, &xbl,
                                       &xbu);
        }
        xoff[x->vx] =
            x->off0 +
            (F90_DPTR_SSTRIDE_G(xdd) * xbl - x->clof) * F90_DPTR_LSTRIDE_G(xdd);
        x->cl += x->cs;
        x->clof += x->clos;
        --x->cn;
      }
      if (x->bn < n)
        n = x->bn;
    }
    ubl += n;
    ubn -= n;
    for (x = z->xhead[dim - 1]; x; x = x->next)
      x->bn -= n;
    while (--n >= 0) {
      if (dim > 1)
        I8(gathscat_loop)(z, uoff, xoff, dim - 1);
      else
        I8(gathscat_element)(z, uoff, xoff);
      uoff += F90_DPTR_SSTRIDE_G(udd) * F90_DPTR_LSTRIDE_G(udd);
      for (x = z->xhead[dim - 1]; x; x = x->next)
        xoff[x->vx] += x->str;
      ++z->ui[dim - 1];
    }
  }
  for (x = z->xhead[dim - 1]; x; x = x->next)
    xoff[x->vx] = x->off0; /* restore offset */
}

/* u = unvectored, v = vectored, x = index */

sked *I8(__fort_gathscat)(gathscat_parm *z)
{
  gathscat_sked *sk;
  gathscat_dim *zd;
  char *rp;
  chdr *repchn;
  DECL_HDR_PTRS(md);
  DECL_HDR_PTRS(ud);
  DECL_HDR_PTRS(vd);
  DECL_HDR_PTRS(rd);
  DECL_HDR_PTRS(xd);
  DECL_F90_DIM_PTR(mdd);
  DECL_DIST_DIM_PTR(mdd) = NULL;
  DECL_DIM_PTRS(udd);
  DECL_F90_DIM_PTR(xdd);
  DECL_DIST_DIM_PTR(xdd) = NULL;
  xstuff *x;

  int *countbuf, *countr, *counts, *goff, *head, *loff, *next, *offr, *offs,
      *offsetbuf, *roff, *soff;

  int alike, cpu, different, incoming, i, j, k, m, n, nr, ns;
  int lclcnt, lcpu, maxcnt, tempz, tcpus, u_covers_v, v_covers_u;
#if defined(DEBUG)
  int has_gb = 0;
#endif
  __INT_T mx, uoff, ux, vx, xx, xoff[MAXDIMS + 1];

  repl_t u_repl; /* unvectored array replication descriptor */

  double t;
  if (__fort_test & DEBUG_TIME)
    t = __fort_second();

  md = z->md;
  ud = z->ud;
  vd = z->vd;
  rd = z->rd;

  lcpu = GET_DIST_LCPU;

#if defined(DEBUG)
  if (z->gathscatfn == NULL || z->scatterfn == NULL)
    gathscat_abort(z->what, "unsupported data type");
#endif

  /* initial bit masks */

  z->conform_x_u = 0; /* index conforms with unvectored */
  z->aligned_x_u = 0; /* index aligned with unvectored */
  z->aligned_v_u = 0; /* vectored aligned with unvectored */
  z->aligned_u_v = 0; /* unvectored aligned with vectored */

  if (~z->indirect & ~(-1 << F90_RANK_G(vd))) {
    /* if any vectored dim is not indirectly indexed... */
    u_covers_v = I8(__fort_covers_procs)(ud, vd);
    v_covers_u = I8(__fort_covers_procs)(vd, ud);
  }

  /* initialize inverse mapping list of mask and index dimensions
     subscripted by each unvectored axis */

  uoff = F90_LBASE_G(ud) - 1;
  for (ux = F90_RANK_G(ud); --ux >= 0;) {
    SET_DIM_PTRS(udd, ud, ux);
    uoff += F90_DPTR_SOFFSET_G(udd) * F90_DPTR_LSTRIDE_G(udd);
    z->xhead[ux] = NULL; /* empty xlists */
  }
  z->xfree = z->xlist; /* first free xlist entry */

  /* check mask -- finished already if mask is scalar .false. */

  if (F90_TAG_G(md) == __DESC) {
    if (F90_KIND_G(md) != __LOG)
      gathscat_abort(z->what, "mask array must be logical");

    /* treat mask as an extra index array for address calculations  */

    I8(__fort_cycle_bounds)(md);
    xoff[MAXDIMS] = F90_LBASE_G(md) - 1;
    for (mx = F90_RANK_G(md); --mx >= 0;) {
      SET_DIM_PTRS(mdd, md, mx);
      xoff[MAXDIMS] += F90_DPTR_SOFFSET_G(mdd) * F90_DPTR_LSTRIDE_G(mdd);

      /* add mask dimension to xlist for the corresponding
         unvectored dimension. mask axis permutation is not
         supported, but the mask rank may be less than the
         unvectored array rank */

      x = z->xfree++;
      x->next = z->xhead[mx];
      z->xhead[mx] = x;
      x->xd = md;
      SET_DIM_PTRS(mdd, md, mx);
      x->F90_DIM_NAME(xdd) = F90_DIM_NAME(mdd);
      x->DIST_DIM_NAME(xdd) = DIST_DIM_NAME(mdd);
      x->str = F90_DPTR_SSTRIDE_G(mdd) * F90_DPTR_LSTRIDE_G(mdd);
      x->vx = MAXDIMS;
      x->xx = mx;
    }
    z->conform_x_u |= I8(__fort_conform)(md, id_map, ud, id_map) << MAXDIMS;
    z->aligned_x_u |= I8(__fort_aligned)(md, id_map, ud, id_map) << MAXDIMS;
    if (!I8(__fort_aligned)(ud, id_map, md, id_map)) {
#if defined(DEBUG)
      if (__fort_test & DEBUG_SCAT) {
        printf("%d %s unvectored array:\n", lcpu, z->what);
        I8(__fort_describe)(z->ub, ud);
        printf("%d %s misaligned mask array:\n", lcpu, z->what);
        I8(__fort_describe)((char *)z->mb, md);
      }
#endif
      gathscat_abort(z->what, "misaligned mask array");
    }
  } else if (!I8(__fort_fetch_log)(z->mb, md))
    return NULL;

  /* check index arrays */

  for (vx = 0; vx < F90_RANK_G(vd); ++vx) {
    zd = &z->dim[vx];

    if (z->indirect >> vx & 1) {

      /* indirection in this vectored array dimension */

      xd = zd->xd;
#if defined(DEBUG)
      if (F90_TAG_G(xd) != __DESC)
        gathscat_abort(z->what, "index must be array");
      if (F90_KIND_G(xd) != __INT)
        gathscat_abort(z->what, "index must be integer");
#endif
      I8(__fort_cycle_bounds)(xd);
      xoff[vx] = F90_LBASE_G(xd) - 1;
      for (xx = F90_RANK_G(xd); --xx >= 0;) {
        SET_DIM_PTRS(xdd, xd, xx);
        xoff[vx] += F90_DPTR_SOFFSET_G(xdd) * F90_DPTR_LSTRIDE_G(xdd);
      }

      if (z->permuted >> vx & 1) {

        /* index axes are a permutation of the unvectored
           array axes. check for identity permutation */

        alike = 1;
        for (xx = F90_RANK_G(xd); --xx >= 0;) {
          ux = zd->xmap[xx] - 1;
#if defined(DEBUG)
          if (ux < 0 || ux >= F90_RANK_G(ud))
            gathscat_abort(z->what, "invalid index axis");
#endif
          alike &= (ux == xx);

          /* set up inverse mapping from unvectored axis to
             list of vectored axis/index axis pairs */

          x = z->xfree++;
          x->next = z->xhead[ux];
          z->xhead[ux] = x;
          x->xd = xd;
          SET_DIM_PTRS(xdd, xd, xx);
          x->F90_DIM_NAME(xdd) = F90_DIM_NAME(xdd);
          x->DIST_DIM_NAME(xdd) = DIST_DIM_NAME(xdd);
          x->str = F90_DPTR_SSTRIDE_G(xdd) * F90_DPTR_LSTRIDE_G(xdd);
          x->vx = vx;
          x->xx = xx;
        }
        if (alike)
          z->permuted &= ~(1 << vx);
#if defined(DEBUG)
        else if (__fort_test & DEBUG_SCAT) {
          printf("%d %s dim %d index axis permutation", lcpu, z->what, vx + 1);
          for (xx = 0; xx < F90_RANK_G(xd); ++xx)
            printf(" %d", zd->xmap[xx]);
          printf("\n");
        }
#endif
      } else {

        /* index axes not permuted */

        zd->xmap = id_map;
#if defined(DEBUG)
        if (F90_RANK_G(xd) > F90_RANK_G(ud))
          gathscat_abort(z->what, "invalid index rank");
#endif

        /* set up inverse mapping from unvectored axis to list
           of vectored axis/index axis pairs */

        for (xx = F90_RANK_G(xd); --xx >= 0;) {
          ux = xx;
          x = z->xfree++;
          x->next = z->xhead[ux];
          z->xhead[ux] = x;
          x->xd = xd;
          SET_DIM_PTRS(xdd, xd, xx);
          x->F90_DIM_NAME(xdd) = F90_DIM_NAME(xdd);
          x->DIST_DIM_NAME(xdd) = DIST_DIM_NAME(xdd);
          x->str = F90_DPTR_SSTRIDE_G(xdd) * F90_DPTR_LSTRIDE_G(xdd);
          x->vx = vx;
          x->xx = xx;
        }
      }

      z->conform_x_u |= I8(__fort_conform)(xd, zd->xmap, ud, id_map) << vx;
      z->aligned_x_u |= I8(__fort_aligned)(xd, zd->xmap, ud, id_map) << vx;

      if (!I8(__fort_aligned)(ud, id_map, xd, zd->xmap)) {
#if defined(DEBUG)
        if (__fort_test & DEBUG_SCAT) {
          printf("%d %s unvectored array:\n", lcpu, z->what);
          I8(__fort_describe)(z->ub, ud);
          printf("%d %s misaligned index array:\n", lcpu, z->what);
          I8(__fort_describe)((char *)zd->xb, xd);
        }
#endif
        gathscat_abort(z->what, "misaligned index array");
      }
    } else {

      /* no indirection in this vectored array dimension */

      zd->xd = NULL;
      zd->xb = NULL;

      xoff[vx] = 0;

      if (z->permuted >> vx & 1) {

        /* this vectored axis is directly indexed by a
           specified unvectored axis */

        ux = *zd->xmap - 1;
#if defined(DEBUG)
        if (ux < 0 || ux >= F90_RANK_G(ud))
          gathscat_abort(z->what, "invalid unvectored axis");
#endif
        if (ux == vx)
          z->permuted &= ~(1 << vx);
      } else {

        /* this vectored axis is directly indexed by the
           corresponding unvectored axis */

        zd->xmap = &id_map[vx];
        ux = vx;
      }

      if (u_covers_v)
        z->aligned_v_u |= I8(__fort_aligned_axes)(vd, vx + 1, ud, ux + 1) << vx;
      if (v_covers_u)
        z->aligned_u_v |= I8(__fort_aligned_axes)(ud, ux + 1, vd, vx + 1) << vx;
    }
  }

  /* check if any communication is required */

  if (LOCAL_MODE || F90_FLAGS_G(ud) & F90_FLAGS_G(vd) & __LOCAL)
    z->communicate = z->replicate = different = 0;
  else {

    /* check if vectored and unvectored arrays are identically
       replicated over the same processor grid */

    different = (DIST_DIST_TARGET_G(vd) != DIST_DIST_TARGET_G(ud) ||
                 DIST_REPLICATED_G(vd) != DIST_REPLICATED_G(ud));

    /* need to broadcast the result if it is replicated and the
       source array is not identically replicated over the same
       processor grid. if the two arrays are identically
       replicated, then the communication pattern is executed
       redundantly by the replicants. otherwise the primaries
       execute the communication and broadcast the result to the
       replicants. */

    z->replicate = (DIST_REPLICATED_G(rd) && different);
    if (z->replicate)
      I8(__fort_describe_replication)(rd, &z->r_repl);

    /* communication occurs if there is a single alignment or
       scalar subscript in a mapped dimension */

    z->communicate =
        ((DIST_SINGLE_G(vd) & DIST_MAPPED_G(DIST_ALIGN_TARGET_G(vd))) |
         (DIST_SINGLE_G(ud) & DIST_MAPPED_G(DIST_ALIGN_TARGET_G(ud)))) != 0;

    /* ... or if any mapped dimension of the vectored array is
       indirectly indexed */

    z->communicate |= z->indirect & DIST_MAPPED_G(vd);

    /* ... or if any directly indexed dimension of the vectored
       array is not mutually aligned with the corresponding
       dimension of the unvectored array */

    z->communicate |= ~(z->indirect | (z->aligned_v_u & z->aligned_u_v) |
                        -1 << F90_RANK_G(vd));
  }

  /* set up cyclic loops over unvectored section */

  I8(__fort_cycle_bounds)(ud);

  /* estimate the amount of temp space needed */

  for (tempz = 1, i = F90_RANK_G(ud); --i >= 0;) {
    SET_DIM_PTRS(udd, ud, i);
    tempz *= (DIST_DPTR_UAB_G(udd) - DIST_DPTR_PO_G(udd)) -
             (DIST_DPTR_LAB_G(udd) + DIST_DPTR_NO_G(udd)) + 1;
  }

#if defined(DEBUG)
  if (__fort_test & DEBUG_SCAT) {
    printf("%d %s indirect   ", lcpu, z->what);
    for (vx = 0; vx < F90_RANK_G(vd); ++vx)
      printf(" %d", z->indirect >> vx & 1);
    printf("\n");

    printf("%d %s permuted   ", lcpu, z->what);
    for (vx = 0; vx < F90_RANK_G(vd); ++vx)
      printf(" %d", z->permuted >> vx & 1);
    printf("\n");

    printf("%d %s conform_x_u", lcpu, z->what);
    for (vx = 0; vx < F90_RANK_G(vd); ++vx)
      if (z->indirect >> vx & 1)
        printf(" %d", z->conform_x_u >> vx & 1);
      else
        printf(" -");
    if (F90_TAG_G(md) == __DESC)
      printf(" %d", z->conform_x_u >> MAXDIMS & 1);
    printf("\n");

    printf("%d %s aligned_x_u", lcpu, z->what);
    for (vx = 0; vx < F90_RANK_G(vd); ++vx)
      if (z->indirect >> vx & 1)
        printf(" %d", z->aligned_x_u >> vx & 1);
      else
        printf(" -");
    if (F90_TAG_G(md) == __DESC)
      printf(" %d", z->aligned_x_u >> MAXDIMS & 1);
    printf("\n");

    printf("%d %s aligned_v_u", lcpu, z->what);
    for (vx = 0; vx < F90_RANK_G(vd); ++vx)
      printf(" %d", z->aligned_v_u >> vx & 1);
    printf("\n");

    printf("%d %s aligned_u_v", lcpu, z->what);
    for (vx = 0; vx < F90_RANK_G(vd); ++vx)
      printf(" %d", z->aligned_u_v >> vx & 1);
    printf("\n");

    printf("%d %s communicate", lcpu, z->what);
    for (vx = 0; vx < F90_RANK_G(vd); ++vx)
      printf(" %d", z->communicate >> vx & 1);
    printf("\n");

    printf("%d %s different %d replicate %d tempz %d\n", lcpu, z->what,
           different, z->replicate, tempz);
  }
  if (z->communicate && LOCAL_MODE)
    gathscat_abort(z->what, "mapped arguments in LOCAL mode");

  /* 
   * check for a gen_block dimension.
   * if we have one, then it's possible that we have a 0 lsize
   * allocated to it.  Therefore, we do not want the debug code
   * aborting if lsize is 0.
   */

  for (i = 1; i <= F90_RANK_G(ud); ++i) {
    if (DFMT(ud, i) == DFMT_GEN_BLOCK) {
      has_gb = 1;
      break;
    }
  }

  if (has_gb && tempz >= 0)
    goto zero_lsize_ok_for_gen_block;

  if (tempz <= 0 || tempz > F90_LSIZE_G(ud)) {
    printf("%d: has_gb=%d dfmt=%d tempz=%d lsize=%d\n", lcpu, has_gb,
           DIST_DFMT_G(ud), tempz, F90_LSIZE_G(ud));
    gathscat_abort(z->what, "temp estimate wrong #1");
  }

zero_lsize_ok_for_gen_block:

#endif

  /* group_offset is zero if this processor is the primary of the
     unvectored array replication group */

  if (DIST_REPLICATED_G(ud)) {
    I8(__fort_describe_replication)(ud, &u_repl);
    z->group_offset = lcpu - u_repl.plow;
  } else
    z->group_offset = 0;

  z->outgoing = 0;

  if (!(z->communicate | z->replicate)) {

    /* local gather-scatter. allocate buffers for offsets */

    offsetbuf = (int *)__fort_malloc(2 * tempz * sizeof(int) + sizeof(DIST_Desc));

    soff = offsetbuf;
    goff = offsetbuf + tempz;

    z->roff = roff = soff;
    z->loff = loff = goff;
    z->head = z->next = NULL;
    z->counts = counts = countr = countbuf = NULL;

    if (~F90_FLAGS_G(ud) & __OFF_TEMPLATE &&
        (z->group_offset == 0 || !different)) {

      /* this processor is the primary of the replication group,
         or the source and result are identically replicated --
         loop over all local unvectored elements to determine
         gather-scatter offsets */

      if (F90_TAG_G(md) == __DESC)
        I8(gathscat_mask_loop)(z, uoff, xoff, F90_RANK_G(ud));
      else
        I8(gathscat_loop)(z, uoff, xoff, F90_RANK_G(ud));

#if defined(DEBUG)
      if (z->outgoing > tempz)
        gathscat_abort(z->what, "temp estimate wrong #2");
#endif
    }
    lclcnt = z->outgoing;
    maxcnt = 0;
  } else {

    /* non-local gather-scatter.  allocate buffers for offsets and
       linked list for aggregating by target processor */

    tcpus = GET_DIST_TCPUS;
    head = (int *)__fort_malloc((tcpus + 3 * tempz) * sizeof(int));
    next = head + tcpus;
    roff = next + tempz;
    loff = roff + tempz;

    for (i = tcpus; --i >= 0;)
      head[i] = 0;

    z->head = head;
    z->next = next;
    z->roff = roff;
    z->loff = loff;

    countbuf = (int *)__fort_gcalloc(tcpus + __fort_np2, sizeof(int));
    counts = countbuf;
    countr = countbuf + tcpus;

    z->counts = counts;

    if (~F90_FLAGS_G(ud) & __OFF_TEMPLATE &&
        (z->group_offset == 0 || !different)) {

      /* this processor is the primary of the replication group,
         or the source and result are identically replicated --
         loop over all local unvectored elements to determine
         local offset, target cpu, and remote offset */

      if (F90_TAG_G(md) == __DESC)
        I8(gathscat_mask_loop)(z, uoff, xoff, F90_RANK_G(ud));
      else
        I8(gathscat_loop)(z, uoff, xoff, F90_RANK_G(ud));

#if defined(DEBUG)
      if (z->outgoing > tempz)
        gathscat_abort(z->what, "temp estimate wrong #3");
#endif

      for (i = tcpus; --i >= 0;)
        countr[i] = counts[i];
    }

    /* exchange counts */

    maxcnt = __fort_exchange_counts(countr);

    incoming = 0;
    for (i = tcpus; --i >= 0;)
      incoming += countr[i];

    /* head     = head of linked list per target processor */
    /* next     = pointers to next in linked list */
    /* roff     = unaggregated offsets in remote vectored array */
    /* loff     = unaggregated offsets in local unvectored array */
    /* counts   = send element count per target processor */
    /* countr   = receive element count per target processor */
    /* maxcnt   = max send/receive count between any processor pair */
    /* outgoing = total number of outgoing elements */
    /* incoming = total number of incoming elements */

    /* allocate buffers for aggregating incoming and outgoing
       offsets */

    offr = (int *)__fort_gmalloc(2 * maxcnt * sizeof(int));
    offs = offr + maxcnt;

    /* allocate buffers for aggregated local offsets */

    offsetbuf = (int *)__fort_malloc((incoming + z->outgoing) * sizeof(int));
    soff = offsetbuf;
    goff = offsetbuf + incoming;

    lclcnt = 0;
    for (i = head[lcpu]; i > 0; i = next[i - 1]) {
      goff[lclcnt] = z->loff[i - 1];
      soff[lclcnt] = z->roff[i - 1];
      ++lclcnt;
    }

#if defined(DEBUG)
    if (lclcnt != counts[lcpu] || lclcnt != countr[lcpu])
      gathscat_abort(z->what, "local counts wrong");
#endif

    /* exchange targets are chosen by xor'ing this processor number
       with all masks in the range 1 .. 2**log2(__fort_tcpus).  Target
       values >= tcpus are skipped (which occurs only if tcpus is not
       a power of two.)  */

    j = k = lclcnt;

    for (m = 1; m < __fort_np2; ++m) {

      /* identify target */

      cpu = lcpu ^ m;
      if (cpu >= tcpus)
        continue;

      /* aggregate local and remote offsets for outgoing
         elements */

      for (i = head[cpu], n = 0; i > 0; i = next[i - 1]) {
        goff[j++] = z->loff[i - 1];
        offs[n++] = z->roff[i - 1];
      }

      nr = countr[cpu];
      ns = counts[cpu];

#if defined(DEBUG)
      if (n != ns)
        gathscat_abort(z->what, "aggregation counts wrong #1");
#endif

      /* exchange offsets */

      if (cpu < lcpu) {
        if (nr > 0)
          __fort_rrecvl(cpu, offr, nr, 1, __CINT, sizeof(int));
        if (ns > 0)
          __fort_rsendl(cpu, offs, ns, 1, __CINT, sizeof(int));
      } else {
        if (ns > 0)
          __fort_rsendl(cpu, offs, ns, 1, __CINT, sizeof(int));
        if (nr > 0)
          __fort_rrecvl(cpu, offr, nr, 1, __CINT, sizeof(int));
      }

      /* aggregate offsets for incoming elements.  copying is
         required because we need to receive into a gmalloc'd
         buffer of equal size (maxcnt) on all processors */

      for (i = 0; i < nr; ++i)
        soff[k++] = offr[i];
    }

#if defined(DEBUG)
    if (j != z->outgoing || k != incoming)
      gathscat_abort(z->what, "aggregation counts wrong #2");
#endif

    __fort_gfree(offr);
    __fort_free(head);
  }

  if (z->replicate && ~F90_FLAGS_G(rd) & __OFF_TEMPLATE) {

/* result elements are local and replicated, but the source
   array is not identically replicated -- broadcast the result */

#if defined(DEBUG)
    if (__fort_test & DEBUG_SCAT) {
      printf("%d %s repl p0 %d pn ", lcpu, z->what, z->r_repl.plow);
      I8(__fort_show_index)(z->r_repl.ndim, z->r_repl.pcnt);
      printf(" ps ");
      I8(__fort_show_index)(z->r_repl.ndim, z->r_repl.pstr);
      printf("\n");
    }
#endif
    repchn =
        __fort_chn_1toN(NULL, z->r_repl.ndim, z->r_repl.plow, z->r_repl.pcnt,
                       z->r_repl.pstr, 0, z->r_repl.plow, NULL, NULL);
    rp = z->rb + DIST_SCOFF_G(rd) * F90_LEN_G(rd);
    if (z->r_repl.plow == lcpu)
      __fort_sendl(repchn, 0, rp, F90_LSIZE_G(rd), 1, F90_KIND_G(rd),
                  F90_LEN_G(rd));
    __fort_recvl(repchn, 0, rp, F90_LSIZE_G(rd), 1, F90_KIND_G(rd),
                F90_LEN_G(rd));
    __fort_chn_prune(repchn);
    __fort_setbase(repchn, rp, rp, F90_KIND_G(rd), F90_LEN_G(rd));
  } else
    repchn = NULL;

  /* create a new gather-scatter schedule */

  sk = (gathscat_sked *)__fort_malloc(sizeof(gathscat_sked));
  sk->sked.tag = __SKED;
  sk->sked.start = I8(gathscat_start);
  sk->sked.free = gathscat_free;
  sk->sked.arg = sk;
  sk->what = z->what;
  sk->gathscatfn = z->gathscatfn;
  sk->scatterfn = z->scatterfn;
  sk->repchn = repchn;
  sk->lclcnt = lclcnt;
  sk->maxcnt = maxcnt;
  sk->countbuf = countbuf;
  sk->offsetbuf = offsetbuf;
  if (z->dir == __GATHER) { /* swap */
    sk->counts = countr;
    sk->countr = counts;
    sk->goff = soff;
    sk->soff = goff;
  } else {
    sk->counts = counts;
    sk->countr = countr;
    sk->goff = goff;
    sk->soff = soff;
  }

  if (__fort_test & DEBUG_TIME) {
    t = __fort_second() - t;
    printf("%d %s schedule %.6f\n", lcpu, sk->what, t);
  }
  return &sk->sked;
}

void *I8(__fort_adjust_index_array)(const char *what, char *idx_array,
                                    char *src, int dim, F90_Desc *is,
                                    F90_Desc *bs)
{

  /* Adjust Index array for scatter routines. This needs to be called
   * when the BASE array argument's lower bound is not 1. We need to
   * then adjust the index array by adding LBASE(BASE, dim) - 1 to each
   * element in array.
   *
   *
   *
   * See HPF 2.0 Spec (6.4.4 Array Combining Scatter Functions)
   * page 93-94 for details.
   *
   *
   * Arguments:
   * what = type of scatter routine
   * idx_array = rslt of adjusted array here. If this is NULL, we
   *             allocate an global array ...
   * src = if idx_array != NULL, this should be same as idx_array.
   * dim = dimension of index array (for use by is)
   * is  = index array descriptor
   * bs  = BASE array (argument in SCATTER routine) descriptor
   *
   */

  __INT_T adj, i;

#if defined(DEBUG)

  if (ISSCALAR(is)) {
    printf("%d %s: index argument must be an array\n", GET_DIST_LCPU, what);
    __fort_abort((char *)0);
  }

  if (idx_array && idx_array != src) {
    printf("%d %s: index_array address must be NULL or == src address\n",
           GET_DIST_LCPU, what);
    __fort_abort((char *)0);
  }

#endif

  if (idx_array == NULL) {
    idx_array = __fort_gmalloc(F90_GSIZE_G(is) * F90_LEN_G(is));
  }

  adj = F90_DIM_LBOUND_G(bs, dim) - 1;

  switch (F90_KIND_G(is)) {

  case __INT1:
    for (i = 0; i < F90_LSIZE_G(is); ++i)
      *((__INT1_T *)idx_array + i) = *((__INT1_T *)src + i) + adj;
    break;
  case __INT2:
    for (i = 0; i < F90_LSIZE_G(is); ++i)
      *((__INT2_T *)idx_array + i) = *((__INT2_T *)src + i) + adj;
    break;

  case __INT4:
    for (i = 0; i < F90_LSIZE_G(is); ++i)
      *((__INT4_T *)idx_array + i) = *((__INT4_T *)src + i) + adj;

    break;

  case __INT8:
    for (i = 0; i < F90_LSIZE_G(is); ++i)
      *((__INT8_T *)idx_array + i) = *((__INT8_T *)src + i) + adj;

    break;

  default:
    printf("%d %s: bad type for index loc=100\n", GET_DIST_LCPU, what);
    __fort_abort((char *)0);
  }

  return idx_array;
}

void *I8(__fort_create_conforming_index_array)(const char *what, char *ab,
                                               void *ib, F90_Desc *as,
                                               F90_Desc *is, F90_Desc *new_is)
{

  /* Create a conforming index array. Returns a pointer to the
   * array and assigns a new descriptor to new_is. Caller is
   * responsible for the __fort_gree() ...
   */

  __INT_T idx_kind, idx_len, i, _255 = 255;
  void *idx_array;

#if defined(DEBUG)

  if (!ISSCALAR(is)) {
    printf("%d %s: index argument must be a scalar\n", GET_DIST_LCPU, what);
    __fort_abort((char *)0);
  }

#endif

  idx_kind = *((int *)is);

  switch (idx_kind) {

  case __INT1:
    idx_len = sizeof(__INT1_T);
    break;
  case __INT2:
    idx_len = sizeof(__INT2_T);
    break;
  case __INT4:
    idx_len = sizeof(__INT4_T);
    break;

  case __INT8:
    idx_len = sizeof(__INT8_T);
    break;

  default:
    printf("%d %s: bad type for index loc=1\n", GET_DIST_LCPU, what);
    __fort_abort((char *)0);
  }

  ENTFTN(INSTANCE, instance)
  (new_is, as, &idx_kind, &idx_len, &_255); /*no overlaps*/

  idx_array = (void *)__fort_gmalloc(F90_GSIZE_G(new_is) * idx_len);

  switch (idx_kind) {

  case __INT1:
    for (i = 0; i < F90_LSIZE_G(new_is); ++i)
      *((__INT1_T *)idx_array + i) = *((__INT1_T *)ib);
    break;
  case __INT2:
    for (i = 0; i < F90_LSIZE_G(new_is); ++i)
      *((__INT2_T *)idx_array + i) = *((__INT2_T *)ib);
    break;

  case __INT4:
    for (i = 0; i < F90_LSIZE_G(new_is); ++i)
      *((__INT4_T *)idx_array + i) = *((__INT4_T *)ib);
    break;

  case __INT8:
    for (i = 0; i < F90_LSIZE_G(new_is); ++i)
      *((__INT8_T *)idx_array + i) = *((__INT8_T *)ib);
    break;

  default:
    printf("%d %s: bad type for index loc=2\n", GET_DIST_LCPU, what);
    __fort_abort((char *)0);
  }

  return idx_array;
}
