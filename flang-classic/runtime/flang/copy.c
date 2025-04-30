/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* copy.c -- copy (permute) array section */

#include "stdioInterf.h"
#include "fioMacros.h"

#if defined(DEBUG)
void I8(print_dist_format)(F90_Desc *);
#endif

#include "fort_vars.h"

/* un-permuted axis map */

static int identity_map[MAXDIMS] = {1, 2, 3, 4, 5, 6, 7};

/* parameters for copy_section */

typedef struct {
  char *base;              /* base address */
  DECL_HDR_PTRS(sect);     /* original descriptor */
  int *axis_map;           /* permuted section dimensions */
  chdr **ch;               /* channel array */
  int islocal;             /* nonzero if all unit dims are local */
  __INT_T lower[MAXDIMS];  /* subsection lower bound */
  __INT_T upper[MAXDIMS];  /* subsection upper bound */
  __INT_T stride[MAXDIMS]; /* subsection stride */
  __INT_T extent[MAXDIMS]; /* subsection extent */
  repl_t repl;             /* replication descriptor */
} copy_sect;

typedef struct {
#if defined(DEBUG)
  char *xdir; /* transfer direction (string) */
#endif
  void (*xfer)(struct chdr *, int, void *, long, long, int, long);
  /* xfer: __fort_sendl or __fort_recvl */
  chdr *cc;     /* current channel */
  chdr *mych;   /* source=dest=my group */
  int permuted; /* bit mask: src/dst axis permutation */
  __INT_T span; /* span over contiguous dims */
  __INT_T cnt;  /* contiguous block length */
  int unit;     /* span over unit dims */
  __INT_T cpu;  /* remote processor number */
  copy_sect dy, sy;
} copy_parm;

static int _1 = 1;

#if defined(DEBUG)

void I8(print_dist_format)(DECL_HDR_PTRS(d))
{

  switch (DIST_DFMT_G(d) & DFMT__MASK) {

  case DFMT_COLLAPSED:

    printf("COLLAPSED");
    break;

  case DFMT_BLOCK:

    printf("BLOCK");
    break;

  case DFMT_BLOCK_K:

    printf("BLOCK_K");
    break;

  case DFMT_GEN_BLOCK:

    printf("GEN_BLOCK");
    break;

  case DFMT_CYCLIC:

    printf("CYCLIC");
    break;

  case DFMT_CYCLIC_K:

    printf("CYCLIC_K");
    break;

  default:

    printf("Unsupported Dist-Format %d", DIST_DFMT_G(d) & DFMT__MASK);
    break;
  }
}

#endif

/* l = local, r = remote */

static void I8(copy_xfer_loop)(copy_parm *z, copy_sect *ly, __INT_T offset,
                               __INT_T dim)
{
  DECL_HDR_PTRS(lc);
  DECL_DIM_PTRS(lcd);
  char *adr;
  int lx;
  __INT_T cnt, str;

  lc = ly->sect;
  cnt = str = 1;
  if (dim > 0) {
    if (dim > z->span) {
      lx = ly->axis_map[dim - 1];
      SET_DIM_PTRS(lcd, lc, lx - 1);
      cnt = ly->extent[lx - 1];
      str = ly->stride[lx - 1] * F90_DPTR_SSTRIDE_G(lcd) *
            F90_DPTR_LSTRIDE_G(lcd);
      if (dim > z->unit) {
        for (; --cnt >= 0; offset += str)
          I8(copy_xfer_loop)(z, ly, offset, dim - 1);
        return;
      }
    } else
      cnt = z->cnt;
  }
  adr = ly->base + offset * F90_LEN_G(lc);
#if defined(DEBUG)
  if (__fort_test & DEBUG_COPY) {
    printf("%d copy %s cpu=%d offset=%d adr=%x cnt=%d str=%d ", GET_DIST_LCPU,
           z->xdir, z->cpu, offset, adr, cnt, str);
    __fort_show_scalar(adr, F90_KIND_G(lc));
    printf("\n");
  }
#endif
  z->xfer(z->cc, z->cpu, adr, cnt, str, F90_KIND_G(lc), F90_LEN_G(lc));
}

/* set up to transfer block */

static void I8(copy_xfer)(copy_parm *z,   /* parameter struct */
                          copy_sect *ly,  /* local parameters */
                          copy_sect *ry,  /* remote parameters */
                          __INT_T offset, /* local array offset */
                          __INT_T cnt)    /* transfer length */
{
  DECL_HDR_PTRS(lc);
  DECL_HDR_PTRS(rc);
  DECL_DIM_PTRS(lcd);
  DECL_DIM_PTRS(rcd);
  proc *p;
  procdim *pd;
  int dfmt, lx, px, rx;
  __INT_T i, m, pc = 0, rg0, rp0;
  __INT_T rpc[MAXDIMS];

  /* Determine the maximum span of dimensions that can be
     transferred as one contiguous block and compute the contiguous
     block length */

  lc = ly->sect;
  rc = ry->sect;
  for (m = i = 1; i <= F90_RANK_G(lc); ++i) {
    if ((z->permuted >> (i - 1)) & 1)
      break;
    lx = ly->axis_map[i - 1];
    SET_DIM_PTRS(lcd, lc, lx - 1);
    rx = ry->axis_map[i - 1];
    SET_DIM_PTRS(rcd, rc, rx - 1);
    if (ly->stride[lx - 1] * F90_DPTR_SSTRIDE_G(lcd) *
                F90_DPTR_LSTRIDE_G(lcd) !=
            m ||
        ry->stride[rx - 1] * F90_DPTR_SSTRIDE_G(rcd) *
                F90_DPTR_LSTRIDE_G(rcd) !=
            m)
      break;
    m *= ly->extent[lx - 1];
  }
  for (; i <= F90_RANK_G(lc); ++i) {
    lx = ly->axis_map[i - 1];
    if (ly->extent[lx - 1] != 1)
      break;
  }
  z->span = i - 1;
  z->cnt = m;

  /* Determine the maximum span of unit dimensions, i.e.
     dimensions that can be transferred with a stride between
     single elements. */

  for (i = 1; i <= F90_RANK_G(lc); ++i) {
    lx = ly->axis_map[i - 1];
    if (ly->extent[lx - 1] != 1)
      break;
  }
  z->unit = i;

#if defined(DEBUG)
  if (__fort_test & DEBUG_COPY) {
    printf("%d copy %s l(", GET_DIST_LCPU, z->xdir);
    for (i = 0; i < F90_RANK_G(lc); ++i) {
      if (i > 0)
        printf(",");
      printf("%d:%d:%d", ly->lower[i], ly->upper[i], ly->stride[i]);
    }
    printf(") r(");
    for (i = 0; i < F90_RANK_G(rc); ++i) {
      if (i > 0)
        printf(",");
      printf("%d:%d:%d", ry->lower[i], ry->upper[i], ry->stride[i]);
    }
    printf(") cnt=%d span=%d unit=%d\n", z->cnt, z->span, z->unit);
  }
#endif

  if (LOCAL_MODE) {

    /* only local communication */

    z->cpu = 0;
  } else {

    /* Determine remote prime owner (low processor number) and
       replication group index. */

    rp0 = DIST_PBASE_G(rc);
    dfmt = DIST_DFMT_G(rc);
    for (rx = 0; rx < F90_RANK_G(rc); ++rx, dfmt >>= DFMT__WIDTH) {
      switch (dfmt & DFMT__MASK) {
      case DFMT_COLLAPSED:
        rpc[rx] = 0;
        continue;
      default:
        __fort_abort("copy: unsupported dist-format");
      }
      rp0 += pc * DIST_DPTR_PSTRIDE_G(rcd); /* owning processor number */
      rpc[rx] = pc;
    }

    if (DIST_REPLICATED_G(lc) | DIST_REPLICATED_G(rc)) {

      /* source or dest replication */

      p = DIST_DIST_TARGET_G(rc);
      m = DIST_REPLICATED_G(rc);
      for (px = 0; m != 0; m >>= 1, ++px) {
        pd = &p->dim[px];
        if (m & 1 && pd->coord > 0)
          rp0 -= pd->coord * pd->stride;
      }

      rg0 = 0;
      for (rx = F90_RANK_G(rc); --rx >= 0;)
        rg0 += rpc[rx] * ry->repl.gstr[rx]; /* replication group index */

      if (rp0 == ry->repl.plow)
        z->cc = z->mych; /* source & dest = my group */
      else
        z->cc = ry->ch[rg0];
      if (z->cc == NULL) {
        if (z->xfer == __fort_sendl) {
          z->cc = __fort_chn_1toN(NULL, ry->repl.ndim, rp0, ry->repl.pcnt,
                                 ry->repl.pstr, ly->repl.ndim, ly->repl.plow,
                                 ly->repl.pcnt, ly->repl.pstr);
        } else {
          z->cc = __fort_chn_1toN(NULL, ly->repl.ndim, ly->repl.plow,
                                 ly->repl.pcnt, ly->repl.pstr, ry->repl.ndim,
                                 rp0, ry->repl.pcnt, ry->repl.pstr);
        }
        if (rp0 == ry->repl.plow)
          z->mych = z->cc;
        ry->ch[rg0] = z->cc;
      }
#if defined(DEBUG)
      if (__fort_test & DEBUG_COPY)
        printf("%d copy %s cc=%x rg0=%d rp0=%d\n", GET_DIST_LCPU, z->xdir,
               z->cc, rg0, rp0);
      if (rg0 < 0 || rg0 >= ry->repl.ngrp)
        __fort_abort("copy_xfer: bad remote group (internal error)");
#endif
    } else          /* no replication */
      z->cpu = rp0; /* remote processor number */
  }
  I8(copy_xfer_loop)(z, ly, offset, F90_RANK_G(lc));
}

/* loop over local section, find maximal blocks to transfer to remote
   processors.  Set up localized descriptors. */

static void I8(copy_loop)(copy_parm *z,   /* parameter struct */
                          copy_sect *ly,  /* local section */
                          copy_sect *ry,  /* remote section */
                          __INT_T offset, /* local array offset */
                          __INT_T cnt,    /* transfer length */
                          int dim)        /* dimension */
{
  DECL_HDR_PTRS(lc);
  DECL_HDR_PTRS(rc);
  DECL_DIM_PTRS(lcd);
  DECL_DIM_PTRS(rcd);
  int lx, rx;
  __INT_T cl, cn, cs, clof, clos, i, ll, ln, lu, n, off, rl, rn, ru;

  lc = ly->sect;
  lx = ly->axis_map[dim - 1] - 1;
  SET_DIM_PTRS(lcd, lc, lx);

  rc = ry->sect;
  rx = ry->axis_map[dim - 1] - 1;
  SET_DIM_PTRS(rcd, rc, rx);

  if (DIST_MAPPED_G(lc) >> lx & 1) {

    /* local is mapped */

    cl = DIST_DPTR_CL_G(lcd); /* cyclic parameters */
    cs = DIST_DPTR_CS_G(lcd);
    cn = DIST_DPTR_CN_G(lcd);
    clof = DIST_DPTR_CLOF_G(lcd);
    clos = DIST_DPTR_CLOS_G(lcd);

    if (DIST_DPTR_CNO_G(lcd) > 1 &&
        DIST_DPTR_BLOCK_G(lcd) == DIST_DPTR_TSTRIDE_G(lcd)) {

      /* local is mapped cyclic, one element per block */

      if (DIST_MAPPED_G(rc) >> rx & 1) {

#if defined(DEBUG)
        if (__fort_test & DEBUG_COPY)
          printf("%d lx %d cyclic : mapped rx %d (skipped)\n", GET_DIST_LCPU,
                 lx + 1, rx + 1);
#endif

      } else {

/* remote is unmapped */

#if defined(DEBUG)
        if (__fort_test & DEBUG_COPY)
          printf("%d lx %d cyclic : unmapped rx %d\n", GET_DIST_LCPU, lx + 1,
                 rx + 1);
#endif

        ly->lower[lx] = cl - DIST_DPTR_TOFFSET_G(lcd);
        if (DIST_DPTR_TSTRIDE_G(lcd) != 1)
          ly->lower[lx] /= DIST_DPTR_TSTRIDE_G(lcd);
        ly->upper[lx] = ly->lower[lx] + cn - 1;
        ly->extent[lx] = cn;

        ry->lower[rx] =
            F90_DPTR_LBOUND_G(rcd) + ly->lower[lx] - F90_DPTR_LBOUND_G(lcd);
        ry->upper[rx] = ry->lower[rx] + (cn - 1) * DIST_DPTR_PSHAPE_G(lcd);
        ry->stride[rx] = DIST_DPTR_PSHAPE_G(lcd);
        ry->extent[rx] = cn;

        off = offset +
              (F90_DPTR_SSTRIDE_G(lcd) * ly->lower[lx] +
               F90_DPTR_SOFFSET_G(lcd) - clof) *
                  F90_DPTR_LSTRIDE_G(lcd);

        if (dim > 1)
          I8(copy_loop)(z, ly, ry, off, cnt * cn, dim - 1);
        else
          I8(copy_xfer)(z, ly, ry, off, cnt * cn);
        return;
      }
    }
  } else {

    /* local is unmapped */

    cl = clof = clos = 0; /* unmapped cyclic same as block */
    cn = cs = 1;

    if (DIST_DPTR_CNO_G(rcd) > 1 &&
        DIST_DPTR_BLOCK_G(rcd) == DIST_DPTR_TSTRIDE_G(rcd)) {

/* remote is mapped cyclic, one element per block */

#if defined(DEBUG)
      if (__fort_test & DEBUG_COPY)
        printf("%d lx %d unmapped : cyclic rx %d\n", GET_DIST_LCPU, lx + 1,
               rx + 1);
#endif

      rn = DIST_DPTR_PSHAPE_G(rcd);
      ly->lower[lx] = F90_DPTR_LBOUND_G(lcd); /* no-op ? */
      ly->upper[lx] = DPTR_UBOUND_G(lcd);     /* no-op ? */
      ly->stride[lx] = rn;

      off = offset +
            (F90_DPTR_SSTRIDE_G(lcd) * ly->lower[lx] + F90_DPTR_SOFFSET_G(lcd) -
             clof) *
                F90_DPTR_LSTRIDE_G(lcd);

      ry->lower[rx] = F90_DPTR_LBOUND_G(rcd); /* no-op ? */
      ry->stride[rx] = 1;
      for (i = rn; i > 0; --i) {
        n = (DPTR_UBOUND_G(rcd) - ry->lower[rx] + rn) / rn;
        if (n <= 0)
          break;
        ry->upper[rx] = ry->lower[rx] + n - 1;
        ly->extent[lx] = ry->extent[rx] = n;
        if (dim > 1)
          I8(copy_loop)(z, ly, ry, off, cnt * cn, dim - 1);
        else
          I8(copy_xfer)(z, ly, ry, off, cnt * cn);
        ly->lower[lx] += 1;
        ry->lower[rx] += 1;
        off += F90_DPTR_SSTRIDE_G(lcd) * F90_DPTR_LSTRIDE_G(lcd);
      }
      return;
    }
  }

/* general case copy */

#if defined(DEBUG)
  if (__fort_test & DEBUG_COPY)
    printf("%d lx %d mapped : mapped rx %d\n", GET_DIST_LCPU, lx + 1, rx + 1);
#endif

  while (cn > 0) {

    /* set up local block bounds */

    if (DIST_MAPPED_G(lc) >> lx & 1) {
      ln = I8(__fort_block_bounds)(lc, lx + 1, cl, &ll, &lu);
    } else {
      /* unmapped case */
      ll = F90_DPTR_LBOUND_G(lcd);
      lu = DPTR_UBOUND_G(lcd);
      ln = F90_DPTR_EXTENT_G(lcd);
    }

    off = offset +
          (F90_DPTR_SSTRIDE_G(lcd) * ll + F90_DPTR_SOFFSET_G(lcd) - clof) *
              F90_DPTR_LSTRIDE_G(lcd);

#if defined(DEBUG)

    if (__fort_test & DEBUG_COPY) {

      printf("%d copy_loop: Local Array dist format is ", GET_DIST_LCPU);
      I8(print_dist_format)(lc);
      printf(" block size=%d tstride=%d tlb=%d tub=%d", DIST_DPTR_BLOCK_G(lcd),
             DIST_DPTR_TSTRIDE_G(lcd), DIST_DPTR_TLB_G(lcd), DIST_DPTR_TUB_G(lcd));

      printf("\n%d copy_loop: Remote Array dist format is ", GET_DIST_LCPU);
      I8(print_dist_format)(rc);
      printf(" block size=%d tstride=%d tlb=%d tub=%d", DIST_DPTR_BLOCK_G(rcd),
             DIST_DPTR_TSTRIDE_G(rcd), DIST_DPTR_TLB_G(rcd), DIST_DPTR_TUB_G(rcd));

      printf("\n");
    }
#endif

    if ((DFMT(rc, rx + 1) == DFMT_GEN_BLOCK)) {

      /* 
       * For the case of the remote array being a gen_block
       * distribution, we cannot use the elegant math for
       * finding the largest overlapping block.  We need
       * to consult the gen_block array for the block size since
       * the block size can vary across processors.
       */

      __INT_T pshape;

      __INT_T i, j;

      __INT_T trl, tru, lLbound, rLbound;
      __INT_T direction;
      __INT_T blockSize;
      __INT_T tExtent, aExtent;

      __INT_T *gb;

      __INT_T *tempGB;

      lLbound = F90_DPTR_LBOUND_G(lcd);
      rLbound = F90_DPTR_LBOUND_G(rcd);

      pshape = DIST_DPTR_PSHAPE_G(rcd);

      /* If we have a negative template stride, then we need to
       * traverse the gen_block array from right to left rather
       * than left to right...otherwise, we may send some array
       * elements out of order ...
       */

      direction = (DIST_DPTR_TSTRIDE_G(rcd) < 0);

      tExtent = (DIST_DPTR_TUB_G(rcd) - DIST_DPTR_TLB_G(rcd)) + 1;
      aExtent = (DPTR_UBOUND_G(rcd) - rLbound) + 1;

      if (tExtent != aExtent) {

        tempGB = I8(__fort_new_gen_block)(rc, rx);
        gb = tempGB;

      } else {

        tempGB = 0;
        gb = DIST_DPTR_GEN_BLOCK_G(rcd);
      }

      gb += ((!direction) ? 0 : (pshape - 1));

      rl = rLbound;
      ru = rl - 1;

      for (i = 0; i < pshape; ++i) {

        blockSize = *gb;

        /* check each element in gen_block array for overlapping
         * block...
         */

        if (blockSize) {

          ru += blockSize;
          trl = tru = 0;

          /* find overlapping bounds */

          for (j = ((ll - lLbound) + 1); j <= ((lu - lLbound) + 1); j++) {

            if (j == ((rl - rLbound) + 1))
              trl = j;

            if (j >= ((rl - rLbound) + 1) && j <= ((ru - rLbound) + 1))
              tru = j;
          }

          if (trl || tru) {

            if (!trl) {
              if (ll > rl)
                trl = ((ll - lLbound) + 1);
              else
                trl = ((rl - rLbound) + 1);

            } else if (!tru) {

              if (lu > ru)
                tru = ((ru - rLbound) + 1);
              else
                tru = ((lu - lLbound) + 1);
            }

            /* we got an overlapping boundary.
             * let's send the request ...
             */

            n = tru - trl + 1;

            ly->lower[lx] = ((trl + lLbound) - 1);
            ly->upper[lx] = (ll + n) - 1;
            ly->extent[lx] = n;
            ry->lower[rx] = ((trl + rLbound) - 1);
            ry->upper[rx] = (rl + n) - 1;
            ry->extent[rx] = n;

            if (dim > 1)
              I8(copy_loop)(z, ly, ry, off, cnt * n, dim - 1);
            else
              I8(copy_xfer)(z, ly, ry, off, cnt * n);

            off += n * F90_DPTR_SSTRIDE_G(lcd) * F90_DPTR_LSTRIDE_G(lcd);
          }
        }

        rl += *gb;

        if (!direction)
          ++gb;
        else
          --gb;
      }

      cl += cs;
      clof += clos;
      --cn;

      if (tempGB)
        __fort_free(tempGB);

      continue; /*jump to while(cn > 0) loop head*/
    }

    i = ll - F90_DPTR_LBOUND_G(lcd);
    rl = F90_DPTR_LBOUND_G(rcd) + i;
    ru = DPTR_UBOUND_G(rcd);

    /* find largest overlapping blocks of local and remote
       sections, i.e. subdivide the local block at remote block
       boundaries */

    while (ln > 0) {

      rn = ru - rl + 1;
      n = Min(ln, rn);

      if (n <= 0) {
        char error[100];
        sprintf(error, "copy_loop: empty block (internal error)");
        __fort_abort(error);
      }
      ly->lower[lx] = ll;
      ly->upper[lx] = ll + n - 1;
      ly->extent[lx] = n;
      ry->lower[rx] = rl;
      ry->upper[rx] = rl + n - 1;
      ry->extent[rx] = n;

      /* transfer one overlapping block */

      if (dim > 1)
        I8(copy_loop)(z, ly, ry, off, cnt * n, dim - 1);
      else
        I8(copy_xfer)(z, ly, ry, off, cnt * n);

      rl += n;
      ll += n;
      off += n * F90_DPTR_SSTRIDE_G(lcd) * F90_DPTR_LSTRIDE_G(lcd);
      ln -= n;
    }

    cl += cs;
    clof += clos;
    --cn;
  }
}

void I8(copy_setup)(copy_sect *y, char *b, F90_Desc *c, int *axis_map)
{
  int cx, i;
  DECL_DIM_PTRS(cd);

  y->base = b;
  y->sect = c;

  y->axis_map = axis_map;

  I8(__fort_cycle_bounds)(c);

  y->islocal = (~F90_FLAGS_G(c) & __OFF_TEMPLATE) && (F90_LSIZE_G(c) > 0);

  for (i = F90_RANK_G(c); i > 0; --i) {
    cx = axis_map[i - 1];
    SET_DIM_PTRS(cd, c, cx - 1);
    y->lower[cx - 1] = F90_DPTR_LBOUND_G(cd);
    y->upper[cx - 1] = DPTR_UBOUND_G(cd);
    y->stride[cx - 1] = 1;
    y->extent[cx - 1] = F90_DPTR_EXTENT_G(cd);
    y->islocal &= (DIST_DPTR_CN_G(cd) > 0);
  }
}

/* base addresses already adjusted for scalar subscripts */

chdr *I8(__fort_copy)(void *db, void *sb, F90_Desc *dc, F90_Desc *sc,
                     int *src_axis_map)
{
  DECL_DIM_PTRS(dcd);
  DECL_DIM_PTRS(scd);
  chdr *ch;
  copy_parm z;
  int dbogus, dg, dl, dn, ds, du, dx, sbogus, sg, sl, sn, ss, su, sx;
  int i, lcpu, j, n, ng;
  __INT_T offset;
  int *tcpus_addr;

  if (src_axis_map == NULL)
    src_axis_map = identity_map;

  lcpu = GET_DIST_LCPU;

#if defined(DEBUG)
  if (F90_TAG_G(dc) != __DESC)
    __fort_abort("copy: invalid destination descriptor");
  if (F90_TAG_G(sc) != __DESC)
    __fort_abort("copy: invalid source descriptor");
  if (__fort_test & DEBUG_COPY) {
    printf("%d copy d", lcpu);
    I8(__fort_show_section)(dc);
    printf("@%x = s", db);
    I8(__fort_show_section)(sc);
    printf("@%x smap=", sb);
    I8(__fort_show_index)(F90_RANK_G(sc), src_axis_map);
    printf("\n");
  }
  if (F90_RANK_G(dc) != F90_RANK_G(sc))
    __fort_abort("copy: section rank mismatch");
  if (F90_KIND_G(dc) != F90_KIND_G(sc) || F90_LEN_G(dc) != F90_LEN_G(sc))
    __fort_abort("copy: type/len mismatch");
#endif

  /* bogus bounds: section setup deferred until we get here. */

  dbogus = F90_FLAGS_G(dc) & __BOGUSBOUNDS;
  sbogus = F90_FLAGS_G(sc) & __BOGUSBOUNDS;

  if (dbogus | sbogus) {

    F90_FLAGS_P(dc, F90_FLAGS_G(dc) & ~__BOGUSBOUNDS);
    F90_FLAGS_P(sc, F90_FLAGS_G(sc) & ~__BOGUSBOUNDS);

    /* narrow bounds to range of mutually valid array indexes */

    for (dx = 1; dx <= F90_RANK_G(dc); ++dx) {
      sx = src_axis_map[dx - 1];

      SET_DIM_PTRS(dcd, dc, dx - 1);
      SET_DIM_PTRS(scd, sc, sx - 1);

      /* compute number of strides to bring section lbound up to
         target array lbound */

      if (dbogus) {
        ds = F90_DPTR_SSTRIDE_G(dcd);
        dl = ds > 0 ? DIST_DPTR_TLB_G(dcd) - 1 : DIST_DPTR_TUB_G(dcd) + 1;
        dn = dl - F90_DPTR_LBOUND_G(dcd) + ds;
        if (ds != 1)
          dn /= ds;
        if (dn < 0)
          dn = 0;
      } else
        dn = 0;

      if (sbogus) {
        ss = F90_DPTR_SSTRIDE_G(scd);
        sl = ss > 0 ? DIST_DPTR_TLB_G(scd) - 1 : DIST_DPTR_TUB_G(scd) + 1;
        sn = sl - F90_DPTR_LBOUND_G(scd) + ss;
        if (ss != 1)
          sn /= ss;
        if (sn < 0)
          sn = 0;
      } else
        sn = 0;

      /* maximize lbound increment */

      n = Max(dn, sn);

      /* adjust lbound, compute extents */

      if (dbogus) {
        dl = F90_DPTR_LBOUND_G(dcd) + n * ds;
        du = ds > 0 ? Min(DPTR_UBOUND_G(dcd), DIST_DPTR_TUB_G(dcd))
                    : Max(DPTR_UBOUND_G(dcd), DIST_DPTR_TLB_G(dcd));
        dn = du - dl + ds;
        if (ds != 1)
          dn /= ds;
        if (dn < 0)
          dn = 0;
      } else if (n != 0)
        __fort_abort("copy: can't adjust dst lbound");
      else
        dn = F90_DPTR_EXTENT_G(dcd);

      if (sbogus) {
        sl = F90_DPTR_LBOUND_G(scd) + sn * ss;
        su = ss > 0 ? Min(DPTR_UBOUND_G(scd), DIST_DPTR_TUB_G(scd))
                    : Max(DPTR_UBOUND_G(scd), DIST_DPTR_TLB_G(scd));
        sn = su - sl + ss;
        if (ss != 1)
          sn /= ss;
        if (sn < 0)
          sn = 0;
      } else if (n != 0)
        __fort_abort("copy: can't adjust src lbound");
      else
        sn = F90_DPTR_EXTENT_G(scd);

      /* minimize extent */

      n = Min(dn, sn);

      if (n <= 0)
        return NULL;

      /* set up sections with adjusted bounds */

      if (dbogus) {
        du = dl + (n - 1) * ds;
        I8(__fort_set_section)(dc, dx, DIST_ACTUAL_ARG_G(dc), 
                                      DIST_DPTR_TAXIS_G(dcd), dl, du, ds);
      } else if (n != dn)
        __fort_abort("copy: can't adjust dst ubound");

      if (sbogus) {
        su = sl + (n - 1) * ss;
        I8(__fort_set_section)(sc, sx, DIST_ACTUAL_ARG_G(sc), 
                                      DIST_DPTR_TAXIS_G(scd), sl, su, ss);
      } else if (n != sn)
        __fort_abort("copy: can't adjust src ubound");
    }

    if (dbogus) {
      DIST_ACTUAL_ARG_P(dc, NULL);
      I8(__fort_finish_section)(dc);
    }

    if (sbogus) {
      DIST_ACTUAL_ARG_P(sc, NULL);
      I8(__fort_finish_section)(sc);
    }
  }

  if (F90_GSIZE_G(dc) <= 0 && F90_GSIZE_G(sc) <= 0)
    return NULL;

  I8(copy_setup)(&z.dy, db, dc, identity_map);
  I8(copy_setup)(&z.sy, sb, sc, src_axis_map);

  z.permuted = 0;
  for (i = F90_RANK_G(dc); i > 0; --i) {
    dx = z.dy.axis_map[i - 1];
    sx = z.sy.axis_map[i - 1];
    if (dx != i || sx != i)
      z.permuted |= 1 << (i - 1);
    SET_DIM_PTRS(dcd, dc, dx - 1);
    SET_DIM_PTRS(scd, sc, sx - 1);
    if (F90_DPTR_EXTENT_G(dcd) != F90_DPTR_EXTENT_G(scd))
      __fort_abort("copy: section shape mismatch");
  }
#if defined(DEBUG)
  if (F90_GSIZE_G(dc) != F90_GSIZE_G(sc))
    __fort_abort("copy: section size mismatch");
#endif

  if ((z.dy.islocal | z.sy.islocal) == 0)
    return NULL;

  if (LOCAL_MODE) {

    /* only local communication */

    z.cc = __fort_chn_1to1(NULL, 0, lcpu, &_1, &_1, 0, lcpu, &_1, &_1);
  } else if (DIST_REPLICATED_G(dc) | DIST_REPLICATED_G(sc)) {

    /* source or destination replication */

    I8(__fort_describe_replication)(dc, &z.dy.repl);
    I8(__fort_describe_replication)(sc, &z.sy.repl);

#if defined(DEBUG)
    if (__fort_test & DEBUG_COPY) {
      printf("%d copy dst ngrp=%d grpi=%d gstr=", lcpu, z.dy.repl.ngrp,
             z.dy.repl.grpi);
      I8(__fort_show_index)(F90_RANK_G(dc), z.dy.repl.gstr);
      printf(" ndim=%d plow=%d pcnt=", z.dy.repl.ndim, z.dy.repl.plow);
      I8(__fort_show_index)(z.dy.repl.ndim, z.dy.repl.pcnt);
      printf(" pstr=");
      I8(__fort_show_index)(z.dy.repl.ndim, z.dy.repl.pstr);
      printf("\n");

      printf("%d copy src ngrp=%d grpi=%d gstr=", lcpu, z.dy.repl.ngrp,
             z.dy.repl.grpi);
      I8(__fort_show_index)(F90_RANK_G(sc), z.sy.repl.gstr);
      printf(" ndim=%d plow=%d pcnt=", z.sy.repl.ndim, z.sy.repl.plow);
      I8(__fort_show_index)(z.sy.repl.ndim, z.sy.repl.pcnt);
      printf(" pstr=");
      I8(__fort_show_index)(z.sy.repl.ndim, z.sy.repl.pstr);
      printf("\n");
    }
#endif

    ng = z.dy.repl.ngrp + z.sy.repl.ngrp;
    z.dy.ch = (chdr **)__fort_calloc(ng, sizeof(chdr *));
    z.sy.ch = z.dy.ch + z.dy.repl.ngrp;
    z.mych = NULL;
    z.cpu = 0; /* always processor 0 */
  } else {
    tcpus_addr = GET_DIST_TCPUS_ADDR;
    z.cc = __fort_chn_1to1(NULL, 1, 0, tcpus_addr, &_1, 1, 0, tcpus_addr, &_1);
  }

  if (z.sy.islocal) {
#if defined(DEBUG)
    z.xdir = "send";
#endif
    z.xfer = __fort_sendl;
    offset = F90_LBASE_G(sc) - 1 - DIST_SCOFF_G(sc);
    if (F90_RANK_G(sc) > 0)
      I8(copy_loop)(&z, &z.sy, &z.dy, offset, 1, F90_RANK_G(sc));
    else
      I8(copy_xfer)(&z, &z.sy, &z.dy, offset, 1);
  }

  if (z.dy.islocal) {
#if defined(DEBUG)
    z.xdir = "recv";
#endif
    z.xfer = __fort_recvl;
    offset = F90_LBASE_G(dc) - 1 - DIST_SCOFF_G(dc);
    if (F90_RANK_G(dc) > 0)
      I8(copy_loop)(&z, &z.dy, &z.sy, offset, 1, F90_RANK_G(dc));
    else
      I8(copy_xfer)(&z, &z.dy, &z.sy, offset, 1);
  }

  if (!LOCAL_MODE && (DIST_REPLICATED_G(dc) | DIST_REPLICATED_G(sc))) {

    /* source or dest replication.  chain channels together in a
       well-defined order across all processors */

    dx = z.sy.repl.grpi;      /* dest = *,  src = me */
    sx = z.dy.repl.grpi * ng; /* dest = me, src = *  */
    z.cc = NULL;
    for (i = j = 0; i + j < ng;) {
      if (i < z.sy.repl.ngrp && sx < dx) {
        dg = z.dy.repl.grpi;
        sg = i;
        ch = z.sy.ch[sg];
        ++i;
        ++sx;
      } else if (dx == sx) {
        dg = z.dy.repl.grpi;
        sg = z.sy.repl.grpi;
        ch = z.mych;
        ++i;
        ++sx;
        ++j;
        dx += ng;
      } else {
        dg = j;
        sg = z.sy.repl.grpi;
        ch = z.dy.ch[dg];
        ++j;
        dx += ng;
      }
      if (ch != NULL) {
        z.cc = __fort_chain_em_up(z.cc, ch);
      }
    }
    __fort_free(z.dy.ch);
  } else
    __fort_chn_prune(z.cc);

  return z.cc;
}

void ENTFTN(PERMUTE_SECTION, permute_section)(void *rb, void *sb, F90_Desc *rs,
                                              F90_Desc *ss, ...)
{
  char *rp, *sp;
  chdr *ch;
  int i, src_axis_map[MAXDIMS];
  va_list va;

  if (!ISPRESENT(rb))
    __fort_abort("permute_section: result absent or not allocated");

  if (!ISPRESENT(sb))
    __fort_abort("permute_section: source absent or not allocated");

  if (rs == NULL || F90_TAG_G(rs) != __DESC)
    __fort_abort("permute_section: invalid result descriptor");

  if (ss == NULL || F90_TAG_G(ss) != __DESC)
    __fort_abort("permute_section: invalid source descriptor");

  va_start(va, ss);
  for (i = 0; i < F90_RANK_G(ss); ++i)
    src_axis_map[i] = *va_arg(va, __INT_T *);
  va_end(va);
  rp = (char *)rb + DIST_SCOFF_G(rs) * F90_LEN_G(rs);
  sp = (char *)sb + DIST_SCOFF_G(ss) * F90_LEN_G(ss);
  ch = I8(__fort_copy)(rp, sp, rs, ss, src_axis_map);
  __fort_doit(ch);
  __fort_frechn(ch);
}

void ENTFTN(COPY_SECTION, copy_section)(void *rb, void *sb, F90_Desc *rs,
                                        F90_Desc *ss)
{
  char *rp, *sp;
  chdr *ch;

  if (!ISPRESENT(rb))
    __fort_abort("copy_section: result absent or not allocated");

  if (!ISPRESENT(sb))
    __fort_abort("copy_section: source absent or not allocated");

  if (rs == NULL || F90_TAG_G(rs) != __DESC)
    __fort_abort("copy_section: invalid result descriptor");

  if (ss == NULL || F90_TAG_G(ss) != __DESC)
    __fort_abort("copy_section: invalid source descriptor");

  rp = (char *)rb + DIST_SCOFF_G(rs) * F90_LEN_G(rs);
  sp = (char *)sb + DIST_SCOFF_G(ss) * F90_LEN_G(ss);
  ch = I8(__fort_copy)(rp, sp, rs, ss, NULL);
  __fort_doit(ch);
  __fort_frechn(ch);
}

sked *ENTFTN(COMM_COPY, comm_copy)(void *rb, void *sb, F90_Desc *rs,
                                   F90_Desc *ss)
{
  char *rp, *sp;
  chdr *ch;

  if (!ISPRESENT(rb))
    __fort_abort("comm_copy: result absent or not allocated");

  if (!ISPRESENT(sb))
    __fort_abort("comm_copy: source absent or not allocated");

  if (rs == NULL || F90_TAG_G(rs) != __DESC)
    __fort_abort("comm_copy: invalid result descriptor");

  if (ss == NULL || F90_TAG_G(ss) != __DESC)
    __fort_abort("comm_copy: invalid source descriptor");

  rp = (char *)rb + DIST_SCOFF_G(rs) * F90_LEN_G(rs);
  sp = (char *)sb + DIST_SCOFF_G(ss) * F90_LEN_G(ss);
  ch = I8(__fort_copy)(rp, sp, rs, ss, NULL);
  return I8(__fort_comm_sked)(ch, rp, sp, F90_KIND_G(ss), F90_LEN_G(ss));
}

void ENTFTN(TRANSPOSE, transpose)(void *rb, void *sb, F90_Desc *rs,
                                  F90_Desc *ss)
{
  char *rp, *sp;
  chdr *ch;
  int src_axis_map[MAXDIMS] = {2, 1, 3, 4, 5, 6, 7};

  if (!ISPRESENT(rb))
    __fort_abort("transpose: result absent or not allocated");

  if (!ISPRESENT(sb))
    __fort_abort("transpose: source absent or not allocated");

  if (rs == NULL || F90_TAG_G(rs) != __DESC)
    __fort_abort("transpose: invalid result descriptor");

  if (ss == NULL || F90_TAG_G(ss) != __DESC)
    __fort_abort("transpose: invalid source descriptor");

  rp = (char *)rb + DIST_SCOFF_G(rs) * F90_LEN_G(rs);
  sp = (char *)sb + DIST_SCOFF_G(ss) * F90_LEN_G(ss);
  ch = I8(__fort_copy)(rp, sp, rs, ss, src_axis_map);
  __fort_doit(ch);
  __fort_frechn(ch);
}

/* copy source array element to temporary location on processor owning
   result element.  result section must not have scalar subscripts. */

void ENTFTN(COPY_SCALAR, copy_scalar)(void *temp, F90_Desc *rd, ...)
{
  va_list va;
  chdr *ch;
  char *sb, *sp;
  DECL_HDR_PTRS(sd);
  DECL_HDR_VARS(rs);
  DECL_HDR_VARS(ss);
  int dim, index;

  if (!ISPRESENT(temp))
    __fort_abort("copy_scalar: result absent or not allocated");

  if (rd == NULL || F90_TAG_G(rd) != __DESC)
    __fort_abort("copy_scalar: invalid result descriptor");

  /* create rank 0 descriptor locating the result element */

  va_start(va, rd);

  __DIST_INIT_SECTION(rs, 0, rd);
  for (dim = 1; dim <= F90_RANK_G(rd); ++dim) {
    index = *va_arg(va, __INT_T *);
    I8(__fort_set_single)((rs), rd, dim, index, __SINGLE);
  }
  I8(__fort_finish_section)((rs));

  /* really want to use the temp address, so jam the base and scalar
     subscript offsets */

  F90_LBASE_P(rs, 1);
  DIST_SCOFF_P(rs, 0);

  /* create rank 0 descriptor locating the source element */

  sb = va_arg(va, void *);
  sd = va_arg(va, F90_Desc *);

  __DIST_INIT_SECTION(ss, 0, sd);
  for (dim = 1; dim <= F90_RANK_G(sd); ++dim) {
    index = *va_arg(va, __INT_T *);
    I8(__fort_set_single)((ss), sd, dim, index, __SCALAR);
  }
  I8(__fort_finish_section)((ss));

  va_end(va);

  /* copy source type to result */

  F90_KIND_P(rs, F90_KIND_G(ss));
  F90_LEN_P(rs, F90_LEN_G(ss));

  sp = sb + DIST_SCOFF_G(ss) * F90_LEN_G(ss);
  ch = I8(__fort_copy)(temp, sp, rs, ss, NULL);
  __fort_doit(ch);
  __fort_frechn(ch);
}
