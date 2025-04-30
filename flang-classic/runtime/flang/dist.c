/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* dist.c -- distribution management routines */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "scatter.h"
#include "fort_vars.h"

void *ENTFTN(COMM_START, comm_start)(sked **skp, void *rb, F90_Desc *rd,
                                     void *sb, F90_Desc *sd);

sked *ENTFTN(COMM_COPY, comm_copy)(void *rb, void *sb, F90_Desc *rs,
                                   F90_Desc *ss);

__INT_T *I8(__fort_new_gen_block)(F90_Desc *d, int dim)
{
  return &f90DummyGenBlock;
}

void I8(__fort_gen_block_bounds)(F90_Desc *d, int dim, __INT_T *the_olb,
                                 __INT_T *the_oub, __INT_T pcoord)
{

  /* calculate olb and oub for gen_block dimension */

  __INT_T i;
  __INT_T olb, oub, dtStride;
  __INT_T *gb, *tempGB;
  __INT_T pshape;
  __INT_T direction;
  __INT_T dUbound;
  __INT_T tExtent, aExtent;
  DECL_DIM_PTRS(dd);

  SET_DIM_PTRS(dd, d, dim);

  /*pcoord = DIST_DPTR_PCOORD_G(dd);*/

  if (pcoord < 0) {
    *the_oub = 0; /*force off processor grid status*/
    *the_olb = 1;
    return;
  }

  dUbound = DPTR_UBOUND_G(dd);
  dtStride = DIST_DPTR_TSTRIDE_G(dd);
  pshape = DIST_DPTR_PSHAPE_G(dd);
  direction = (dtStride < 0);

  tExtent = (DIST_DPTR_TUB_G(dd) - DIST_DPTR_TLB_G(dd)) + 1;
  aExtent = (dUbound - F90_DPTR_LBOUND_G(dd)) + 1;

  if (tExtent != aExtent) {

    tempGB = I8(__fort_new_gen_block)(d, dim);
    gb = tempGB;

  } else {

    tempGB = 0;
    gb = DIST_DPTR_GEN_BLOCK_G(dd);
  }

  if (gb[pcoord] == 0) {
    oub = 0; /*force off processor grid status*/
    olb = 1;
  } else {
    gb += ((!direction) ? 0 : (pshape - 1));
    olb = F90_DPTR_LBOUND_G(dd);
    oub = *gb + (olb - 1);
    if (!direction)
      i = 0;
    else
      i = (pshape - 1);
    for (; i != pcoord;) {
      olb += *gb;
      if (!direction) {
        ++gb;
        ++i;
      } else {
        --gb;
        --i;
      }
      oub += *gb;
    }
  }

  if (tempGB)
    __fort_free(tempGB);

  *the_olb = olb;
  *the_oub = oub;

#if defined(DEBUG)

  if (olb < F90_DPTR_LBOUND_G(dd) || oub > dUbound) {

    __fort_abort("__fort_gen_block_bounds: bad gen_block");
  }

#endif
}

/* given a divisor n which is a power of two, compute the (positive)
   right shift amount equivalent to division by n.  return -1 if n is
   not a power of two. */

static int
div_shift(int n)
{
  int z = 0;
  int k = 4 * sizeof(int);
  unsigned int m = n;
  while (k) {
    if (m >> k) {
      m >>= k;
      z += k;
    }
    k >>= 1;
  }
  if (n == 1 << z)
    return z;
  else
    return -1; /* n not a power of 2 */
}

/* T3E, T3D, (Cray)Sparc, and T90 w/IEEE have int_mult_upper
   intrinsic; C90 has fast 64-bit multiply allowing emulation */

#ifndef DESC_I8

__INT_T
ENTRY(INT_MULT_UPPER, int_mult_upper)
(__INT_T *x, __INT_T *y)
{
  register unsigned long a, b;
  a = (unsigned long)*x;
  b = (unsigned long)*y;
  return (unsigned long long)a * (unsigned long long)b >> RECIP_FRACBITS;
}

unsigned long
_int_mult_upper(int x, int y)
{
  register unsigned long a, b;
  a = (unsigned long)x;
  b = (unsigned long)y;
  return (unsigned long long)a * (unsigned long long)b >> RECIP_FRACBITS;
}

#endif

/* greatest common divisor */

#ifndef DESC_I8
int
__fort_gcd(int u, int v)
{
  int k, m, n, t;

  if (u < 0)
    u = -u; /* gcd(u,v) = gcd(-u,v) */
  if (v == 0)
    return u; /* gcd(u,0) = abs(u) */
  if (v < 0)
    v = -v;
  if (u == 0)
    return v;

  /* Knuth V.2, 4.5.2, Algorithm B (Binary gcd algorithm). */

  m = u | v;
  m &= -m; /* == least significant bit of (u|v) */

  k = 0;
  n = 4 * sizeof(int);
  while (m != 1) {
    t = m >> n;
    if (t != 0) {
      k += n;
      m = t;
    }
    n >>= 1;
  }

  u >>= k;
  v >>= k;

  if (u & 1)
    t = -v;
  else
    t = u;

  while (t != 0) {
    while ((t & 1) == 0)
      t /= 2;
    if (t > 0)
      u = t;
    else
      v = -t;
    t = u - v;
  }

  return (u << k);
}

/* least common multiple */

int
__fort_lcm(int u, int v)
{
  int p;

  p = u * v;
  return (p == 0 ? 0 : (p > 0 ? p : -p) / __fort_gcd(u, v));
}
#endif

int I8(__fort_owner)(F90_Desc *d, __INT_T *idxv)
{
  return 0;
}

/* Given indices of an array element, return the processor number of
   its nearest owner.  The nearest owner is the one whose processor
   coordinate in each replicated dimension is equal to this
   processor's coordinate. */

__INT_T
ENTFTN(OWNER, owner)
(F90_Desc *d, ...)
{
  va_list va;
  proc *p;
  procdim *pd;
  __INT_T dx, idxv[MAXDIMS], owner, px, repl;

#if defined(DEBUG)
  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("OWNER: invalid descriptor");
#endif
  va_start(va, d);
  for (dx = 0; dx < F90_RANK_G(d); ++dx)
    idxv[dx] = *va_arg(va, __INT_T *);
  va_end(va);

  owner = I8(__fort_owner)(d, idxv);

  /* add the "scalar subscript" component of the processor number
     (which was subtracted in __fort_owner!) */

  p = DIST_DIST_TARGET_G(d);
  repl = DIST_REPLICATED_G(d);
  for (px = 0; repl != 0; repl >>= 1, ++px) {
    if (repl & 1) {
      pd = &p->dim[px];
      if (pd->coord > 0)
        owner += pd->stride * pd->coord;
    }
  }

  return owner;
}

/* Construct a description of the replication of an object over the
   processor grid.  Replication occurs over the subset of processor
   grid dimensions onto which no array or template dimensions have
   been mapped.  This divides the processor grid into sets of
   processors having identical copies of mapped portions of the array
   or template. */

void I8(__fort_describe_replication)(F90_Desc *d, repl_t *r)
{
  DECL_DIM_PTRS(dd);
  proc *p;
  procdim *pd;
  __INT_T grpi, dx, m, ncopies, ndim, ngrp, plow, px;

  ngrp = 1;
  grpi = 0;
  plow = DIST_PBASE_G(d);
  m = DIST_MAPPED_G(d);
  for (dx = 0; dx < F90_RANK_G(d); m >>= 1, ++dx) {
    if (m & 1) {
      SET_DIM_PTRS(dd, d, dx);
      if (DIST_DPTR_PCOORD_G(dd) > 0) {
        grpi += DIST_DPTR_PCOORD_G(dd) * ngrp;
        plow += DIST_DPTR_PCOORD_G(dd) * DIST_DPTR_PSTRIDE_G(dd);
      }
      r->gstr[dx] = ngrp; /* replication group multiplier */
      ngrp *= DIST_DPTR_PSHAPE_G(dd);
    } else
      r->gstr[dx] = 0;
  }
  r->grpi = grpi; /* my replication group index */
  r->ngrp = ngrp; /* number of replication groups */

  p = DIST_DIST_TARGET_G(d);
  ndim = 0;
  ncopies = 1;
  m = DIST_REPLICATED_G(d);
  for (px = 0; m != 0; m >>= 1, ++px) {
    pd = &p->dim[px];
    if (m & 1 && pd->shape > 1) {

      if (pd->coord > 0)
        plow -= pd->coord * pd->stride;

      r->pcnt[ndim] = pd->shape;  /* processor counts */
      r->pstr[ndim] = pd->stride; /* processor strides */
      ncopies *= pd->shape;
      ++ndim;
    }
  }
  r->ncopies = ncopies; /* number of replicated copies */
  r->ndim = ndim;       /* number of replicated proc dims */
  r->plow = plow;       /* my replication group low proc number */
}

/* Iterate over all processors owning a copy of the first element of
   section d.  pcoord is an integer vector of size MAXDIMS which saves the
   processor coordinates in the replication group. */

int I8(__fort_next_owner)(F90_Desc *d, repl_t *r, int *pcoord, int owner)
{
  int i;

  for (i = 0; i < r->ndim; ++i) {
    pcoord[i]++;
    owner += r->pstr[i];
    if (pcoord[i] < r->pcnt[i])
      return owner; /* keep going */
    pcoord[i] = 0;
    owner -= r->pcnt[i] * r->pstr[i];
  }
  return -1; /* finished */
}

/* Determine whether or not a global index in dimension dim of an
   array is local to this processor. */

__LOG_T
ENTFTN(ISLOCAL_IDX, islocal_idx)
(F90_Desc *d, __INT_T *dimb, __INT_T *idxb)
{
  DECL_DIM_PTRS(dd);
  __INT_T dim, idx;

  dim = *dimb;
  idx = *idxb;
#if defined(DEBUG)
  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("ISLOCAL_IDX: invalid descriptor");
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("ISLOCAL_IDX: invalid dimension");
#endif
  SET_DIM_PTRS(dd, d, dim - 1);
  return GET_DIST_TRUE_LOG;
}

/* Determine whether or not the element of section d referenced by the
   indices is local to this processor. */

int I8(__fort_islocal)(F90_Desc *d, __INT_T *idxv)
{
  DECL_DIM_PTRS(dd);
  __INT_T dfmt, dx;

  if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
    return 0;
  for (dx = 0, dfmt = DIST_DFMT_G(d); dfmt != 0; dfmt >>= DFMT__WIDTH, ++dx) {
    SET_DIM_PTRS(dd, d, dx);
  }
  return 1;
}

/* Determine whether or not the array element referenced by the
   indices is local to this processor. */

__LOG_T
ENTFTN(ISLOCAL, islocal)
(F90_Desc *d, ...)
{
  va_list va;
  __INT_T dx, idxv[MAXDIMS];

#if defined(DEBUG)
  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("ISLOCAL: invalid descriptor");
#endif
  va_start(va, d);
  for (dx = 0; dx < F90_RANK_G(d); ++dx)
    idxv[dx] = *va_arg(va, __INT_T *);
  va_end(va);
  return I8(__fort_islocal)(d, idxv) ? GET_DIST_TRUE_LOG : 0;
}

/* Determine the prime owner processor number and the local array
   offset for the element of section d identified by the index
   vector. */

void I8(__fort_localize)(F90_Desc *d, __INT_T *idxv, int *cpu, __INT_T *off)
{
  DECL_DIM_PTRS(dd);
  proc *p;
  procdim *pd;
  __INT_T dfmt, dx, lab, lidx, offset, owner, pcoord = 0, px, repl;

  owner = DIST_PBASE_G(d);
  offset = 0;
  for (dx = 0, dfmt = DIST_DFMT_G(d); dx < F90_RANK_G(d);
       dfmt >>= DFMT__WIDTH, ++dx) {
    SET_DIM_PTRS(dd, d, dx);
#if defined(DEBUG)
    if (idxv[dx] < F90_DPTR_LBOUND_G(dd) || idxv[dx] > DPTR_UBOUND_G(dd)) {
      printf("%d localize: index %d out of bounds %d:%d rank=%d\n",
             GET_DIST_LCPU, idxv[dx], F90_DPTR_LBOUND_G(dd), DPTR_UBOUND_G(dd),
             F90_RANK_G(d));
      __fort_abort((char *)0);
    }
#endif
    lidx = F90_DPTR_SSTRIDE_G(dd) * idxv[dx] + F90_DPTR_SOFFSET_G(dd);

    switch (dfmt & DFMT__MASK) {
    case DFMT_COLLAPSED:
      offset += F90_DPTR_LSTRIDE_G(dd) * (lidx - DIST_DPTR_LAB_G(dd));
      continue;

    default:
      __fort_abort("localize: unsupported dist-format");
    }

    /* find remote processor's lower allocated bound */

    /*
     * We calculated lab in the gen_block case ...
     */

    if (!DIST_DPTR_GEN_BLOCK_G(dd)) {
      lab = DIST_DPTR_CLB_G(dd) + pcoord * DIST_DPTR_BLOCK_G(dd) -
            DIST_DPTR_AOFFSET_G(dd);

      if (DIST_DPTR_ASTRIDE_G(dd) != 1) {
        if (DIST_DPTR_ASTRIDE_G(dd) < 0)
          lab += DIST_DPTR_BLOCK_G(dd) - DIST_DPTR_CYCLE_G(dd);
        if (DIST_DPTR_ASTRIDE_G(dd) == -1)
          lab = -lab;
        else
          lab = Ceil(lab, DIST_DPTR_ASTRIDE_G(dd));
      }
    }

    lab -= DIST_DPTR_NO_G(dd);

    /* accumulate processor number and offset */

    owner += DIST_DPTR_PSTRIDE_G(dd) * pcoord;
    offset += F90_DPTR_LSTRIDE_G(dd) * (lidx - lab);
  }

  /* compensate for the "scalar subscript" component of the
     processor number in pbase. The difference really ought to be
     kept in a separate descriptor item. */

  p = DIST_DIST_TARGET_G(d);
  repl = DIST_REPLICATED_G(d);
  for (px = 0; repl != 0; repl >>= 1, ++px) {
    if (repl & 1) {
      pd = &p->dim[px];
      if (pd->coord > 0)
        owner -= pd->stride * pd->coord;
    }
  }

  *cpu = owner;
  *off = offset;

#if defined(DEBUG)
  if (__fort_test & DEBUG_DIST) {
    printf("%d localize: cpu=%d off=%d + lstride=%d * (lidx=%d - lab=%d)\n",
           GET_DIST_LCPU, owner, offset, F90_DPTR_LSTRIDE_G(dd), lidx, lab);
  }
#endif
}

/* localize an index in dimension dim of the array described by d */

void ENTFTN(LOCALIZE_DIM, localize_dim)(F90_Desc *d, __INT_T *dimp,
                                        __INT_T *idxp, __INT_T *pcoordp,
                                        __INT_T *lindexp)
{
  DECL_DIM_PTRS(dd);
  __INT_T dim, idx, lab = 0, lidx, pcoord = 0;

  dim = *dimp;
  idx = *idxp;

  SET_DIM_PTRS(dd, d, dim - 1);
#if defined(DEBUG)
  if (idx < F90_DPTR_LBOUND_G(dd) || idx > DPTR_UBOUND_G(dd))
    __fort_abort("LOCALIZE_DIM: index out of bounds");
#endif

  lidx = F90_DPTR_SSTRIDE_G(dd) * idx + F90_DPTR_SOFFSET_G(dd);

  switch (DFMT(d, dim)) {
  case DFMT_COLLAPSED:
    *pcoordp = 0;
    *lindexp = lidx;
    return;

  default:
    __fort_abort("LOCALIZE_DIM: unsupported dist-format");
  }

  /* remote proc's lab calculated for gen_block above */

  if (!DIST_DPTR_GEN_BLOCK_G(dd)) {

    /* find remote processor's lower allocated bound */

    lab = DIST_DPTR_CLB_G(dd) + pcoord * DIST_DPTR_BLOCK_G(dd) -
          DIST_DPTR_AOFFSET_G(dd);
    if (DIST_DPTR_ASTRIDE_G(dd) != 1) {
      if (DIST_DPTR_ASTRIDE_G(dd) < 0)
        lab += DIST_DPTR_BLOCK_G(dd) - DIST_DPTR_CYCLE_G(dd);
      if (DIST_DPTR_ASTRIDE_G(dd) == -1)
        lab = -lab;
      else
        lab = Ceil(lab, DIST_DPTR_ASTRIDE_G(dd));
    }
  }
  lab -= DIST_DPTR_NO_G(dd);

  /* return the remote processor's coordinate and the local index on
     this processor with the same element offset */

  *pcoordp = pcoord;
  *lindexp = lidx - lab + DIST_DPTR_LAB_G(dd);
}

/* Given the indices of an element of section d, return its local
   offset or -1 if it is not local.  The offset does not reflect any
   scalar subscript. */

__INT_T
I8(__fort_local_offset)(F90_Desc *d, __INT_T *idxv)
{
  DECL_DIM_PTRS(dd);
  __INT_T dx, idx, lidx, offset;

  if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
    return -1;

  offset = F90_LBASE_G(d) - 1;

  if (F90_FLAGS_G(d) & __SEQUENCE) {
    for (dx = F90_RANK_G(d); --dx >= 0;) {
      SET_DIM_PTRS(dd, d, dx);
      lidx = F90_DPTR_SSTRIDE_G(dd) * idxv[dx] + F90_DPTR_SOFFSET_G(dd);
      offset += F90_DPTR_LSTRIDE_G(dd) * lidx;
    }
    return offset;
  }

  for (dx = 0; dx < F90_RANK_G(d); ++dx) {
    SET_DIM_PTRS(dd, d, dx);
    idx = idxv[dx];
#if defined(DEBUG)
    if (idx < F90_DPTR_LBOUND_G(dd) || idx > DPTR_UBOUND_G(dd))
      __fort_abort("local_offset: index out of bounds");
#endif
    lidx = F90_DPTR_SSTRIDE_G(dd) * idx + F90_DPTR_SOFFSET_G(dd);
    offset += F90_DPTR_LSTRIDE_G(dd) * lidx;
  }
#if defined(DEBUG)
  if (__fort_test & DEBUG_DIST) {
    printf("%d local_offset: offset=%d + lstride=%d * lidx=%d\n",
           GET_DIST_LCPU, offset, F90_DPTR_LSTRIDE_G(dd), lidx);
  }
#endif
  return offset;
}

/* Given the indices of an element of section d, return its local
   address or NULL if it is not local.  The base address is assumed to
   be adjusted for scalar subscripts. */

void *I8(__fort_local_address)(void *base, F90_Desc *d, __INT_T *idxv)
{
  DECL_DIM_PTRS(dd);
  __INT_T dfmt, dx, idx, lidx, offset;

  if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
    return NULL;

  offset = F90_LBASE_G(d) - 1 - DIST_SCOFF_G(d);

  if (F90_FLAGS_G(d) & __SEQUENCE) {
    for (dx = F90_RANK_G(d); --dx >= 0;) {
      SET_DIM_PTRS(dd, d, dx);
      idx = idxv[dx];
      lidx = F90_DPTR_SSTRIDE_G(dd) * idx + F90_DPTR_SOFFSET_G(dd);
      offset += F90_DPTR_LSTRIDE_G(dd) * lidx;
    }
    return (char *)base + offset * F90_LEN_G(d);
  }

  for (dx = 0, dfmt = DIST_DFMT_G(d); dx < F90_RANK_G(d);
       dfmt >>= DFMT__WIDTH, ++dx) {
    SET_DIM_PTRS(dd, d, dx);
    idx = idxv[dx];
#if defined(DEBUG)
    if (DPTR_UBOUND_G(dd) < 0)
      __fort_abort("local_address: index out of bounds");
#endif
    lidx = F90_DPTR_SSTRIDE_G(dd) * idx + F90_DPTR_SOFFSET_G(dd);

    switch (dfmt & DFMT__MASK) {
    case DFMT_COLLAPSED:
      break;
    case DFMT_BLOCK:
    case DFMT_BLOCK_K:
    case DFMT_GEN_BLOCK:
      if (idx < DIST_DPTR_OLB_G(dd) || idx > DIST_DPTR_OUB_G(dd))
        return NULL;
      break;

    default:
      __fort_abort("local_offset: unsupported dist-format");
    }
    offset += F90_DPTR_LSTRIDE_G(dd) * lidx;
  }
  return (char *)base + offset * F90_LEN_G(d);
}

/* Localize a global index in dimension dim of array a.  This is only
   necessary for dimensions with cyclic or block-cyclic distributions.
   It is assumed that the index is local */

__INT_T
ENTFTN(LOCALIZE_INDEX, localize_index)
(F90_Desc *d, __INT_T *dimb, __INT_T *idxb)
{
  DECL_DIM_PTRS(dd);
  int dim, idx, lidx;

  dim = *dimb;
  idx = *idxb;
#if defined(DEBUG)
  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("LOCALIZE_INDEX: invalid descriptor");
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("LOCALIZE_INDEX: invalid dimension");
#endif
  SET_DIM_PTRS(dd, d, dim - 1);
#if defined(DEBUG)
  if (idx < F90_DPTR_LBOUND_G(dd) || idx > DPTR_UBOUND_G(dd))
    __fort_abort("LOCALIZE_INDEX: index out of bounds");
#endif
  lidx = F90_DPTR_SSTRIDE_G(dd) * idx + F90_DPTR_SOFFSET_G(dd);
  return lidx;
}

/* Given the loop bounds l, u, and s that range over dimension dim of
   section d, return cycle loop lower bound cl, iteration count cn,
   stride cs, initial cyclic offset lof, and cyclic offset stride los
   that localize the loop to this processor. */

static int I8(cyclic_setup)(F90_Desc *d, __INT_T dim, __INT_T l, __INT_T u,
                            __INT_T s, __INT_T *pcl, __INT_T *pcs,
                            __INT_T *plof, __INT_T *plos)
{
  DECL_DIM_PTRS(dd);
  __INT_T cl = 0, cn = 0, cs, cu, lof = 0, los = 0, n, ts;

  SET_DIM_PTRS(dd, d, dim - 1);

  /* adjust lower bound to fall within array index range */

  if (s > 0) {
    n = F90_DPTR_LBOUND_G(dd) - l + s - 1;
    if (n > 0) {
      if (s != 1)
        n /= s;
      l += n * s;
    }
  } else {
    n = DPTR_UBOUND_G(dd) - l + s + 1;
    if (n < 0) {
      if (s == -1)
        n = -n;
      else
        n /= s;
      l += n * s;
    }
  }

  ts = DIST_DPTR_TSTRIDE_G(dd) * s; /* stride in ultimate template */

  cs = (ts < 0) ? -DIST_DPTR_CYCLE_G(dd)
                : DIST_DPTR_CYCLE_G(dd); /* cycle stride */

  /* check for zero-trip loop, no local data, or not cyclic */

  if (s > 0 ? (l > u || l > DIST_DPTR_OUB_G(dd) || u < DIST_DPTR_OLB_G(dd))
            : (l < u || l < DIST_DPTR_OLB_G(dd) || u > DIST_DPTR_OUB_G(dd))) {

    /* no local data or zero-trip loop */

    cl = DIST_DPTR_CLB_G(dd);
    cu = cl - cs;
    cn = lof = los = 0;
  } else
    switch (DFMT(d, dim)) {

    default:
      __fort_abort("cyclic_setup: unsupported dist-format");
    }
#if defined(DEBUG)
  if (__fort_test & DEBUG_DIST)
    printf("%d cyclic dim=%d %d:%d:%d -> %d:%d:%d cn=%d lof=%d los=%d\n",
           GET_DIST_LCPU, dim, l, u, s, cl, cu, cs, cn, lof, los);
#endif
  *pcl = cl;
  *pcs = cs;
  *plof = lof;
  *plos = los;
  return cn;
}

/* Cache the cycle loop bounds in the section descriptor. The cached
   parameters describe loops over the entire section. */

void I8(__fort_cycle_bounds)(F90_Desc *d)
{
  DECL_DIM_PTRS(dd);
  __INT_T dim;

  for (dim = F90_RANK_G(d); dim > 0; --dim) {
    if ((~DIST_CACHED_G(d) >> (dim - 1)) & 1) {
      SET_DIM_PTRS(dd, d, dim - 1);
      DIST_DPTR_CN_P(
          dd, I8(cyclic_setup)(d, dim, F90_DPTR_LBOUND_G(dd), DPTR_UBOUND_G(dd),
                               1, &DIST_DPTR_CL_G(dd), &DIST_DPTR_CS_G(dd),
                               &DIST_DPTR_CLOF_G(dd), &DIST_DPTR_CLOS_G(dd)));
    }
  }
  DIST_CACHED_P(d, (DIST_CACHED_G(d) | ~(-1 << F90_RANK_G(d))));
}

/* Set the lower and upper cycle loop bounds, cycle loop stride,
   cyclic index offset, and cyclic offset stride and return the cycle
   loop trip count (number of local blocks) for the loop specified by
   l:u:s over dimension dim of section d.  For a stride 1 loop over
   the entire dimension, the cycle loop parameters are cached in the
   descriptor. */

__INT_T
I8(__fort_cyclic_loop)(F90_Desc *d, __INT_T dim, __INT_T l, __INT_T u, 
                      __INT_T s, __INT_T *cl, __INT_T *cu, __INT_T *cs, 
                      __INT_T *clof, __INT_T *clos)
{
  DECL_DIM_PTRS(dd);
  __INT_T cn, m;

#if defined(DEBUG)
  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("cyclic_loop: invalid descriptor");
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("cyclic_loop: invalid dimension");
  if (s == 0)
    __fort_abort("cyclic_loop: invalid stride");
#endif

  SET_DIM_PTRS(dd, d, dim - 1);

  if (l == F90_DPTR_LBOUND_G(dd) && u == DPTR_UBOUND_G(dd) && s == 1) {

    /* loop bounds match section bounds */

    m = 1 << (dim - 1);
    if (~DIST_CACHED_G(d) & m) {

      /* not cached - store cycle loop bounds in descriptor */

      DIST_DPTR_CN_P(dd,
                    I8(cyclic_setup)(d, dim, l, u, s, &DIST_DPTR_CL_G(dd),
                                     &DIST_DPTR_CS_G(dd), &DIST_DPTR_CLOF_G(dd),
                                     &DIST_DPTR_CLOS_G(dd)));
      DIST_CACHED_P(d, DIST_CACHED_G(d) | m);
    }

    /* return previously cached cycle loop bounds */

    *cl = DIST_DPTR_CL_G(dd);
    *cs = DIST_DPTR_CS_G(dd);
    *clof = DIST_DPTR_CLOF_G(dd);
    *clos = DIST_DPTR_CLOS_G(dd);

    cn = DIST_DPTR_CN_G(dd);
  } else {

    /* loop bounds don't match section bounds */

    cn = I8(cyclic_setup)(d, dim, l, u, s, cl, cs, clof, clos);
  }

  *cu = *cl + (cn - 1) * (*cs);
  return cn;
}

void ENTFTN(CYCLIC_LOOP, cyclic_loop)(F90_Desc *d, __INT_T *dim, __INT_T *l,
                                      __INT_T *u, __INT_T *s, __INT_T *cl,
                                      __INT_T *cu, __INT_T *cs, __INT_T *clof,
                                      __INT_T *clos)
{
  __INT_T xcl, xcu, xcs, xclof, xclos;

  (void)I8(__fort_cyclic_loop)(d, *dim, *l, *u, *s, &xcl, &xcu, &xcs, &xclof,
                               &xclos);
  *cl = xcl;
  *cu = xcu;
  *cs = xcs;
  *clof = xclof;
  *clos = xclos;
}

/* Given loop bounds l:u:s over dimension dim of array d (not
   necessarily spanning the entire dimension), localize the loop to
   the local block specified by cycle index ci (which must increment
   through the outer cycle loop in order to cover all elements in the
   l:u:s section). Set block loop bounds bl, bu. */

void I8(block_setup)(F90_Desc *d, int dim, __INT_T l, __INT_T u, int s, int ci,
                     __INT_T *bl, __INT_T *bu)
{
  DECL_DIM_PTRS(dd);
  __INT_T bb, lob, uob, m, n;
#if defined(DEBUG)
  __INT_T gl = l;
  __INT_T gu = u;

  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("block_setup: invalid descriptor");
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("block_setup: invalid dimension");
  if (s == 0)
    __fort_abort("block_setup: invalid stride");
#endif

  SET_DIM_PTRS(dd, d, dim - 1);

  /* adjust lower bound to fall within array index range */

  m = s > 0 ? F90_DPTR_LBOUND_G(dd) - 1 : DPTR_UBOUND_G(dd) + 1;
  n = m - l + s;
  if (s != 1)
    n /= s;
  if (n < 0)
    n = 0;
  l += n * s;

  switch (DFMT(d, dim)) {
  case DFMT_COLLAPSED:
  case DFMT_BLOCK:
  case DFMT_BLOCK_K:
  case DFMT_GEN_BLOCK:
    lob = DIST_DPTR_OLB_G(dd);
    uob = DIST_DPTR_OUB_G(dd);
    break;

  case DFMT_CYCLIC:
  case DFMT_CYCLIC_K:
    m = DIST_DPTR_TSTRIDE_G(dd);
    bb = DIST_DPTR_BLOCK_G(dd) - 1;
    if ((m ^ s) < 0)
      bb = -bb;
    lob = uob = ci - DIST_DPTR_TOFFSET_G(dd);
    if (s > 0)
      uob += bb;
    else
      lob += bb;
    if (m != 1) {
      lob = Ceil(lob, m);
      uob = Floor(uob, m);
    }
    break;

  default:
    __fort_abort("block_setup: unsupported dist-format");
  }

  if (s > 0) {
    if (l < lob) {
      if (s != 1)
        l += s * ((lob - l + s - 1) / s);
      else
        l = lob;
    }
    if (u > uob)
      u = uob;
  } else {
    if (l > uob) {
      if (s != -1)
        l += s * ((uob - l + s + 1) / s);
      else
        l = uob;
    }
    if (u < lob)
      u = lob;
  }
#if defined(DEBUG)
  if (__fort_test & DEBUG_DIST)
    printf("%d block dim=%d %d:%d:%d ci=%d -> %d:%d:%d\n", GET_DIST_LCPU, dim,
           gl, gu, s, ci, l, u, s);
#endif
  *bl = l;
  *bu = u;
}

void ENTFTN(BLOCK_LOOP, block_loop)(F90_Desc *d, __INT_T *dim, __INT_T *l,
                                    __INT_T *u, __INT_T *s, __INT_T *ci,
                                    __INT_T *bl, __INT_T *bu)
{
  __INT_T xbl, xbu;

  I8(block_setup)(d, *dim, *l, *u, *s, *ci, &xbl, &xbu);
  *bl = xbl;
  *bu = xbu;
}

/* Same as block_setup, but return loop trip count. */

int I8(__fort_block_loop)(F90_Desc *d, int dim, __INT_T l, __INT_T u, int s,
                          __INT_T ci, __INT_T *bl, __INT_T *bu)
{
  int bn;

  I8(block_setup)(d, dim, l, u, s, ci, bl, bu);
  bn = (*bu - *bl + s) / s;

  return bn;
}

/* Set the lower and upper index bounds and return the number of local
   elements for the local block at cycle index ci in dimension dim of
   section d. Same as block_setup, but assumes a stride 1 loop over
   the entire dimension and returns the loop trip count. */

__INT_T
I8(__fort_block_bounds)(F90_Desc *d, __INT_T dim, __INT_T ci, 
                       __INT_T *bl, __INT_T *bu)
{
  DECL_DIM_PTRS(dd);

  SET_DIM_PTRS(dd, d, dim - 1);
  I8(block_setup)(d, dim, F90_DPTR_LBOUND_G(dd), DPTR_UBOUND_G(dd), 
                    1, ci, bl, bu);
  return *bu - *bl + 1;
}

/* Given loop bounds and stride that range over dimension dim of array
   a which does NOT have a cyclic distribution, return new loop bounds
   bl, bu that localize the bounds to this processor. */

void ENTFTN(LOCALIZE_BOUNDS,
            localize_bounds)(F90_Desc *d, __INT_T *gdim, __INT_T *gl,
                             __INT_T *gu, __INT_T *gs, __INT_T *bl, __INT_T *bu)
{
  DECL_DIM_PTRS(dd);
  int dim, l, u, s, m, n, lob, uob;

  dim = *gdim;
  l = *gl;
  u = *gu;
  s = *gs;

#if defined(DEBUG)
  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("LOCALIZE_BOUNDS: invalid descriptor");
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("LOCALIZE_BOUNDS: invalid dimension");
  if (s == 0)
    __fort_abort("LOCALIZE_BOUNDS: invalid stride");
#endif

  SET_DIM_PTRS(dd, d, dim - 1);

  /* adjust lower bound to fall within array index range */

  m = s > 0 ? F90_DPTR_LBOUND_G(dd) - 1 : DPTR_UBOUND_G(dd) + 1;
  n = m - l + s;
  if (s != 1)
    n /= s;
  if (n < 0)
    n = 0;
  l += n * s;

  lob = DIST_DPTR_OLB_G(dd);
  uob = DIST_DPTR_OUB_G(dd);
  if (s == 1) {
    if (l < lob)
      l = lob;
    if (u > uob)
      u = uob;
  } else if (s > 0) {
    if (l < lob)
      l += s * ((lob - l + s - 1) / s);
    if (u > uob)
      u = uob;
  } else {
    if (l > uob) {
      if (s != -1)
        l += s * ((uob - l + s + 1) / s);
      else
        l = uob;
    }
    if (u < lob)
      u = lob;
  }
  *bl = l;
  *bu = u;
}

/* Create a new processor descriptor */

static void
proc_setup(proc *p)
{
  procdim *pd;
  int i, m, size;
  char msg[80];

  size = 1;
  for (i = 0; i < p->rank; ++i) {
    pd = &p->dim[i];
    pd->shape_shift = div_shift(pd->shape);
    pd->shape_recip = RECIP(pd->shape);
    pd->stride = size;
    size *= pd->shape;
  }
  p->size = size;
  if (p->base + size > GET_DIST_TCPUS) {
    sprintf(msg, "Too few processors.  Need %d, got %d.", p->base + size,
            GET_DIST_TCPUS);
    __fort_abort(msg);
  }
  m = GET_DIST_LCPU - p->base;
  if (m >= 0 && m < size) {
    for (i = 0; i < p->rank; ++i) {
      pd = &p->dim[i];
      RECIP_DIVMOD(&m, &pd->coord, m, pd->shape);
    }
  } else {
    for (i = 0; i < p->rank; ++i) {
      pd = &p->dim[i];
      pd->coord = -1;
    }
    p->flags |= __OFF_TEMPLATE;
  }
}

void ENTFTN(PROCESSORS, processors)(proc *p, __INT_T *rankp, ...)
{
  va_list va;
  procdim *pd;
  __INT_T i, rank;

  rank = *rankp;
#if defined(DEBUG)
  if (rank < 0 || rank > MAXDIMS)
    __fort_abort("PROCESSORS: invalid rank");
#endif
  p->tag = __PROC;
  p->rank = rank;
  p->flags = 0;
  p->base = 0;
  va_start(va, rankp);
  for (i = 0; i < rank; ++i) {
    pd = &p->dim[i];
    pd->shape = *va_arg(va, __INT_T *);
    if (pd->shape < 1)
      __fort_abort("PROCESSORS: invalid shape");
  }
  va_end(va);
  proc_setup(p);
}

/* Create a default processor grid of given rank.  Factor the number
   of processors into the squarest possible set of rank terms, in
   ascending order.  Default processor descriptors are cached by
   rank and never need to be freed. */

#if !defined(DESC_I8)

static proc *default_proc_list[MAXDIMS + 1] = {NULL};

#define NPRIMES 31
static int prime[NPRIMES] = {2,  3,  5,  7,   11,  13,  17,  19,  23, 29, 31,
                             37, 41, 43, 47,  53,  59,  61,  67,  71, 73, 79,
                             83, 89, 97, 101, 103, 107, 109, 113, 127};

proc *
__fort_defaultproc(int rank)
{
  proc *p;
  int i, k, np, power[NPRIMES], shape[MAXDIMS];

  if (rank < 0 || rank > MAXDIMS)
    __fort_abort("DEFAULTPROC: invalid processor rank");

  if (rank == 0)
    rank = 1; /* substitute rank 1 for rank 0 */

  p = default_proc_list[rank];
  if (p != NULL)
    return p;

  for (i = 0; i < rank; ++i)
    shape[i] = 1;

  np = GET_DIST_TCPUS;
  if (rank > 1 && np > 1) {

    /* first determine the power of each prime factor */

    power[0] = 0; /* powers of two */
    while ((np & 1) == 0) {
      power[0]++;
      np >>= 1;
    }
    for (k = 1; k < NPRIMES && np >= prime[k]; ++k) {
      power[k] = 0;
      while (np % prime[k] == 0) {
        power[k]++;
        np /= prime[k];
      }
    }

    /* now construct the shape vector, using the prime factors
       from largest to smallest.  keep the shape vector sorted in
       ascending order at each step. */

    shape[rank - 1] = np;

    while (--k >= 0) {
      while (--power[k] >= 0) {
        shape[0] *= prime[k];
        for (i = 1; i < rank && shape[i - 1] > shape[i]; ++i) {
          int t = shape[i - 1];
          shape[i - 1] = shape[i];
          shape[i] = t;
        }
      }
    }
  } else if (rank == 1)
    shape[0] = np;

  /* set up the descriptor */

  p = (proc *)__fort_malloc(sizeof(proc) - (MAXDIMS - rank) * sizeof(procdim));
  p->tag = __PROC;
  p->rank = rank;
  p->flags = 0;
  p->base = 0;
  for (i = 0; i < rank; ++i)
    p->dim[i].shape = shape[i];
  proc_setup(p);
  default_proc_list[rank] = p;
  return p;
}

/* rank 0 processor descriptor for the local processor */

static proc *local_proc;

proc *
__fort_localproc()
{
  if (local_proc == NULL) {
    local_proc =
        (proc *)__fort_malloc(sizeof(proc) - MAXDIMS * sizeof(procdim));
    local_proc->tag = __PROC;
    local_proc->rank = 0;
    local_proc->flags = __LOCAL;
    local_proc->base = GET_DIST_LCPU;
    proc_setup(local_proc);
  }
  return local_proc;
}
#endif

void I8(__fort_set_alignment)(F90_Desc *d, __INT_T dim, __INT_T lbound,
                              __INT_T ubound, __INT_T taxis, __INT_T tstride,
                              __INT_T toffset, ...)
{
  DECL_DIM_PTRS(dd);
  __INT_T extent;

#if defined(DEBUG)
  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("set_alignment: invalid descriptor");
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("set_alignment: invalid dim");
#endif

  extent = ubound - lbound + 1;
  if (extent < 0) {
    lbound = 1;
    ubound = 0;
    extent = 0;
  }

  SET_DIM_PTRS(dd, d, dim - 1);

  F90_DPTR_LBOUND_P(dd, lbound);
  DPTR_UBOUND_P(dd, ubound);
  F90_DPTR_SSTRIDE_P(dd, 1); /* section stride */
  F90_DPTR_SOFFSET_P(dd, 0); /* section offset */
  F90_DPTR_LSTRIDE_P(dd, 0);

#if defined(DEBUG)
  if (__fort_test & DEBUG_DIST)
    printf("%d set_align dim=%d lb=%d ub=%d tx=%d st=%d of=%d"
           " tlb=%d tub=%d clb=%d cno=%d olb=%d oub=%d\n",
           GET_DIST_LCPU, dim, F90_DPTR_LBOUND_G(dd), DPTR_UBOUND_G(dd),
           DIST_DPTR_TAXIS_G(dd), DIST_DPTR_TSTRIDE_G(dd), DIST_DPTR_TOFFSET_G(dd),
           DIST_DPTR_TLB_G(dd), DIST_DPTR_TUB_G(dd), DIST_DPTR_CLB_G(dd),
           DIST_DPTR_CNO_G(dd), DIST_DPTR_OLB_G(dd), DIST_DPTR_OUB_G(dd));
#endif
}

/* Set the local bounds for dimension dim of array d to use the local
   storage associated with the corresponding dimension of array a. */

void I8(__fort_use_allocation)(F90_Desc *d, __INT_T dim, __INT_T no, __INT_T po,
                               F90_Desc *a)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T k;
#if defined(DEBUG)
  __INT_T aextent, dextent;
#endif

#if defined(DEBUG)
  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("use_allocation: invalid descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("use_allocation: invalid array descriptor");
  if (F90_RANK_G(a) != F90_RANK_G(d))
    __fort_abort("use_allocation: descriptor ranks differ");
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("use_allocation: invalid dim");
#endif

  /* array descriptor; not template */
  F90_FLAGS_P(d, F90_FLAGS_G(d) & ~__TEMPLATE);

  DIST_NONSEQUENCE_P(d, DIST_NONSEQUENCE_G(a));

  SET_DIM_PTRS(dd, d, dim - 1);
  SET_DIM_PTRS(ad, a, dim - 1);

#if defined(DEBUG)
  if (F90_DPTR_SSTRIDE_G(ad) != 1)
    __fort_abort("use_allocation: can't use strided section");
  if (no > DIST_DPTR_NO_G(ad) || po > DIST_DPTR_PO_G(ad))
    __fort_abort("use_allocation: can't increase overlaps");
#endif

  if (DPTR_UBOUND_G(dd) < F90_DPTR_LBOUND_G(dd)) { /* zero-size */
#if defined(DEBUG)
    if (DIST_DPTR_OUB_G(dd) != DIST_DPTR_OLB_G(dd) - 1 ||
        DIST_DPTR_UAB_G(dd) != DIST_DPTR_LAB_G(dd) - 1)
      __fort_abort("use_allocation: bad bounds for zero-size");

#endif
    DIST_DPTR_NO_P(dd, 0);     /* negative overlap allowance */
    DIST_DPTR_PO_P(dd, 0);     /* positive overlap allowance */
    DIST_DPTR_COFSTR_P(dd, 0); /* cyclic offset stride */
  } else {
    DIST_DPTR_NO_P(dd, no); /* negative overlap allowance */
    DIST_DPTR_PO_P(dd, po); /* positive overlap allowance */

    if (~F90_FLAGS_G(a) & F90_FLAGS_G(d) & __LOCAL) { /* coercing to local */
      DIST_DPTR_COFSTR_P(dd, 0);
      k = DIST_DPTR_OLB_G(dd) - DIST_DPTR_OLB_G(ad);
    } else {
      DIST_DPTR_COFSTR_P(dd, DIST_DPTR_COFSTR_G(ad));
      k = F90_DPTR_LBOUND_G(dd) - F90_DPTR_LBOUND_G(ad);
    }
    k -= F90_DPTR_SOFFSET_G(ad);
    DIST_DPTR_LAB_P(dd, (DIST_DPTR_LAB_G(ad) + (DIST_DPTR_NO_G(ad) - no) + k));
    DIST_DPTR_UAB_P(dd, (DIST_DPTR_UAB_G(ad) - (DIST_DPTR_PO_G(ad) - po) + k));

#if defined(DEBUG)
    aextent = DIST_DPTR_UAB_G(ad) - DIST_DPTR_LAB_G(ad) + 1;
    dextent = DIST_DPTR_UAB_G(dd) - DIST_DPTR_LAB_G(dd) + 1;
    if (dim < F90_RANK_G(d)) {
      if (aextent != dextent)
        __fort_abort("use_allocation: allocated extent changed");
    } else if (dextent > aextent)
      __fort_abort("use_allocation: allocated extent increased");
#endif
  }

#if defined(DEBUG)
  if (__fort_test & DEBUG_DIST)
    printf("%d use_alloc dim=%d lb=%d ub=%d no=%d po=%d"
           " lab=%d uab=%d cofstr=%d\n",
           GET_DIST_LCPU, dim, F90_DPTR_LBOUND_G(dd), DPTR_UBOUND_G(dd),
           DIST_DPTR_NO_G(dd), DIST_DPTR_PO_G(dd), DIST_DPTR_LAB_G(dd),
           DIST_DPTR_UAB_G(dd), DIST_DPTR_COFSTR_G(dd));
#endif
}

/* Map descriptor d onto the single/scalar-subscript coordinate idx in
   axis dim of template/array a.  For a scalar subscript, adjust the
   index base offsets */

void I8(__fort_set_single)(F90_Desc *d, F90_Desc *a, __INT_T dim, __INT_T idx,
                           _set_single_enum what)
{
  DECL_DIM_PTRS(ad);
  __INT_T k, lidx;

#if defined(DEBUG)
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("set_single: invalid array descriptor");
  if (dim < 1 || dim > F90_RANK_G(a))
    __fort_abort("set_single: invalid array dimension");
#endif

  SET_DIM_PTRS(ad, a, dim - 1);

  /* localize scalar subscript */
  if (what != __SINGLE) {
    /* adjust index base offsets */
    lidx = F90_DPTR_SSTRIDE_G(ad) * idx + F90_DPTR_SOFFSET_G(ad);
    k = F90_DPTR_LSTRIDE_G(ad) * (lidx - F90_DPTR_LBOUND_G(ad));
    F90_LBASE_P(d, F90_LBASE_G(d) + k - DIST_DPTR_LOFFSET_G(ad));
  }

#if defined(DEBUG)
  if (__fort_test & DEBUG_DIST)
    printf("%d set_single dim=%d idx=%d pbase=%d lbase=%d scoff=%d%s\n",
           GET_DIST_LCPU, dim, idx, DIST_PBASE_G(d), F90_LBASE_G(d),
           DIST_SCOFF_G(d),
           F90_FLAGS_G(d) & __OFF_TEMPLATE ? " OFF_TEMPLATE" : "");
#endif
}

/* Compute the global array size, local array size, local index
   multiplier and offset, and local index base offset.  This routine
   should not be called for templates. */

void I8(__fort_finish_descriptor)(F90_Desc *d)
{
  DECL_DIM_PTRS(dd);
  __INT_T gsize, i, lextent, lsize, lbase;
  __INT_T rank = F90_RANK_G(d);

  gsize = lsize = lbase = 1;
  for (i = 0; i < rank; ++i) {
    SET_DIM_PTRS(dd, d, i);
    gsize *= F90_DPTR_EXTENT_G(dd);
    F90_DPTR_LSTRIDE_P(dd, lsize);
    DIST_DPTR_LOFFSET_P(dd,
                       -lsize * DIST_DPTR_LAB_G(dd)); /* local index offset */
    lbase -= lsize * F90_DPTR_LBOUND_G(dd);
    lextent = F90_DPTR_EXTENT_G(dd);
    if (lextent > 0)
      lsize *= lextent;
    else
      lsize = 0;
  }
  F90_GSIZE_P(d, gsize); /* global array size */
  F90_LSIZE_P(d, lsize); /* local array size */
  F90_LBASE_P(d, lbase);

  /* global heap block multiplier is the per-processor heap block
     size divided by the data item length */

  if (__fort_heap_block > 0 && F90_LEN_G(d) > 0) {
    if (F90_KIND_G(d) == __STR || F90_KIND_G(d) == __DERIVED ||
        F90_KIND_G(d) == __NONE)
      DIST_HEAPB_P(d, __fort_heap_block / F90_LEN_G(d));
    else
      DIST_HEAPB_P(d, __fort_heap_block >> GET_DIST_SHIFTS(F90_KIND_G(d)));
    if (DIST_HEAPB_G(d) <= 0)
      __fort_abort("heap block overflow; -heapz too large");
  } else
    DIST_HEAPB_P(d, 0);
}

/* Map axis ddim of descriptor d onto section l:u:s of dimension adim
   of array a. */

/* for F90 */
void I8(__fort_set_sectionx)(F90_Desc *d, __INT_T ddim, F90_Desc *a,
                             __INT_T adim, __INT_T l, __INT_T u, __INT_T s,
                             __INT_T noreindex)
{
  DECL_DIM_PTRS(dd);
  DECL_DIM_PTRS(ad);
  __INT_T extent, myoffset;

#if defined(DEBUG)
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("set_section: invalid array descriptor");
  if (adim < 1 || adim > F90_RANK_G(a))
    __fort_abort("set_section: invalid array dimension");
#endif

  SET_DIM_PTRS(ad, a, adim - 1);
  SET_DIM_PTRS(dd, d, ddim - 1);

#if defined(DEBUG)
  if ((F90_FLAGS_G(d) & (__SEQUENCE | __ASSUMED_SIZE | __BOGUSBOUNDS)) == 0 &&
      (l < F90_DPTR_LBOUND_G(ad) || u > DPTR_UBOUND_G(ad)))
    __fort_abort("set_section: index out of bounds");
#endif

  extent = u - l + s; /* section extent */
  if (s != 1) {
    if (s == -1)
      extent = -extent;
    else
      extent /= s;
  }
  if (extent < 0)
    extent = 0;

  if (noreindex && s == 1) {
    F90_DPTR_LBOUND_P(dd, l);                   /* lower bound */
    DPTR_UBOUND_P(dd, extent == 0 ? l - 1 : u); /* upper bound */
    myoffset = 0;
  } else {
    F90_DPTR_LBOUND_P(dd, 1);  /* lower bound */
    DPTR_UBOUND_P(dd, extent); /* upper bound */
    myoffset = l - s;
  }

/* adjust section stride and offset; local array index mapping is
   unchanged */

  /* no longer need section stride/section offset */
  F90_DPTR_SSTRIDE_P(dd, F90_DPTR_SSTRIDE_G(ad));
  F90_DPTR_SOFFSET_P(dd, (F90_DPTR_SOFFSET_G(ad)));
  F90_DPTR_LSTRIDE_P(dd, F90_DPTR_LSTRIDE_G(ad) * s);

#if defined(DEBUG)
  if (__fort_test & DEBUG_DIST)
    printf("%d set_section %d(%d:%d)->%d(%d:%d:%d) o(%d:%d) a(%d:%d)"
           " %dx+%d lbase=%d scoff=%d F90_DPTR_SOFFSET_G(ad)=%d"
           " F90_DPTR_SOFFSET_G(dd)=%d\n",
           GET_DIST_LCPU, ddim, F90_DPTR_LBOUND_G(dd), DPTR_UBOUND_G(dd), adim,
           l, u, s, DIST_DPTR_OLB_G(dd), DIST_DPTR_OUB_G(dd), DIST_DPTR_LAB_G(dd),
           DIST_DPTR_UAB_G(dd), F90_DPTR_LSTRIDE_G(dd), DIST_DPTR_LOFFSET_G(dd),
           F90_LBASE_G(d), DIST_SCOFF_G(d), F90_DPTR_SOFFSET_G(ad),
           F90_DPTR_SOFFSET_G(dd));
#endif
}

void I8(__fort_set_section)(F90_Desc *d, __INT_T ddim, F90_Desc *a,
                            __INT_T adim, __INT_T l, __INT_T u, __INT_T s)
{
  __DIST_SET_SECTIONX(d, ddim, a, adim, l, u, s, 0);
}

/* Compute the global section size.  Scalar subscript offset, local
   base, and local index offsets were adjusted in set_section.  Local
   size does not change. */

void I8(__fort_finish_section)(F90_Desc *d)
{
  __INT_T gsize, i;
  __INT_T rank = F90_RANK_G(d);

  if (DIST_NONSEQUENCE_G(d))
    F90_FLAGS_P(d, F90_FLAGS_G(d) & ~__SEQUENCE);

  gsize = 1;
  for (i = 0; i < rank; ++i) {
    gsize *= F90_DIM_EXTENT_G(d, i);
  }
  F90_GSIZE_P(d, gsize); /* global section size */
}

/* Create a new descriptor for a section of array a.  Variable length
   argument list gives lower, upper, and stride for each array
   dimension followed by a bitmask indicating vector dimensions (not
   scalar subscripts.) */

#define BOGUSFLAG 0x100

/* for F90 */
void ENTFTN(SECT, sect)(F90_Desc *d, F90_Desc *a,
                        ...) /* ... = {lower, upper, stride,}* flags */
{
  va_list va;
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, flags, rank;
  __INT_T lower[MAXDIMS], upper[MAXDIMS], stride[MAXDIMS];
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* get bounds, strides, and flags argument */

  va_start(va, a);
  wrk_rank = F90_RANK_G(a);
  for (ax = 0; ax < wrk_rank; ++ax) {
    lower[ax] = *va_arg(va, __INT_T *);
    upper[ax] = *va_arg(va, __INT_T *);
    stride[ax] = *va_arg(va, __INT_T *);
  }
  flags = *va_arg(va, __INT_T *);
  va_end(va);

/* determine section rank - popcnt of flags bits */

#if MAXDIMS != 7
  __fort_abort("SECT: need to recode for different MAXDIMS");
#endif
  rank = (flags & 0x55) + (flags >> 1 & 0x15);
  rank = (rank & 0x33) + (rank >> 2 & 0x13);
  rank += rank >> 4;
  rank &= 0x7;

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);
  if (F90_LEN_G(d) == GET_DIST_SIZE_OF(F90_KIND_G(d)))
    F90_FLAGS_P(d, F90_FLAGS_G(d) | __SEQUENTIAL_SECTION);

  /* bogus bounds: defer section setup until copy */

  gsize = 1;
  if (flags & BOGUSFLAG) {
    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    return;
  }

  /* normal section : set up each dimension and compute GSIZE*/

  wrk_rank = F90_RANK_G(a);
  for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
    if ((flags >> (ax - 1)) & 1) {
      dx++;
      __DIST_SET_SECTIONXX(d, dx, a, ax, lower[ax - 1], upper[ax - 1],
                          stride[ax - 1], (flags & __NOREINDEX), gsize);
    } else {
      I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
  }
  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
}

/* ASECTION invokes DIST_SET_SECTIONXX to
 * set bounds, strides; dx and gsize are updated directly */
#define ASECTION(d, dx, a, ax, lb, ub, st, gsize, flags)                       \
  if (flags & (1 << (ax - 1))) {                                               \
    dx++;                                                                      \
    __DIST_SET_SECTIONXX(d, dx, a, ax, lb, ub, st, (flags & __NOREINDEX),       \
                        gsize);                                                \
  } else                                                                       \
    I8(__fort_set_single)(d, a, ax, lb, __SCALAR);

/* TSECTION is used when the address to be used
 * is the address of the first element of the section */
#define TSECTION(d, dx, a, ax, lb, ub, st, gsize, flags)                       \
  if (flags & (1 << (ax - 1))) {                                               \
    DECL_DIM_PTRS(__dd);                                                       \
    DECL_DIM_PTRS(__ad);                                                       \
    __INT_T __extent, u, l, s;                                                 \
    dx++;                                                                      \
    SET_DIM_PTRS(__ad, a, ax - 1);                                             \
    SET_DIM_PTRS(__dd, d, dx - 1);                                             \
    l = lb;                                                                    \
    u = ub;                                                                    \
    s = st;                                                                    \
    __extent = u - l + s; /* section extent */                                 \
    if (s != 1) {                                                              \
      if (s == -1) {                                                           \
        __extent = -__extent;                                                  \
      } else {                                                                 \
        __extent /= s;                                                         \
      }                                                                        \
    }                                                                          \
    if (__extent < 0) {                                                        \
      __extent = 0;                                                            \
    }                                                                          \
    F90_DPTR_LBOUND_P(__dd, 1);                                                \
    DPTR_UBOUND_P(__dd, __extent);                                             \
    F90_DPTR_SSTRIDE_P(__dd, 1);                                               \
    F90_DPTR_SOFFSET_P(__dd, 0);                                               \
    F90_DPTR_LSTRIDE_P(__dd, F90_DPTR_LSTRIDE_G(__ad) * s);                    \
    F90_LBASE_P(d, F90_LBASE_G(d) - F90_DPTR_LSTRIDE_G(__dd));                 \
    if (F90_DPTR_LSTRIDE_G(__dd) != gsize)                                     \
      F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));                \
    gsize *= __extent;                                                         \
  }

/* for F90 */
void ENTF90(SECT, sect)(F90_Desc *d, F90_Desc *a, __INT_T *prank,
                        ...) /* ... = {lower, upper, stride,}* flags */
{
  va_list va;
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, flags, rank;
  __INT_T lower[MAXDIMS], upper[MAXDIMS], stride[MAXDIMS];
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* get bounds, strides, and flags argument */

  va_start(va, prank);
  wrk_rank = *prank;
  for (ax = 0; ax < wrk_rank; ++ax) {
    lower[ax] = *va_arg(va, __INT_T *);
    upper[ax] = *va_arg(va, __INT_T *);
    stride[ax] = *va_arg(va, __INT_T *);
  }
  flags = *va_arg(va, __INT_T *);
  va_end(va);

/* determine section rank - popcnt of flags bits */

#if MAXDIMS != 7
  __fort_abort("SECT: need to recode for different MAXDIMS");
#endif
  rank = (flags & 0x55) + (flags >> 1 & 0x15);
  rank = (rank & 0x33) + (rank >> 2 & 0x13);
  rank += rank >> 4;
  rank &= 0x7;

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);
  if (F90_LEN_G(d) == GET_DIST_SIZE_OF(F90_KIND_G(d)))
    F90_FLAGS_P(d, F90_FLAGS_G(d) | __SEQUENTIAL_SECTION);

  /* bogus bounds: defer section setup until copy */

  gsize = 1;
  if (flags & BOGUSFLAG) {
    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    F90_LSIZE_P(d, gsize); /* global section size */
    return;
  }

  /* normal section : set up each dimension and compute GSIZE*/

  wrk_rank = F90_RANK_G(a);
  if (flags & __SECTZBASE) {
    F90_LBASE_P(d, 1);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      TSECTION(d, dx, a, ax, lower[ax - 1], upper[ax - 1], stride[ax - 1],
               gsize, flags);
    }
  } else {
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      ASECTION(d, dx, a, ax, lower[ax - 1], upper[ax - 1], stride[ax - 1],
               gsize, flags);
    }
  }
  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
  F90_LSIZE_P(d, gsize); /* global section size */
}

/* for F90 */
void ENTF90(SECT1, sect1)(F90_Desc *d, F90_Desc *a, __INT_T *prank,
                          /* ... = {lower, upper, stride,}* flags */
                          __INT_T *lw0, __INT_T *up0, __INT_T *st0,
                          __INT_T *bfg)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, flags, rank;
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* get flags argument */

  flags = *bfg;

#if MAXDIMS != 7
  __fort_abort("SECT: need to recode for different MAXDIMS");
#endif
  /* determine section rank - popcnt of flags bits */
  /* rank is at most 1 */
  rank = (flags & 0x1);

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);

  /* bogus bounds: defer section setup until copy */

  gsize = 1;
  if (flags & BOGUSFLAG) {
    __INT_T lower[1], upper[1], stride[1];
    lower[0] = *lw0;
    upper[0] = *up0;
    stride[0] = *st0;

    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    F90_LSIZE_P(d, gsize); /* global section size */
    return;
  }

  dx = 0;
  if (flags & __SECTZBASE) {
    F90_LBASE_P(d, 1);
    TSECTION(d, dx, a, 1, *lw0, *up0, *st0, gsize, flags);
  } else {
    ASECTION(d, dx, a, 1, *lw0, *up0, *st0, gsize, flags);
  }

  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
  F90_LSIZE_P(d, gsize); /* global section size */
}

/* for F90 */
void ENTF90(SECT1v, sect1v)(F90_Desc *d, F90_Desc *a, __INT_T *prank,
                            /* ... = {lower, upper, stride,}* flags */
                            __INT_T lw0, __INT_T up0, __INT_T st0,
                            __INT_T flags)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, rank;
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* determine section rank - popcnt of flags bits */

  /* rank is at most 1 */
  rank = (flags & 0x1);

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);
  gsize = 1;

  /* bogus bounds: defer section setup until copy */

  if (flags & BOGUSFLAG) {
    __INT_T lower[1], upper[1], stride[1];
    lower[0] = lw0;
    upper[0] = up0;
    stride[0] = st0;

    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    F90_LSIZE_P(d, gsize); /* global section size */
    return;
  }

  /* normal section : set up each dimension and compute GSIZE*/

  dx = 0;
  if (flags & __SECTZBASE) {
    F90_LBASE_P(d, 1);
    TSECTION(d, dx, a, 1, lw0, up0, st0, gsize, flags);
  } else {
    ASECTION(d, dx, a, 1, lw0, up0, st0, gsize, flags);
  }

  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
  F90_LSIZE_P(d, gsize); /* global section size */
}

/* for F90 */
void ENTF90(SECT2, sect2)(F90_Desc *d, F90_Desc *a, __INT_T *prank,
                          /* ... = {lower, upper, stride,}* flags */
                          __INT_T *lw0, __INT_T *up0, __INT_T *st0,
                          __INT_T *lw1, __INT_T *up1, __INT_T *st1,
                          __INT_T *bfg)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, flags, rank;
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* get flags argument */

  flags = *bfg;

#if MAXDIMS != 7
  __fort_abort("SECT: need to recode for different MAXDIMS");
#endif
  /* determine section rank - popcnt of flags bits */
  /* rank is at most 2 */
  rank = (flags & 0x1) + (flags >> 1 & 0x1);

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);

  /* bogus bounds: defer section setup until copy */

  gsize = 1;
  if (flags & BOGUSFLAG) {
    __INT_T lower[2], upper[2], stride[2];
    lower[0] = *lw0;
    upper[0] = *up0;
    stride[0] = *st0;
    lower[1] = *lw1;
    upper[1] = *up1;
    stride[1] = *st1;

    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    F90_LSIZE_P(d, gsize); /* global section size */
    return;
  }

  dx = 0;
  if (flags & __SECTZBASE) {
    F90_LBASE_P(d, 1);
    TSECTION(d, dx, a, 1, *lw0, *up0, *st0, gsize, flags);
    TSECTION(d, dx, a, 2, *lw1, *up1, *st1, gsize, flags);
  } else {
    ASECTION(d, dx, a, 1, *lw0, *up0, *st0, gsize, flags);
    ASECTION(d, dx, a, 2, *lw1, *up1, *st1, gsize, flags);
  }

  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
  F90_LSIZE_P(d, gsize); /* global section size */
}

/* for F90 */
void ENTF90(SECT2v, sect2v)(F90_Desc *d, F90_Desc *a, __INT_T *prank,
                            /* ... = {lower, upper, stride,}* flags */
                            __INT_T lw0, __INT_T up0, __INT_T st0, __INT_T lw1,
                            __INT_T up1, __INT_T st1, __INT_T flags)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, rank;
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* determine section rank - popcnt of flags bits */

  /* rank is at most 2 */
  rank = (flags & 0x1) + (flags >> 1 & 0x1);

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);
  gsize = 1;

  /* bogus bounds: defer section setup until copy */

  if (flags & BOGUSFLAG) {
    __INT_T lower[2], upper[2], stride[2];
    lower[0] = lw0;
    upper[0] = up0;
    stride[0] = st0;
    lower[1] = lw1;
    upper[1] = up1;
    stride[1] = st1;

    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    F90_LSIZE_P(d, gsize); /* global section size */
    return;
  }

  /* normal section : set up each dimension and compute GSIZE*/

  dx = 0;
  if (flags & __SECTZBASE) {
    F90_LBASE_P(d, 1);
    TSECTION(d, dx, a, 1, lw0, up0, st0, gsize, flags);
    TSECTION(d, dx, a, 2, lw1, up1, st1, gsize, flags);
  } else {
    ASECTION(d, dx, a, 1, lw0, up0, st0, gsize, flags);
    ASECTION(d, dx, a, 2, lw1, up1, st1, gsize, flags);
  }

  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
  F90_LSIZE_P(d, gsize); /* global section size */
}

/* for F90 */
void ENTF90(SECT3, sect3)(F90_Desc *d, F90_Desc *a, __INT_T *prank,
                          /* ... = {lower, upper, stride,}* flags */
                          __INT_T *lw0, __INT_T *up0, __INT_T *st0,
                          __INT_T *lw1, __INT_T *up1, __INT_T *st1,
                          __INT_T *lw2, __INT_T *up2, __INT_T *st2,
                          __INT_T *bfg)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, flags, rank;
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* get flags argument */

  flags = *bfg;

#if MAXDIMS != 7
  __fort_abort("SECT: need to recode for different MAXDIMS");
#endif
  /* determine section rank - popcnt of flags bits */
  /* rank is at most 3 */
  rank = (flags & 0x5) + (flags >> 1 & 0x1);
  rank = (rank & 0x3) + (rank >> 2 & 0x1);

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);

  /* bogus bounds: defer section setup until copy */

  gsize = 1;
  if (flags & BOGUSFLAG) {
    __INT_T lower[MAXDIMS], upper[MAXDIMS], stride[MAXDIMS];
    lower[0] = *lw0;
    upper[0] = *up0;
    stride[0] = *st0;
    lower[1] = *lw1;
    upper[1] = *up1;
    stride[1] = *st1;
    lower[2] = *lw2;
    upper[2] = *up2;
    stride[2] = *st2;

    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    F90_LSIZE_P(d, gsize); /* global section size */
    return;
  }

  dx = 0;
  if (flags & __SECTZBASE) {
    F90_LBASE_P(d, 1);
    TSECTION(d, dx, a, 1, *lw0, *up0, *st0, gsize, flags);
    TSECTION(d, dx, a, 2, *lw1, *up1, *st1, gsize, flags);
    TSECTION(d, dx, a, 3, *lw2, *up2, *st2, gsize, flags);
  } else {
    ASECTION(d, dx, a, 1, *lw0, *up0, *st0, gsize, flags);
    ASECTION(d, dx, a, 2, *lw1, *up1, *st1, gsize, flags);
    ASECTION(d, dx, a, 3, *lw2, *up2, *st2, gsize, flags);
  }

  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
  F90_LSIZE_P(d, gsize); /* global section size */
}

/* for F90 */
void ENTF90(SECT3v, sect3v)(F90_Desc *d, F90_Desc *a, __INT_T *prank,
                            /* ... = {lower, upper, stride,}* flags */
                            __INT_T lw0, __INT_T up0, __INT_T st0, __INT_T lw1,
                            __INT_T up1, __INT_T st1, __INT_T lw2, __INT_T up2,
                            __INT_T st2, __INT_T flags)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, rank;
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* determine section rank - popcnt of flags bits */

  /* rank is at most 3 */
  rank = (flags & 0x5) + (flags >> 1 & 0x1);
  rank = (rank & 0x3) + (rank >> 2 & 0x1);

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);
  gsize = 1;

  /* bogus bounds: defer section setup until copy */

  if (flags & BOGUSFLAG) {
    __INT_T lower[3], upper[3], stride[3];
    lower[0] = lw0;
    upper[0] = up0;
    stride[0] = st0;
    lower[1] = lw1;
    upper[1] = up1;
    stride[1] = st1;
    lower[2] = lw2;
    upper[2] = up2;
    stride[2] = st2;

    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    F90_LSIZE_P(d, gsize); /* global section size */
    return;
  }

  /* normal section : set up each dimension and compute GSIZE*/

  dx = 0;
  if (flags & __SECTZBASE) {
    F90_LBASE_P(d, 1);
    TSECTION(d, dx, a, 1, lw0, up0, st0, gsize, flags);
    TSECTION(d, dx, a, 2, lw1, up1, st1, gsize, flags);
    TSECTION(d, dx, a, 3, lw2, up2, st2, gsize, flags);
  } else {
    ASECTION(d, dx, a, 1, lw0, up0, st0, gsize, flags);
    ASECTION(d, dx, a, 2, lw1, up1, st1, gsize, flags);
    ASECTION(d, dx, a, 3, lw2, up2, st2, gsize, flags);
  }

  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
  F90_LSIZE_P(d, gsize); /* global section size */
}

#undef ASECTION
#undef TSECTION

/* BSECTION updates dx, gsize directly */
#define BSECTION(d, dx, a, ax, lb, ub, st, gsize, flags)                       \
  if (flags & (1 << (ax - 1))) {                                               \
    dx++;                                                                      \
    __DIST_SET_SECTIONXX(d, dx, a, ax, lb, ub, st, (flags & __NOREINDEX),       \
                        gsize);                                                \
  } else                                                                       \
    I8(__fort_set_single)(d, a, ax, lb, __SCALAR);

/* for F90 */
void ENTFTN(SECT3, sect3)(F90_Desc *d, F90_Desc *a,
                          /* ... = {lower, upper, stride,}* flags */
                          __INT_T *lw0, __INT_T *up0, __INT_T *st0,
                          __INT_T *lw1, __INT_T *up1, __INT_T *st1,
                          __INT_T *lw2, __INT_T *up2, __INT_T *st2,
                          __INT_T *bfg)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, flags, rank;
  __INT_T gsize;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* get flags argument */

  flags = *bfg;

#if MAXDIMS != 7
  __fort_abort("SECT: need to recode for different MAXDIMS");
#endif
  /* determine section rank - popcnt of flags bits */
  /* rank is at most 3 */
  rank = (flags & 0x5) + (flags >> 1 & 0x1);
  rank = (rank & 0x3) + (rank >> 2 & 0x1);

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);

  /* bogus bounds: defer section setup until copy */

  gsize = 1;
  if (flags & BOGUSFLAG) {
    __INT_T lower[MAXDIMS], upper[MAXDIMS], stride[MAXDIMS];
    lower[0] = *lw0;
    upper[0] = *up0;
    stride[0] = *st0;
    lower[1] = *lw1;
    upper[1] = *up1;
    stride[1] = *st1;
    lower[2] = *lw2;
    upper[2] = *up2;
    stride[2] = *st2;

    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    return;
  }

  dx = 0;
  BSECTION(d, dx, a, 1, *lw0, *up0, *st0, gsize, flags);
  BSECTION(d, dx, a, 2, *lw1, *up1, *st1, gsize, flags);
  BSECTION(d, dx, a, 3, *lw2, *up2, *st2, gsize, flags);

  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
}

/* for F90 */
void ENTFTN(SECT3v, sect3v)(F90_Desc *d, F90_Desc *a,
                            /* ... = {lower, upper, stride,}* flags */
                            __INT_T lw0, __INT_T up0, __INT_T st0, __INT_T lw1,
                            __INT_T up1, __INT_T st1, __INT_T lw2, __INT_T up2,
                            __INT_T st2, __INT_T flags)
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(dd);
  __INT_T ax, dx, rank;
  __INT_T gsize = 0;
  __INT_T wrk_rank;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("SECT: missing section descriptor");
  if (a == NULL || F90_TAG_G(a) != __DESC)
    __fort_abort("SECT: invalid array descriptor");
#endif

  /* determine section rank - popcnt of flags bits */

  /* rank is at most 3 */
  rank = (flags & 0x5) + (flags >> 1 & 0x1);
  rank = (rank & 0x3) + (rank >> 2 & 0x1);

  /* initialize descriptor */

  SET_F90_DIST_DESC_PTR(d, rank);
  __DIST_INIT_SECTION(d, rank, a);

  /* bogus bounds: defer section setup until copy */

  if (flags & BOGUSFLAG) {
    __INT_T lower[3], upper[3], stride[3];
    lower[0] = lw0;
    upper[0] = up0;
    stride[0] = st0;
    lower[1] = lw1;
    upper[1] = up1;
    stride[1] = st1;
    lower[2] = lw2;
    upper[2] = up2;
    stride[2] = st2;

    F90_FLAGS_P(d, F90_FLAGS_G(d) | __BOGUSBOUNDS);
    wrk_rank = F90_RANK_G(a);
    for (dx = 0, ax = 1; ax <= wrk_rank; ++ax) {
      if ((flags >> (ax - 1)) & 1) {
        SET_DIM_PTRS(dd, d, dx);
        dx++;
        SET_DIM_PTRS(ad, a, ax - 1);
        F90_DPTR_LBOUND_P(dd, lower[ax - 1]);
        DPTR_UBOUND_P(dd, upper[ax - 1]);
        F90_DPTR_SSTRIDE_P(dd, stride[ax - 1]);
        if (F90_DPTR_SSTRIDE_G(dd) != 1 || F90_DPTR_LSTRIDE_G(dd) != gsize) {
          F90_FLAGS_P(d, (F90_FLAGS_G(d) & ~__SEQUENTIAL_SECTION));
        }
        gsize *= F90_DPTR_EXTENT_G(dd);
      } else
        I8(__fort_set_single)(d, a, ax, lower[ax - 1], __SCALAR);
    }
    F90_GSIZE_P(d, gsize); /* global section size */
    return;
  }

  /* normal section : set up each dimension and compute GSIZE*/

  dx = 0;
  BSECTION(d, dx, a, 1, lw0, up0, st0, gsize, flags);
  BSECTION(d, dx, a, 2, lw1, up1, st1, gsize, flags);
  BSECTION(d, dx, a, 3, lw2, up2, st2, gsize, flags);

  /* no longer need section stride/section offset */
  F90_GSIZE_P(d, gsize); /* global section size */
}
#undef BSECTION

/* Copy the contents of descriptor d0 into d.  Descriptor d is assumed
   to be large enough. */

void I8(__fort_copy_descriptor)(F90_Desc *d, F90_Desc *d0)
{
  if (F90_TAG_G(d0) == __DESC) {
    __fort_bcopy((char *)d, (char *)d0,
                 SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(d0)));
    SET_F90_DIST_DESC_PTR(d, F90_RANK_G(d));
  } else {
    F90_TAG_P(d, F90_TAG_G(d0));
  }
}

/* Create a copy of the align-target template in space reserved
   following the descriptor. */

F90_Desc *I8(__fort_inherit_template)(F90_Desc *d, __INT_T rank,
                                      F90_Desc *target)
{
  DECL_HDR_PTRS(t);
  __INT_T dz, dzr;

#if defined(DEBUG)
  if (rank < 0 || rank > MAXDIMS)
    __fort_abort("inherit_descriptor: invalid  rank");
  if (target == NULL || F90_TAG_G(target) != __DESC)
    __fort_abort("inherit_descriptor: invalid align-target descriptor");
#endif

  dz = SIZE_OF_RANK_n_ARRAY_DESC(rank);
  dzr = ALIGNR(dz);
  t = (F90_Desc *)((char *)d + dzr);

  I8(__fort_copy_descriptor)(t, target);

  F90_FLAGS_P(t, F90_FLAGS_G(t) | __TEMPLATE);
  F90_FLAGS_P(t, F90_FLAGS_G(t) & ~__NOT_COPIED);

  F90_LSIZE_P(t, 0);

  DIST_ALIGN_TARGET_P(t, t);
  DIST_NEXT_ALIGNEE_P(t, NULL);
  DIST_ACTUAL_ARG_P(t, NULL);

  return t;
}

/* Return the section extent. */

__INT_T
ENTFTN(EXTENT, extent)
(F90_Desc *d, __INT_T *gdim)
{
  __INT_T dim;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("EXTENT: invalid descriptor");
#endif

  if (F90_TAG_G(d) != __DESC)
    return 1; /* scalar or sequential */

  dim = *gdim;

#if defined(DEBUG)
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("EXTENT: invalid dimension");
#endif

  return F90_DIM_EXTENT_G(d, dim - 1);
}

/* this is just like the above, but with an extra argument
 * that is set for 'local' bounds, zero for 'global' bounds.
 */
__INT_T
ENTFTN(GLEXTENT, glextent)
(F90_Desc *d, __INT_T *gdim, __INT_T *glocal)
{
  DECL_DIM_PTRS(dd);
  __INT_T cl, cn, dim, extent, l, u, local;

#if defined(DEBUG)
  if (d == NULL)
    __fort_abort("GLEXTENT: invalid descriptor");
#endif

  if (F90_TAG_G(d) != __DESC)
    return 1; /* scalar or sequential */

  dim = *gdim;
  local = *glocal;

#if defined(DEBUG)
  if (dim < 1 || dim > F90_RANK_G(d))
    __fort_abort("GLEXTENT: invalid dimension");
#endif

  SET_DIM_PTRS(dd, d, dim - 1);

  if (local && ~F90_FLAGS_G(d) & __LOCAL) {

    /* coercing global to local: return local extent */

    if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
      return 0;

    I8(__fort_cycle_bounds)(d);

    extent = 0;
    for (cl = DIST_DPTR_CL_G(dd), cn = DIST_DPTR_CN_G(dd); --cn >= 0;
         cl += DIST_DPTR_CS_G(dd))
      extent += I8(__fort_block_bounds)(d, dim, cl, &l, &u);
  } else {

    /* normal case; return global extent */

    extent = F90_DPTR_EXTENT_G(dd);
  }

  return extent;
}

/* Return the lower bound for the specified dimension */

__INT_T
ENTFTN(LBOUND, lbound)(__INT_T *dim, F90_Desc *pd)
{
  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  if (!ISPRESENT(dim) || *dim < 1 || *dim > F90_RANK_G(pd))
    __fort_abort("LBOUND: invalid dim");
  return F90_DIM_LBOUND_G(pd, *dim - 1);
}

__INT8_T
ENTFTN(KLBOUND, klbound)(__INT_T *dim, F90_Desc *pd)
{
  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  if (!ISPRESENT(dim) || *dim < 1 || *dim > F90_RANK_G(pd))
    __fort_abort("LBOUND: invalid dim");
  return F90_DIM_LBOUND_G(pd, *dim - 1);
}

/* return the upper bound for the specified dimension */

__INT_T
ENTFTN(UBOUND, ubound)(__INT_T *dim, F90_Desc *pd)
{
  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  if (!ISPRESENT(dim) || *dim < 1 || *dim > F90_RANK_G(pd))
    __fort_abort("UBOUND: invalid dim");
  return DIM_UBOUND_G(pd, *dim - 1);
}

__INT8_T
ENTFTN(KUBOUND, kubound)(__INT_T *dim, F90_Desc *pd)
{
  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  if (!ISPRESENT(dim) || *dim < 1 || *dim > F90_RANK_G(pd))
    __fort_abort("UBOUND: invalid dim");
  return DIM_UBOUND_G(pd, *dim - 1);
}

/* Return lower bounds for all dimensions as a rank 1 array */

void ENTFTN(LBOUNDA, lbounda)(__INT_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDA1, lbounda1)(__INT1_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDA2, lbounda2)(__INT2_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDA4, lbounda4)(__INT4_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDA8, lbounda8)(__INT8_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDAZ, lboundaz)(__INT4_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDAZ1, lboundaz1)(__INT1_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDAZ2, lboundaz2)(__INT2_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDAZ4, lboundaz4)(__INT4_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(LBOUNDAZ8, lboundaz8)(__INT8_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(KLBOUNDA, klbounda)(__INT_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

void ENTFTN(KLBOUNDAZ, klboundaz)(__INT8_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("LBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_LBOUND_G(pd, dim);
}

/* Return upper bounds for all dimensions as a rank 1 array */

void ENTFTN(UBOUNDA, ubounda)(__INT_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDA1, ubounda1)(__INT1_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDA2, ubounda2)(__INT2_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDA4, ubounda4)(__INT4_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDA8, ubounda8)(__INT8_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDAZ, uboundaz)(__INT4_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDAZ1, uboundaz1)(__INT1_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDAZ2, uboundaz2)(__INT2_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDAZ4, uboundaz4)(__INT4_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_UBOUND_G(pd, dim);
}

void ENTFTN(UBOUNDAZ8, uboundaz8)(__INT8_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = F90_DIM_UBOUND_G(pd, dim);
}

void ENTFTN(KUBOUNDA, kubounda)(__INT_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = DIM_UBOUND_G(pd, dim);
}

void ENTFTN(KUBOUNDAZ, kuboundaz)(__INT8_T *arr, F90_Desc *pd)
{
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("UBOUND: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim)
    arr[dim] = DIM_UBOUND_G(pd, dim);
}

/* If dim is present, return the extent for the specified dimension.
   otherwise, return the size of the array (the product of all
   extents) */

__INT_T
ENTFTN(SIZE, size)(__INT_T *dim, F90_Desc *pd)
{
  __INT_T size = 0;

  if (F90_TAG_G(pd) != __DESC) {
    return 1;
  }
  if (!ISPRESENT(dim))
    size = F90_GSIZE_G(pd);
  else if (*dim < 1 || *dim > F90_RANK_G(pd))
    __fort_abort("SIZE: invalid dim");
  else {
    size = F90_DIM_EXTENT_G(pd, *dim - 1);
  }
  return size;
}

__INT8_T
ENTFTN(KSIZE, ksize)(__INT_T *dim, F90_Desc *pd)
{

  /*
   * -i8 variant of __size
   */

  __INT_T size = 0;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("SIZE: arg not associated with array");
  if (!ISPRESENT(dim))
    size = F90_GSIZE_G(pd);
  else if (*dim < 1 || *dim > F90_RANK_G(pd))
    __fort_abort("SIZE: invalid dim");
  else {
    size = F90_DIM_EXTENT_G(pd, *dim - 1);
  }
  return (__INT8_T)size;
}

/* Return the array shape as a rank 1 array */

void ENTFTN(SHAPE, shape)(__INT4_T *arr, F90_Desc *pd)
{
  DECL_DIM_PTRS(pdd);
  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("SHAPE: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim) {
    SET_DIM_PTRS(pdd, pd, dim);
    arr[dim] = F90_DIM_EXTENT_G(pd, dim);
  }
}

void ENTFTN(KSHAPE, kshape)(__INT8_T *arr, F90_Desc *pd)
{

  /*
   * -i8 variant of SHAPE
   */

  __INT_T dim, rank;

  if (F90_TAG_G(pd) != __DESC)
    __fort_abort("SHAPE: arg not associated with array");
  rank = F90_RANK_G(pd);
  for (dim = 0; dim < rank; ++dim) {
    arr[dim] = F90_DIM_EXTENT_G(pd, dim);
  }
}

void I8(__fort_reverse_array)(char *db, char *ab, F90_Desc *dd, F90_Desc *ad)
{

  /* make a "reversed" copy of descriptor ad and store it in dd,
   * then copy the data in ab and store it in reverse in db ...
   *
   * Internal procedure....we assume only the run-time calls this
   * routine... (The compiler could do a much better job at generating
   *             equivalent code for performing this function)
   *
   */

  __INT_T flags, i;
  __INT_T rank;
  __INT_T kind, len;
  __INT_T _0 = 0;
  __INT_T isstar;
  DECL_HDR_VARS(dd2);
  DECL_DIM_PTRS(add);
  __INT_T lbound[MAXDIMS], ubound[MAXDIMS], stride[MAXDIMS], dstfmt[MAXDIMS];
  __INT_T paxis[MAXDIMS], no[MAXDIMS], po[MAXDIMS];
  __INT_T *gen_block[MAXDIMS];
  sked *s;
  void *xfer;

  rank = F90_RANK_G(ad);

  SET_F90_DIST_DESC_PTR(dd2, rank);

  flags = (__PRESCRIPTIVE_DIST_TARGET + __PRESCRIPTIVE_DIST_FORMAT +
           __DIST_TARGET_AXIS + __ASSUMED_GB_EXTENT + __DUMMY_COLLAPSE_PAXIS);

  isstar = 0;
  for (i = 0; i < rank; ++i) {
    SET_DIM_PTRS(add, ad, i);
    stride[i] = -DIST_DPTR_TSTRIDE_G(add);
    lbound[i] = DIST_DPTR_TLB_G(add);
    ubound[i] = DIST_DPTR_TUB_G(add);
    paxis[i] = DIST_DPTR_PAXIS_G(add);
    gen_block[i] = DIST_DPTR_GEN_BLOCK_G(add);
    no[i] = DIST_DPTR_NO_G(add);
    po[i] = DIST_DPTR_PO_G(add);

    switch (DFMT(ad, i + 1)) {

    case DFMT_GEN_BLOCK:
      isstar |= 0x01 << (7 + 3 * i);
      dstfmt[i] = 0;
      break;
    case DFMT_COLLAPSED:
      isstar |= 0x01 << i;
      dstfmt[i] = 0;
      break;
    case DFMT_BLOCK:
      dstfmt[i] = 0;
      break;
    case DFMT_BLOCK_K:
      dstfmt[i] = DIST_DPTR_BLOCK_G(add);
      break;
    case DFMT_CYCLIC:
      dstfmt[i] = -1;
      break;
    case DFMT_CYCLIC_K:
      dstfmt[i] = -(DIST_DPTR_BLOCK_G(add));
      break;

    default:
      __fort_abort("__fort_reverse: invalid dist format (internal)");
    }
  }

  switch (rank) {

  case 1:

    ENTFTN(TEMPLATE, template)
    (dd2, &rank, &flags, DIST_DIST_TARGET_G(ad), &isstar, &paxis[0],
     (!gen_block[0]) ? &dstfmt[0] : gen_block[0], &lbound[0], &ubound[0]);
    break;

  case 2:

    ENTFTN(TEMPLATE, template)
    (dd2, &rank, &flags, DIST_DIST_TARGET_G(ad), &isstar, &paxis[0],
     (!gen_block[0]) ? &dstfmt[0] : gen_block[0], &lbound[0], &ubound[0],
     &paxis[1], (!gen_block[1]) ? &dstfmt[1] : gen_block[1], &lbound[1],
     &ubound[1]);
    break;

  case 3:

    ENTFTN(TEMPLATE, template)
    (dd2, &rank, &flags, DIST_DIST_TARGET_G(ad), &isstar, &paxis[0],
     (!gen_block[0]) ? &dstfmt[0] : gen_block[0], &lbound[0], &ubound[0],
     &paxis[1], (!gen_block[1]) ? &dstfmt[1] : gen_block[1], &lbound[1],
     &ubound[1], &paxis[2], (!gen_block[2]) ? &dstfmt[2] : gen_block[2],
     &lbound[2], &ubound[2]);
    break;

  case 4:

    ENTFTN(TEMPLATE, template)
    (dd2, &rank, &flags, DIST_DIST_TARGET_G(ad), &isstar, &paxis[0],
     (!gen_block[0]) ? &dstfmt[0] : gen_block[0], &lbound[0], &ubound[0],
     &paxis[1], (!gen_block[1]) ? &dstfmt[1] : gen_block[1], &lbound[1],
     &ubound[1], &paxis[2], (!gen_block[2]) ? &dstfmt[2] : gen_block[2],
     &lbound[2], &ubound[2], &paxis[3],
     (!gen_block[3]) ? &dstfmt[3] : gen_block[3], &lbound[3], &ubound[3]);

    break;

  case 5:

    ENTFTN(TEMPLATE, template)
    (dd2, &rank, &flags, DIST_DIST_TARGET_G(ad), &isstar, &paxis[0],
     (!gen_block[0]) ? &dstfmt[0] : gen_block[0], &lbound[0], &ubound[0],
     &paxis[1], (!gen_block[1]) ? &dstfmt[1] : gen_block[1], &lbound[1],
     &ubound[1], &paxis[2], (!gen_block[2]) ? &dstfmt[2] : gen_block[2],
     &lbound[2], &ubound[2], &paxis[3],
     (!gen_block[3]) ? &dstfmt[3] : gen_block[3], &lbound[3], &ubound[3],
     &paxis[4], (!gen_block[4]) ? &dstfmt[4] : gen_block[4], &lbound[4],
     &ubound[4]);
    break;

  case 6:

    ENTFTN(TEMPLATE, template)
    (dd2, &rank, &flags, DIST_DIST_TARGET_G(ad), &isstar, &paxis[0],
     (!gen_block[0]) ? &dstfmt[0] : gen_block[0], &lbound[0], &ubound[0],
     &paxis[1], (!gen_block[1]) ? &dstfmt[1] : gen_block[1], &lbound[1],
     &ubound[1], &paxis[2], (!gen_block[2]) ? &dstfmt[2] : gen_block[2],
     &lbound[2], &ubound[2], &paxis[3],
     (!gen_block[3]) ? &dstfmt[3] : gen_block[3], &lbound[3], &ubound[3],
     &paxis[4], (!gen_block[4]) ? &dstfmt[4] : gen_block[4], &lbound[4],
     &ubound[4], &paxis[5], (!gen_block[5]) ? &dstfmt[5] : gen_block[5],
     &lbound[5], &ubound[5]);
    break;

  case 7:

    ENTFTN(TEMPLATE, template)
    (dd2, &rank, &flags, DIST_DIST_TARGET_G(ad), &isstar, &paxis[0],
     (!gen_block[0]) ? &dstfmt[0] : gen_block[0], &lbound[0], &ubound[0],
     &paxis[1], (!gen_block[1]) ? &dstfmt[1] : gen_block[1], &lbound[1],
     &ubound[1], &paxis[2], (!gen_block[2]) ? &dstfmt[2] : gen_block[2],
     &lbound[2], &ubound[2], &paxis[3],
     (!gen_block[3]) ? &dstfmt[3] : gen_block[3], &lbound[3], &ubound[3],
     &paxis[4], (!gen_block[4]) ? &dstfmt[4] : gen_block[4], &lbound[4],
     &ubound[4], &paxis[5], (!gen_block[5]) ? &dstfmt[5] : gen_block[5],
     &lbound[5], &ubound[5], &paxis[6],
     (!gen_block[6]) ? &dstfmt[6] : gen_block[6], &lbound[6], &ubound[6]);

    break;

  default:

    __fort_abort("reverse_array: Temp Invalid Rank (internal error)");
  }

  kind = F90_KIND_G(ad);
  len = F90_LEN_G(ad);

  switch (rank) {

  case 1:

    ENTFTN(INSTANCE, instance)
    (dd2, dd2, &kind, &len, &_0, &no[0], &po[0]);
    break;
  case 2:

    ENTFTN(INSTANCE, instance)
    (dd2, dd2, &kind, &len, &_0, &no[0], &po[0], &no[1], &po[1]);
    break;
  case 3:

    ENTFTN(INSTANCE, instance)
    (dd2, dd2, &kind, &len, &_0, &no[0], &po[0], &no[1], &po[1], &no[2],
     &po[2]);
    break;
  case 4:

    ENTFTN(INSTANCE, instance)
    (dd2, dd2, &kind, &len, &_0, &no[0], &po[0], &no[1], &po[1], &no[2], &po[2],
     &no[3], &po[3]);
    break;
  case 5:

    ENTFTN(INSTANCE, instance)
    (dd2, dd2, &kind, &len, &_0, &no[0], &po[0], &no[1], &po[1], &no[2], &po[2],
     &no[3], &po[3], &no[4], &po[4]);
    break;
  case 6:

    ENTFTN(INSTANCE, instance)
    (dd2, dd2, &kind, &len, &_0, &no[0], &po[0], &no[1], &po[1], &no[2], &po[2],
     &no[3], &po[3], &no[4], &po[4], &no[5], &po[5]);
    break;
  case 7:

    ENTFTN(INSTANCE, instance)
    (dd2, dd2, &kind, &len, &_0, &no[0], &po[0], &no[1], &po[1], &no[2], &po[2],
     &no[3], &po[3], &no[4], &po[4], &no[5], &po[5], &no[6], &po[6]);
    break;

  default:
    __fort_abort("reverse_array: Instance Invalid Rank (internal error)");
  }

  /* swap bounds for negative stride */

  for (i = 0; i < rank; ++i) {
    if (stride[i] < 0) {
      __INT_T t;

      t = ubound[i];
      ubound[i] = lbound[i];
      lbound[i] = t;
    }
  }

  switch (rank) {

  case 1:

    ENTFTN(SECT, sect)(dd, dd2, &lbound[0], &ubound[0], &stride[0], &rank);
    break;

  case 2:

    ENTFTN(SECT,sect) (dd,dd2,&lbound[0],&ubound[0],&stride[0],
                               &lbound[1],&ubound[1],&stride[1],&rank);
    break;

  case 3:

    ENTFTN(SECT,sect) (dd,dd2,&lbound[0],&ubound[0],&stride[0],
                               &lbound[1],&ubound[1],&stride[1],
                               &lbound[2],&ubound[2],&stride[2],&rank);
    break;

  case 4:

    ENTFTN(SECT,sect) (dd,dd2,&lbound[0],&ubound[0],&stride[0],
                               &lbound[1],&ubound[1],&stride[1],
                               &lbound[2],&ubound[2],&stride[2],
                               &lbound[3],&ubound[3],&stride[3],&rank);
    break;

  case 5:
    ENTFTN(SECT,sect) (dd,dd2,&lbound[0],&lbound[0],&stride[0],
                               &lbound[1],&ubound[1],&stride[1],
                               &lbound[2],&ubound[2],&stride[2],
                               &lbound[3],&ubound[3],&stride[3],
                               &lbound[4],&ubound[4],&stride[4],&rank);
    break;

  case 6:

    ENTFTN(SECT,sect) (dd,dd2,&ubound[0],&lbound[0],&stride[0],
                               &lbound[1],&ubound[1],&stride[1],
                               &lbound[2],&ubound[2],&stride[2],
                               &lbound[3],&ubound[3],&stride[3],
                               &lbound[4],&ubound[4],&stride[4],
                               &lbound[5],&ubound[5],&stride[5],&rank);
    break;

  case 7:

    ENTFTN(SECT,sect) (dd,dd2,&ubound[0],&lbound[0],&stride[0],
                               &lbound[1],&ubound[1],&stride[1],
                               &lbound[2],&ubound[2],&stride[2],
                               &lbound[3],&ubound[3],&stride[3],
                               &lbound[4],&ubound[4],&stride[4],
                               &lbound[5],&ubound[5],&stride[5],
                               &lbound[6],&ubound[6],&stride[6],&rank);
    break;

  default:

    __fort_abort("reverse_array: Sect Invalid rank (internal error)");
  }

  /* copy ab to db */

  s = (sked *)ENTFTN(COMM_COPY, comm_copy)(db, ab, dd, ad);

  xfer = (void *)ENTFTN(COMM_START, comm_start)(&s, db, dd, (char *)ab, ad);

  ENTFTN(COMM_FINISH, comm_finish)(xfer);
}
