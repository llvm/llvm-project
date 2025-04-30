/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "red.h"

static __INT_T _1 = 1;

/* translate a linearized index within a processor subgrid to the
   corresponding processor number (the linearized index within the
   whole grid).  If the index is invalid, then -1 is returned.  */

static __INT_T I8(map_to_processor)(int c,       /* linearized subgrid index */
                                    __INT_T p0,  /* base processor number */
                                    __INT_T pr,  /* subgrid rank */
                                    __INT_T *pe, /* grid extents */
                                    __INT_T *ps) /* grid strides */
{
  __INT_T i, k;

  for (i = 0; c > 0 && i < pr; ++i) {
    if (pe[i] > 1) {
      k = c;
      c /= pe[i];
      k -= c * pe[i];
      p0 += k * ps[i];
    }
  }
  if (c != 0)
    p0 = -1;
  return p0;
}

/* broadcast within processor subgrid */

static void
    I8(broadcast)(__INT_T from, /* source cpu's linearized subgrid index */
                  __INT_T me,   /* this cpu's linearized subgrid index */
                  __INT_T np,   /* number of processors in subgrid */
                  __INT_T p0,   /* base processor number */
                  __INT_T pr,   /* subgrid rank */
                  __INT_T *pe,  /* grid extents */
                  __INT_T *ps,  /* grid strides */
                  void *adr,    /* address of first data item */
                  __INT_T cnt,  /* number of items */
                  __INT_T str,  /* item stride */
                  dtype typ,    /* item type */
                  __INT_T len)  /* item length */
{
  __INT_T bit, cpu, dst, src;

  /* translate grid indexes so the source processor is 0 */

  me -= from;
  if (me < 0)
    me += np;

  if (me != 0) {
    bit = me & -me; /* least significant bit of (me) */
    src = (me ^ bit) + from;
    if (src >= np)
      src -= np;
    cpu = I8(map_to_processor)(src, p0, pr, pe, ps);
    __fort_rrecvl(cpu, adr, cnt, str, typ, len);
  } else {
    bit = 1;
    while (bit < np)
      bit <<= 1;
  }
  bit >>= 1;

  while (bit > 0) {
    dst = me ^ bit;
    if (dst < np) {
      dst += from;
      if (dst >= np)
        dst -= np;
      cpu = I8(map_to_processor)(dst, p0, pr, pe, ps);
      __fort_rsendl(cpu, adr, cnt, str, typ, len);
    }
    bit >>= 1;
  }
}

/* global, parallel, in-place reduction (accumulation?) of a scalar or
   array across selected mapped dimensions of a home array or
   template.  */

static void
global_reduce_abort(const char *what, const char *msg)
{
  char str[120];
  sprintf(str, "GLOBAL_%s: %s", what, msg);
  __fort_abort(str);
}

void I8(__fort_global_reduce)(char *rb, char *hb, int dims, F90_Desc *rd,
                              F90_Desc *hd, const char *what,
                              global_reduc_fn fn[__NTYPES])
{
  DECL_HDR_PTRS(ht); /* align-target */
  DECL_DIM_PTRS(hdd);
  DECL_DIM_PTRS(htd);
  char *tmp;     /* temp buffer pointer */
  double buf[8]; /* small buffer */
  dtype kind;
  __INT_T bufsiz, it, bit;
  __INT_T cnt, cpu, i, idx, len, mask, me, np, offgrid, pc, pl, pr;
  __INT_T pe[MAXDIMS], ps[MAXDIMS];

#if defined(DEBUG)
  if (hd == NULL || F90_TAG_G(hd) != __DESC)
    global_reduce_abort(what, "invalid home descriptor");
#endif

  if (LOCAL_MODE)
    return;

  if (F90_TAG_G(rd) == __DESC) {
    kind = F90_KIND_G(rd);
    len = F90_LEN_G(rd);
    cnt = F90_LSIZE_G(rd);
  } else if (ISSCALAR(rd) || ISSEQUENCE(rd)) {
    kind = TYPEKIND(rd);
    len = GET_DIST_SIZE_OF(kind);
    cnt = 1;
  } else
    global_reduce_abort(what, "invalid result descriptor");

  mask = DIST_MAPPED_G(hd) & dims;

  if (mask != 0) {

    /* all processors allocate same size temporary buffer */

    bufsiz = ALIGNR(cnt * len); /* round up buffer size */
    if (bufsiz > sizeof(buf))
      tmp = (char *)__fort_gmalloc(bufsiz);
    else
      tmp = (char *)buf;

    /* set up home template and processor grid */

    np = 1;
    pl = GET_DIST_LCPU;
    pr = me = 0;
    offgrid = 0;
    for (i = 1; i <= F90_RANK_G(hd); ++i) {
      if ((mask >> (i - 1)) & 1) {
        SET_DIM_PTRS(hdd, hd, i - 1);

        /* only processors in the grid participate in
           reduction */

        offgrid |= (DIST_DPTR_PCOORD_G(hdd) < 0);
        if (!offgrid) {
          pl -= DIST_DPTR_PCOORD_G(hdd) * DIST_DPTR_PSTRIDE_G(hdd);
          me += DIST_DPTR_PCOORD_G(hdd) * np;
          pe[pr] = DIST_DPTR_PSHAPE_G(hdd);
          ps[pr] = DIST_DPTR_PSTRIDE_G(hdd);
          pr++;
        }
        np *= DIST_DPTR_PSHAPE_G(hdd);
      }
    }

    /* probably need to exclude processors because of scalar
       subscripts or single alignments ? */

    if (!offgrid) {

#if defined(DEBUG)
      if (__fort_test & DEBUG_REDU) {
        printf("%d REDUCE_GLOBAL np=%d me=%d pr=%d pl=%d pe=", GET_DIST_LCPU,
               np, me, pr, pl);
        I8(__fort_show_index)(pr, pe);
        printf(" ps=");
        I8(__fort_show_index)(pr, ps);
        printf("\n");
      }
#endif

      /* this pattern guarantees identical results on all
         processors.  collect and reduce partial results from
         tree below */

      bit = 1;
      while (bit < np) {
        it = me ^ bit;
        bit <<= 1;
        if (it < me)
          break;
        if (it < np) {
          cpu = I8(map_to_processor)(it, pl, pr, pe, ps);
          __fort_rrecvl(cpu, tmp, cnt, 1, kind, len);
          fn[kind](cnt, rb, tmp, /* optional args */ NULL, NULL, 0);
        }
      }
      bit >>= 1;

      /* send partial results up the tree, receive final results */

      it = me ^ bit;
      if (it < me) {
        cpu = I8(map_to_processor)(it, pl, pr, pe, ps);
        __fort_rsendl(cpu, rb, cnt, 1, kind, len);
        __fort_rrecvl(cpu, rb, cnt, 1, kind, len);
        bit >>= 1;
      }

      /* send final result to tree below */

      while (bit > 0) {
        it = me ^ bit;
        bit >>= 1;
        if (it > me && it < np) {
          cpu = I8(map_to_processor)(it, pl, pr, pe, ps);
          __fort_rsendl(cpu, rb, cnt, 1, kind, len);
        }
      }
    }

    /* free temporary buffer */

    if (tmp != (char *)buf)
      __fort_gfree(tmp);
  }

  /* replicate the result across the mapped dimensions of the
     align-target which had scalar subscripts or single
     alignments in the home array.  result replication has
     already occurred in the unmapped dimensions, so a similar
     broadcast pattern is done within each replication group. */

  ht = (SUBGROUP_MODE) ? hd : DIST_ALIGN_TARGET_G(hd);
  mask = DIST_SINGLE_G(hd) & DIST_MAPPED_G(ht);
  if (mask != 0) {

    np = 1;
    cpu = pl = GET_DIST_LCPU;
    idx = me = 0;
    offgrid = 0;
    for (pr = i = 0; i < F90_RANK_G(ht); ++i) {
      if (mask >> i & 1) {
        SET_DIM_PTRS(htd, ht, i);
        offgrid |= (DIST_DPTR_PCOORD_G(htd) < 0);
        if (!offgrid) {
          pc = DIST_INFO_G(hd, i) - DIST_DPTR_TLB_G(htd);
          if (DFMT(ht, i + 1) == DFMT_GEN_BLOCK) {

            /* 
             * need to find the processor that owns pc
             * based on gen_block array
             */

            __INT_T *gb;
            __INT_T idxUB;
            __INT_T pshape;
            __INT_T j;
            pshape = DIST_DPTR_PSHAPE_G(htd);
            gb = DIST_DPTR_GEN_BLOCK_G(htd);
            ++pc; /* in terms of fortran indices (1 ... upperbound) */

#if defined(DEBUG)
            if (__fort_test & DEBUG_REDU)
              printf("owner of single alignment %d (gen_block dim)", pc);
#endif
            idxUB = *gb;
            for (j = 0; j < pshape; ++j) {
              if (idxUB >= pc)
                break;
              ++gb;
              idxUB += *gb;
            }

            if (j < pshape)
              pc = j;
            else
              __fort_abort("global_reduce: bad gen_block (internal error)");
#if defined(DEBUG)
            if (__fort_test & DEBUG_REDU)
              printf(" is processor %d\n", pc);
#endif

          } else {
            if (DIST_DPTR_BLOCK_G(htd) != 1)
              RECIP_DIV(&pc, pc, DIST_DPTR_BLOCK_G(htd));
            if (pc >= DIST_DPTR_PSHAPE_G(htd)) /* cyclic */
              RECIP_MOD(&pc, pc, DIST_DPTR_PSHAPE_G(htd));
          }

          cpu -= (DIST_DPTR_PCOORD_G(htd) - pc) * DIST_DPTR_PSTRIDE_G(htd);
          idx += pc * np;
          pl -= DIST_DPTR_PCOORD_G(htd) * DIST_DPTR_PSTRIDE_G(htd);
          me += DIST_DPTR_PCOORD_G(htd) * np;
          pe[pr] = DIST_DPTR_PSHAPE_G(htd);
          ps[pr] = DIST_DPTR_PSTRIDE_G(htd);
          ++pr;
        }
        np *= DIST_DPTR_PSHAPE_G(htd);
      }
    }
    if (!offgrid) {
#if defined(DEBUG)
      if (__fort_test & DEBUG_REDU) {
        printf("%d repl cpu=%d idx=%d me=%d np=%d pr=%d pl=%d pe=",
               GET_DIST_LCPU, cpu, idx, me, np, pr, pl);
        I8(__fort_show_index)(pr, pe);
        printf(" ps=");
        I8(__fort_show_index)(pr, ps);
        printf("\n");
      }
#endif
      I8(broadcast)(idx, me, np, pl, pr, pe, ps, rb, cnt, 1, kind, len);
    }
  }

/* also replicate results to all processors outside the processor
   grid.  */

  pl = 0;
  np = GET_DIST_TCPUS - pl;
  me = GET_DIST_LCPU - pl;

  if (np > 1 && me >= 0)
    I8(broadcast)(0, me, np, pl, 1, &np, &_1, rb, cnt, 1, kind, len);
}

/* global, parallel reduction across the mapped dimensions of the
   source section.  if dim <= 0, reduce in all dimensions, otherwise
   reduce in dimension dim only.  arguments vec1 and optional vec2 are
   local intermediate results that are combined in-place with results
   from all other processors to produce the final result. */

void I8(__fort_reduce_section)(void *vec1, dtype typ1, int len1, void *vec2,
                               dtype typ2, int len2, int cnt,
                               global_reduc_fn fn_g, int dim, F90_Desc *a)
{
  DECL_DIM_PTRS(ad);
  char *tmp1, *tmp2; /* temporary buffer pointers */
  double buf1[4], buf2[4];
  __INT_T bit, cpu, i, it, mask, n, offgrid, pr, np, pc, pl, pe[MAXDIMS],
      ps[MAXDIMS];

  mask = DIST_MAPPED_G(a);
  if (dim > 0) {
    mask &= (1 << (dim - 1)); /* reduce only in specified dim */
  }

  if (LOCAL_MODE || mask == 0)
    return;

  np = 1;
  pl = GET_DIST_LCPU;
  pr = pc = 0;
  offgrid = 0;
  for (i = 0; i < F90_RANK_G(a); ++i) {
    if (mask >> i & 1) {
      SET_DIM_PTRS(ad, a, i);
      offgrid |= (DIST_DPTR_PCOORD_G(ad) < 0);
      if (!offgrid) {
        pl -= DIST_DPTR_PCOORD_G(ad) * DIST_DPTR_PSTRIDE_G(ad);
        pc += DIST_DPTR_PCOORD_G(ad) * np;
        pe[pr] = DIST_DPTR_PSHAPE_G(ad);
        ps[pr] = DIST_DPTR_PSTRIDE_G(ad);
        pr++;
      }
      np *= DIST_DPTR_PSHAPE_G(ad);
    }
  }
  if (np == 1)
    return;

  /* allocate temporary buffers */

  n = cnt * len1;
  if (n > sizeof(buf1))
    tmp1 = (char *)__fort_gmalloc(n);
  else
    tmp1 = (char *)buf1;

  if (vec2 != NULL) {
    n = cnt * len2;
    if (n > sizeof(buf2))
      tmp2 = (char *)__fort_gmalloc(n);
    else
      tmp2 = (char *)buf2;
  } else
    tmp2 = NULL;

  /* only processors in the grid participate */

  if (!offgrid) {

#if defined(DEBUG)
    if (__fort_test & DEBUG_REDU) {
      printf("%d reduce np=%d pc=%d pr=%d pl=%d pe=", GET_DIST_LCPU, np, pc,
             pr, pl);
      I8(__fort_show_index)(pr, pe);
      printf(" ps=");
      I8(__fort_show_index)(pr, ps);
      printf("\n");
    }
#endif

    /* this pattern guarantees identical results on all processors.
       collect and reduce partial results from tree below */

    bit = 1;
    while (bit < np) {
      it = pc ^ bit;
      bit <<= 1;
      if (it < pc)
        break;
      if (it < np) {
        cpu = I8(map_to_processor)(it, pl, pr, pe, ps);
        __fort_rrecvl(cpu, tmp1, cnt, 1, typ1, len1);
        if (tmp2 != NULL)
          __fort_rrecvl(cpu, tmp2, cnt, 1, typ2, len2);
        fn_g(cnt, vec1, tmp1, vec2, tmp2, /* optional arg */ 0);
      }
    }
    bit >>= 1;

    /* send partial result up the tree and receive final result */

    it = pc ^ bit;
    if (it < pc) {
      bit >>= 1;
      cpu = I8(map_to_processor)(it, pl, pr, pe, ps);
      __fort_rsendl(cpu, vec1, cnt, 1, typ1, len1);
      if (vec2 != NULL)
        __fort_rsendl(cpu, vec2, cnt, 1, typ2, len2);
      __fort_rrecvl(cpu, vec1, cnt, 1, typ1, len1);
      if (vec2 != NULL)
        __fort_rrecvl(cpu, vec2, cnt, 1, typ2, len2);
    }

    /* send final result to tree below */

    while (bit > 0) {
      it = pc ^ bit;
      bit >>= 1;
      if (it > pc && it < np) {
        cpu = I8(map_to_processor)(it, pl, pr, pe, ps);
        __fort_rsendl(cpu, vec1, cnt, 1, typ1, len1);
        if (vec2 != NULL)
          __fort_rsendl(cpu, vec2, cnt, 1, typ2, len2);
      }
    }
  }

  /* free temporary buffers */

  if (tmp1 != (char *)buf1)
    __fort_gfree(tmp1);
  if (tmp2 != (char *)buf2)
    __fort_gfree(tmp2);
}

/* replicate result in distributed scalar dimensions of the source
   array section.  ALSO NEED TO REPLICATE SCALAR RESULT TO PROCESSORS
   OUTSIDE THE PROCESSOR ARRANGEMENT. */

void I8(__fort_replicate_result)(void *vec1, dtype typ1, int len1, void *vec2,
                                dtype typ2, int len2, int cnt, F90_Desc *a)
{
  DECL_HDR_PTRS(t);
  DECL_DIM_PTRS(td);
  __INT_T fromcpu, fromidx, i, mask, me, np, offgrid, p0, pc, pr, pe[MAXDIMS],
      ps[MAXDIMS];

  if (LOCAL_MODE)
    return;

  /* replicate the result across the partitioned dimensions of the
     align-target which had scalar subscripts or single alignments
     in the alignee.  result replication has already occurred in the
     non-partitioned dimensions, so a similar broadcast pattern is
     done within each replication group.  */

  t = DIST_ALIGN_TARGET_G(a);
  mask = DIST_SINGLE_G(a) & DIST_MAPPED_G(t);
  if (mask != 0) {

    np = 1;
    fromcpu = p0 = GET_DIST_LCPU;
    fromidx = me = 0;
    offgrid = 0;
    for (pr = i = 0; i < F90_RANK_G(t); ++i) {
      if (mask >> i & 1) {
        SET_DIM_PTRS(td, t, i);
        offgrid |= (DIST_DPTR_PCOORD_G(td) < 0);
        if (!offgrid) {
          pc = DIST_INFO_G(a, i) - DIST_DPTR_TLB_G(td);
          if (DIST_DPTR_BLOCK_G(td) != 1)
            RECIP_DIV(&pc, pc, DIST_DPTR_BLOCK_G(td));
          if (pc >= DIST_DPTR_PSHAPE_G(td)) /* cyclic */
            RECIP_MOD(&pc, pc, DIST_DPTR_PSHAPE_G(td));
          fromcpu -= (DIST_DPTR_PCOORD_G(td) - pc) * DIST_DPTR_PSTRIDE_G(td);
          fromidx += pc * np;
          p0 -= DIST_DPTR_PCOORD_G(td) * DIST_DPTR_PSTRIDE_G(td);
          me += DIST_DPTR_PCOORD_G(td) * np;
          pe[pr] = DIST_DPTR_PSHAPE_G(td);
          ps[pr] = DIST_DPTR_PSTRIDE_G(td);
          ++pr;
        }
        np *= DIST_DPTR_PSHAPE_G(td);
      }
    }
    if (!offgrid) {
#if defined(DEBUG)
      if (__fort_test & DEBUG_REDU) {
        printf("%d repl fromcpu=%d fromidx=%d me=%d np=%d"
               " pr=%d p0=%d pe=",
               GET_DIST_LCPU, fromcpu, fromidx, me, np, pr, p0);
        I8(__fort_show_index)(pr, pe);
        printf(" ps=");
        I8(__fort_show_index)(pr, ps);
        printf("\n");
      }
#endif
      I8(broadcast)(fromidx, me, np, p0, pr, pe, ps, vec1, cnt, 1, typ1, len1);

      if (vec2 != NULL)
        I8(broadcast)(fromidx, me, np, p0, pr, pe, ps, vec2, cnt, 1,
			      typ2, len2);
    }
  }

/* also replicate a scalar result to all processors outside the
   processor grid.  */

}
