/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file 
 * \brief Dynamic, realign, redistribute arrays
 */

#include "stdioInterf.h"
#include "fioMacros.h"

#include "fort_vars.h"

/* reallocate and copy.  ad = new descriptor, dd = old descriptor, pd
   = descriptor associated with pointer and offset variables, i.e. the
   original descriptor location. */
static void
I8(recopy)(F90_Desc *ad, F90_Desc *dd, F90_Desc *pd)
{
  char *ab, *af, *db, *df, *base, **ptr;
  __POINT_T *off;
  chdr *ch;

  if (F90_FLAGS_G(ad) & __TEMPLATE)
    return;

  ptr = ((char **)pd) - 2;      /* array pointer variable */
  off = (__POINT_T *)(ptr + 1); /* array offset variable */

  db = *ptr; /* array address */

  if (!ISPRESENT(db))
    return;

  /* allocate the new array */

  base = db - (*off - 1) * F90_LEN_G(ad);
  ab = I8(__fort_allocate)(F90_LSIZE_G(ad), F90_KIND_G(ad), F90_LEN_G(ad), base,
                          ptr, off);

  /* copy the old into the new */

  af = ab + DIST_SCOFF_G(ad) * F90_LEN_G(ad);
  df = db + DIST_SCOFF_G(dd) * F90_LEN_G(dd);
  ch = I8(__fort_copy)(af, df, ad, dd, NULL);
  __fort_doit(ch);
  __fort_frechn(ch);

  /* free the old array */

  if (~F90_FLAGS_G(dd) & __NOT_COPIED)
    I8(__fort_deallocate)(db);

  F90_FLAGS_P(ad, F90_FLAGS_G(ad) & ~__NOT_COPIED);
#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d recopy ab=%x base=%x offset=%x\n", GET_DIST_LCPU, ab, base,
           *off);
  }
#endif
}

/* \brief realign the alignee with the align-target template 'td' 
 *
 *<pre>
 * varargs are:
 * [ __INT_T *collapse,
 *  { __INT_T *taxis, __INT_T *tstride, __INT_T *toffset, }*
 *    __INT_T *single, { __INT_T *coordinate, }* ]
 *</pre>
 */
void
ENTFTN(REALIGN, realign)(F90_Desc *ad, __INT_T *p_rank, __INT_T *p_flags,
                              F90_Desc *td, __INT_T *p_conform, ...)
{
  va_list va;
  DECL_HDR_VARS(dd);
  DECL_HDR_PTRS(ud);
  DECL_HDR_PTRS(prev);
  DECL_HDR_PTRS(next);
  DECL_DIM_PTRS(add);
  DECL_DIM_PTRS(ddd);
  DECL_DIM_PTRS(tdd);
  proc *ap, *tp;
  __INT_T flags, collapse, m, single = 0;
  __INT_T taxis[MAXDIMS], tstride[MAXDIMS], toffset[MAXDIMS];
  __INT_T coordinate[MAXDIMS];
  __INT_T ak, i, realign, rank, tk, tm, px, tx;

  rank = *p_rank;
  flags = *p_flags;

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d REALIGN alignee=%x new-align-target=%x\n", GET_DIST_LCPU, ad,
           td);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (ad == NULL || F90_TAG_G(ad) != __DESC)
    __fort_abort("REALIGN: invalid alignee descriptor");
  if (td == NULL || F90_TAG_G(td) != __DESC)
    __fort_abort("REALIGN: invalid new-align-target descriptor");
  if (~F90_FLAGS_G(ad) & __DYNAMIC)
    __fort_abort("REALIGN: alignee is not DYNAMIC");
  if (F90_RANK_G(ad) != rank)
    __fort_abort("REALIGN: alignee rank differs");
  if (flags &
      (__DIST_TARGET_MASK << __DIST_TARGET_SHIFT |
       __DIST_FORMAT_MASK << __DIST_FORMAT_SHIFT | __INHERIT | __SEQUENCE))
    __fort_abort("REALIGN: distribution, inherit, or sequence disallowed");
#endif

  va_start(va, p_conform);

  if (flags & __IDENTITY_MAP) {
    collapse = 0;
    for (i = 1; i <= rank; ++i) {
      taxis[i - 1] = i;
      tstride[i - 1] = 1;
      toffset[i - 1] = 0;
    }
  } else {
    collapse = *va_arg(va, __INT_T *);

    for (i = 0; i < rank; ++i) {
      if (collapse >> i & 1) {
        taxis[i] = 0;
        tstride[i] = 1;
        toffset[i] = 0;
      } else {
        taxis[i] = *va_arg(va, __INT_T *);
        tstride[i] = *va_arg(va, __INT_T *);
        toffset[i] = *va_arg(va, __INT_T *);
      }
    }
    single = *va_arg(va, __INT_T *);
    if (single >> F90_RANK_G(td))
      __fort_abort("REALIGN: invalid single alignment axis");
    for (i = 0; i < F90_RANK_G(td); ++i) {
      if (single >> i & 1)
        coordinate[i] = *va_arg(va, __INT_T *);
      else
        coordinate[i] = 0;
    }
  }
  va_end(va);

  ap = DIST_DIST_TARGET_G(ad);
  tp = DIST_DIST_TARGET_G(td);

  realign = (ap->base != tp->base || ap->size != tp->size);

  for (i = 0; !realign && i < rank; ++i) {
    SET_DIM_PTRS(add, ad, i);

    /* realignment required if different processor axes are
       targeted or if the processor shapes or strides differ */

    tx = taxis[i];
    if (tx > 0) {
      SET_DIM_PTRS(tdd, td, tx - 1);
      px = DIST_DPTR_PAXIS_G(tdd);
    } else
      px = 0;

    realign = (px != DIST_DPTR_PAXIS_G(add));
    if (realign)
      break;

    if (px == 0)
      continue; /* collapsed dimension */

    realign = (DIST_DPTR_PSHAPE_G(add) != DIST_DPTR_PSHAPE_G(tdd) ||
               DIST_DPTR_PSTRIDE_G(add) != DIST_DPTR_PSTRIDE_G(tdd));
    if (realign)
      break;

    /* realignment required if the template mappings aren't
       equivalent... */

    /* offset in ultimately-aligned template of actual array */

    ak = DIST_DPTR_TSTRIDE_G(add) * F90_DPTR_LBOUND_G(add) +
         DIST_DPTR_TOFFSET_G(add) - DIST_DPTR_TLB_G(add);

    /* mapping onto ultimate align-target */

    tm = DIST_DPTR_TSTRIDE_G(tdd) * tstride[i];
    tk = DIST_DPTR_TSTRIDE_G(tdd) * toffset[i] + DIST_DPTR_TOFFSET_G(tdd);

    /* offset in ultimately-aligned template of align-target */

    tk = tm * F90_DPTR_LBOUND_G(tdd) + tk - DIST_DPTR_TLB_G(tdd);

#if defined(DEBUG)
    if (__fort_test & DEBUG_RDST) {
      printf("%d target tm=%d tk=%d tb=%d tmab=%d tkab=%d\n", GET_DIST_LCPU,
             tm, tk, DIST_DPTR_BLOCK_G(tdd), tm * DIST_DPTR_BLOCK_G(add),
             tk * DIST_DPTR_BLOCK_G(add));
      printf("%d actual am=%d ak=%d ab=%d amtb=%d aktb=%d\n", GET_DIST_LCPU,
             DIST_DPTR_TSTRIDE_G(add), ak, DIST_DPTR_BLOCK_G(add),
             DIST_DPTR_TSTRIDE_G(add) * DIST_DPTR_BLOCK_G(tdd),
             ak * DIST_DPTR_BLOCK_G(tdd));
    }
#endif
    realign = (DIST_DPTR_BLOCK_G(tdd) * DIST_DPTR_TSTRIDE_G(add) !=
                   DIST_DPTR_BLOCK_G(add) * tm ||
               DIST_DPTR_BLOCK_G(tdd) * ak != DIST_DPTR_BLOCK_G(add) * tk);
  }

  ud = DIST_ALIGN_TARGET_G(ad);
#if defined(DEBUG)
  if (ud == NULL || F90_TAG_G(ud) != __DESC)
    __fort_abort("REALIGN: invalid old align-target descriptor");
  if (DIST_ALIGN_TARGET_G(ud) != ud)
    __fort_abort("REALIGN: old align-target is not ultimate align-target");
#endif

  if (F90_FLAGS_G(ud) & __DYNAMIC) {
    if (ud == ad) {

      /* array is distributee */

      if (DIST_NEXT_ALIGNEE_G(ad) != NULL)
        __fort_abort("REALIGN: array is dynamic align-target");
    } else {

      /* unlink from old ultimate align-target's alignees list */

      prev = ud;
      next = DIST_NEXT_ALIGNEE_G(ud);
      while (next != NULL && next != ad) {
        prev = next;
        next = DIST_NEXT_ALIGNEE_G(prev);
      }
      if (next != ad)
        __fort_abort("REALIGN: alignee not in old align-target's list");

      DIST_NEXT_ALIGNEE_P(prev, DIST_NEXT_ALIGNEE_G(ad));
#if defined(DEBUG)
      if (__fort_test & DEBUG_RDST) {
        printf("%d unlinked ud=%x prev=%x next=%lx\n", GET_DIST_LCPU, ud, prev,
               DIST_NEXT_ALIGNEE_G(prev));
      }
#endif
    }
  }

  if (realign) {

    /* make a copy of the old descriptor */

    I8(__fort_copy_descriptor)(dd, ad);

    /* update the descriptor in place.  init_descriptor links the
       descriptor to the new ultimate align-target. */

    ud = DIST_ALIGN_TARGET_G(td);

    __DIST_INIT_DESCRIPTOR(ad, F90_RANK_G(ad), F90_KIND_G(ad), F90_LEN_G(ad),
                          flags, ud);
    for (i = 1; i <= rank; ++i) {
      tx = taxis[i - 1];
      if (tx > 0) {
        SET_DIM_PTRS(tdd, td, tx - 1);
        tx = DIST_DPTR_TAXIS_G(tdd);
      }
      if (tx > 0) {
        tm = DIST_DPTR_TSTRIDE_G(tdd) * tstride[i - 1];
        tk = DIST_DPTR_TSTRIDE_G(tdd) * toffset[i - 1] + DIST_DPTR_TOFFSET_G(tdd);
      } else {
        tm = 1;
        tk = 0;
      }
      SET_DIM_PTRS(ddd, dd, i - 1);

      /*
       * added last arg which passes the gen_block field in...
       */

      I8(__fort_set_alignment)(ad, i, F90_DPTR_LBOUND_G(ddd), 
                                    DPTR_UBOUND_G(ddd), tx, tm, tk, 
                                    (tx>0)?(&DIST_DPTR_GEN_BLOCK_G(tdd)): 
                                           (&DIST_DPTR_GEN_BLOCK_G(ddd)));
    }
    /* NEC 127 / tpr 2597 */

    m = single;
    for (i = 1; m > 0; ++i, m >>= 1) {
      if (m & 1)
        I8(__fort_set_single)(ad, td, i, coordinate[i - 1], __SINGLE);
    }
    m = DIST_SINGLE_G(td);
    for (i = 1; m > 0; ++i, m >>= 1) {
      if (m & 1)
        I8(__fort_set_single)(ad, DIST_ALIGN_TARGET_G(td), i, 
                                     DIST_INFO_G(td, i-1), __SINGLE);
    }

    for (i = 1; i <= rank; ++i) {
      SET_DIM_PTRS(ddd, dd, i - 1);
      if (~F90_FLAGS_G(dd) & __TEMPLATE)
        __DIST_SET_ALLOCATION(ad, i, DIST_DPTR_NO_G(ddd), DIST_DPTR_PO_G(ddd));
    }

    I8(__fort_finish_descriptor)(ad);

#if defined(DEBUG)
    if (__fort_test & DEBUG_RDST) {
      printf("%d linked ud=%x next=%lx\n", GET_DIST_LCPU, ud,
             DIST_NEXT_ALIGNEE_G(ud));
    }
#endif

    /* reallocate and copy the old into the new */

    I8(recopy)(ad, dd, ad);
  } else {

    /* link to new align-target.  descriptor does not need to
       change and array does not need to be copied.  */

    DIST_ALIGN_TARGET_P(ad, DIST_ALIGN_TARGET_G(td));
    DIST_DIST_TARGET_P(ad, DIST_DIST_TARGET_G(td));
    DIST_NEXT_ALIGNEE_P(ad, DIST_NEXT_ALIGNEE_G(td));
    DIST_NEXT_ALIGNEE_P(td, ad);
  }
}

/** \brief redistribute the distributee and all objects that are currently
 * ultimately-aligned with it (within the scope of the calling
 * subprogram).  redistribution does not change alignment
 * relationships.
 *
 *<pre>
 * varargs are:
 * [ proc *dist_target, ]
 * __INT_T *isstar,
 * { [__INT_T paxis,](__INT_T *dstfmt, |
 *   (__INT_T * gen_block_array, __INT_T extent) ) }*
 *</pre>
 */
void
ENTFTN(REDISTRIBUTE, redistribute)(F90_Desc *dd, __INT_T *p_rank,
                                        __INT_T *p_flags, ...)
{
  va_list va;
  DECL_HDR_PTRS(ad);
  DECL_HDR_PTRS(ud);
  DECL_HDR_PTRS(next);
  DECL_DIM_PTRS(odd);
  DECL_DIM_PTRS(udd);
  proc *tp, *up;
  DECL_HDR_VARS(od);
  DECL_HDR_VARS(td);
  __INT_T nmapped, block[MAXDIMS];
  __INT_T flags, dist_format_spec, dist_target_spec;
  __INT_T isstar, paxis[MAXDIMS];
  __INT_T dfmt, ddfmt, tdfmt, i, rank, redistribute = 0, ux;

  __INT_T *gbCopy[MAXDIMS]; /*hold gen_block dims*/
  __INT_T gbIdx = 0, j;

  for (i = 0; i < MAXDIMS; ++i)
    gbCopy[i] = 0;

  rank = *p_rank;
  flags = *p_flags;

  dist_target_spec =
      (_io_spec)(flags >> __DIST_TARGET_SHIFT & __DIST_TARGET_MASK);
  dist_format_spec =
      (_io_spec)(flags >> __DIST_FORMAT_SHIFT & __DIST_FORMAT_MASK);

#if defined(DEBUG)
  if (dd == NULL || F90_TAG_G(dd) != __DESC)
    __fort_abort("REDISTRIBUTE: invalid distributee descriptor");
  if (F90_RANK_G(dd) != rank)
    __fort_abort("REDISTRIBUTE: distributee has incorrect rank");
  if (flags & (__ALIGN_TARGET_MASK << __ALIGN_TARGET_SHIFT | __SEQUENCE))
    __fort_abort("REDISTRIBUTE: invalid flags");
#endif

  ud = DIST_ALIGN_TARGET_G(dd);

#if defined(DEBUG)
  if (ud == NULL || F90_TAG_G(ud) != __DESC)
    __fort_abort("REDISTRIBUTE: invalid ultimate template descriptor");
  if (DIST_ALIGN_TARGET_G(ud) != ud)
    __fort_abort("REDISTRIBUTE: template is not ultimate align-target");
  if (~F90_FLAGS_G(ud) & __DYNAMIC)
    __fort_abort("REDISTRIBUTE: ultimate template is not DYNAMIC");

  if (__fort_test & DEBUG_RDST) {
    printf("%d REDISTRIBUTE distributee=%x ultimate template=%x\n",
           GET_DIST_LCPU, dd, ud);
    __fort_show_flags(flags);
    printf("\n");
  }
#endif
  if (F90_RANK_G(ud) != rank)
    __fort_abort("REDISTRIBUTE: ultimate template has incorrect rank");

  /* get distribution target spec */

  va_start(va, p_flags);

  switch (dist_target_spec) {

  case __PRESCRIPTIVE:
    tp = va_arg(va, proc *);
    break;

  case __OMITTED:
    tp = NULL;
    break;

  case __DESCRIPTIVE:
  case __TRANSCRIPTIVE:
  default:
    __fort_abort("REDISTRIBUTE: bad dist-target flags");
  }

  /* get distribution format spec */

  nmapped = 0; /* no. of distributed dimensions */
  ddfmt = 0;

  switch (dist_format_spec) {

  case __PRESCRIPTIVE:
    isstar = *va_arg(va, __INT_T *);
    for (i = 0; i < rank; ++i) {
      if (isstar >> i & 1) {
        paxis[i] = 0;
        block[i] = 0;
      } else if (((isstar & EXTENSION_BLOCK_MASK) >> (7 + 3 * i)) & 0x01) {

        /* 
         * got a gen_block dimension.  The arguments for
         * ENTFTN(redistribute) are slightly different for
         * gen_block, so we need to handle this as a special
         * case.
         */

        if (flags & __DIST_TARGET_AXIS) {
          paxis[i] = *va_arg(va, __INT_T *);
          if (paxis[i] != 0)
            ++nmapped;
        } else
          paxis[i] = ++nmapped;

        gbCopy[gbIdx++] = va_arg(va, __INT_T *);
        block[i] = *va_arg(va, __INT_T *);
        ddfmt |= DFMT_GEN_BLOCK << DFMT__WIDTH * i;
        if (DFMT(ud, i + 1) == DFMT_GEN_BLOCK) {

          /* NEC problem 211 / tpr 2488
           * redistribute if gen_block array changed.
           */

          int elem;
          __INT_T *newgb, *oldgb;

          newgb = DIST_DIM_GEN_BLOCK_G(ud, i);
          oldgb = gbCopy[gbIdx - 1];
          for (elem = 0; elem < block[i]; ++elem)

            if (*(oldgb + elem) != *(newgb + elem)) {
              redistribute = 1;
              break;
            }
        }
      }

      else {
        int dstfmt;

        if (flags & __DIST_TARGET_AXIS) {
          paxis[i] = *va_arg(va, __INT_T *);
          if (paxis[i] != 0)
            ++nmapped;
        } else
          paxis[i] = ++nmapped;

        dstfmt = *va_arg(va, __INT_T *);
        if (dstfmt >= 0) {

          block[i] = dstfmt;
          if (dstfmt == 0)
            ddfmt |= DFMT_BLOCK << DFMT__WIDTH * i;
          else
            ddfmt |= DFMT_BLOCK_K << DFMT__WIDTH * i;
        } else {
          block[i] = -dstfmt;
          if (dstfmt == -1)
            ddfmt |= DFMT_CYCLIC << DFMT__WIDTH * i;
          else
            ddfmt |= DFMT_CYCLIC_K << DFMT__WIDTH * i;
        }
      }
    }
    break;

  case __OMITTED:
    for (i = 0; i < rank; ++i) {
      paxis[i] = 0;
      block[i] = 0;
    }
    break;

  case __DESCRIPTIVE:
  case __TRANSCRIPTIVE:
  default:
    __fort_abort("REDISTRIBUTE: bad dist-format flags");
  }
  va_end(va);

  if (tp == NULL)
    tp = __fort_defaultproc(nmapped);
  else if (tp->tag != __PROC || tp->rank < nmapped)
    __fort_abort("REDISTRIBUTE: invalid dist-target");

  /* shuffle dist-formats to match align-target axis permutation */

  tdfmt = 0;
  dfmt = ddfmt;
  for (i = 0; i < rank; ++i, dfmt >>= DFMT__WIDTH) {
    if (dfmt & DFMT__MASK) {
      ux = DIST_DIM_TAXIS_G(dd, i);
      if (ux > 0)
        tdfmt |= (dfmt & DFMT__MASK) << DFMT__WIDTH * (ux - 1);
      else
        __fort_abort("REDISTRIBUTE: no align-target axis for mapped dim");
    }
  }

  /* check conformance of actual distribution vs. dist-target */

  redistribute |= (tdfmt != DIST_DFMT_G(ud));
  up = DIST_DIST_TARGET_G(ud);
  if (!redistribute && up != tp) {
    redistribute =
        (up->rank != tp->rank || up->base != tp->base || up->size != tp->size);
    for (i = 0; !redistribute && i < tp->rank; ++i)
      redistribute = (up->dim[i].shape != tp->dim[i].shape);
  }
  for (i = 0; !redistribute && i < rank; ++i) {
    ux = DIST_DIM_TAXIS_G(dd, i);
    SET_DIM_PTRS(udd, ud, ux - 1);
    redistribute = (DIST_DPTR_PAXIS_G(udd) != paxis[i]);
    if (redistribute)
      break;
    switch (DFMT(ud, ux)) {
    case DFMT_COLLAPSED:
    case DFMT_BLOCK:
    case DFMT_CYCLIC:
    case DFMT_GEN_BLOCK: 
      break;
    case DFMT_BLOCK_K:
    case DFMT_CYCLIC_K:
      redistribute = (DIST_DPTR_BLOCK_G(udd) != block[i]);
      break;
    default:
      __fort_abort("REDISTRIBUTE: unsupported dist-format");
    }
  }
#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    for (i = 0; i < rank; ++i) {
      printf("%d dim=%d ddfmt=%d paxis=%d block=%d\n", GET_DIST_LCPU, i + 1,
             ddfmt >> DFMT__WIDTH * i & DFMT__MASK, paxis[i], block[i]);
    }
    printf("%d nmapped=%d redistribute=%d\n", GET_DIST_LCPU, nmapped,
           redistribute);
  }
#endif

  if (!redistribute)
    return;

  /* create a duplicate of the new align-target first.  After all
     the alignees have been redistributed, then the original
     align-target is updated in place and the align-target pointer
     in each alignee is reset back to the original align-target.  */

  __DIST_INIT_DESCRIPTOR(td, rank, F90_KIND_G(ud), F90_LEN_G(ud), flags, tp);
  j = 0; /*for gen_block*/
  for (i = 1; i <= rank; ++i) {
    ux = DIST_DIM_TAXIS_G(dd, i - 1);
#if defined(DEBUG)
    if (ux <= 0)
      __fort_abort("REDISTRIBUTE: invalid distributee align axis");
#endif
    SET_DIM_PTRS(udd, ud, ux - 1);
    DIST_DFMT_P(td, tdfmt);

    if ((tdfmt >> DFMT__WIDTH * (i - 1) & DFMT__MASK) == DFMT_GEN_BLOCK) {
      DIST_DIM_GEN_BLOCK_P(td, i - 1, gbCopy[j++]);
    } else {
      DIST_DIM_GEN_BLOCK_P(td, i - 1, 0);
    }

    __DIST_SET_DISTRIBUTION(td, ux, F90_DPTR_LBOUND_G(udd), DPTR_UBOUND_G(udd),
                           paxis[i - 1], &block[i - 1]);

    if (~F90_FLAGS_G(ud) & __TEMPLATE)
      __DIST_SET_ALLOCATION(td, ux, DIST_DPTR_NO_G(udd), DIST_DPTR_PO_G(udd));
  }
  if (~F90_FLAGS_G(ud) & __TEMPLATE)
    I8(__fort_finish_descriptor)((td));

  /* reallocate and copy the old into the new */

  I8(recopy)(td, ud, ud);

  /* redistribute each alignee  */

  ad = DIST_NEXT_ALIGNEE_G(ud);
  while (ad != NULL) {

#if defined(DEBUG)
    if (ad == ud)
      __fort_abort("REDISTRIBUTE: distributee in own alignee's list");
    if (DIST_ALIGN_TARGET_G(ad) != ud)
      __fort_abort("REDISTRIBUTE: alignee has different align-target");
#endif

    /* make a copy of the old alignee descriptor */

    I8(__fort_copy_descriptor)(od, ad);

    /* update alignee descriptor in place */

    __DIST_INIT_DESCRIPTOR(ad, F90_RANK_G(od), F90_KIND_G(od), F90_LEN_G(od),
                          F90_FLAGS_G(od), td);
    for (i = 1; i <= F90_RANK_G(od); ++i) {
      int tx;

      SET_DIM_PTRS(odd, od, i - 1);

      /* 
       * pasing gen_block field in thru last arg
       */

      tx = DIST_DPTR_TAXIS_G(odd);

      I8(__fort_set_alignment)(ad, i, F90_DPTR_LBOUND_G(odd), 
                                    DPTR_UBOUND_G(odd), tx, 
                                    DIST_DPTR_TSTRIDE_G(odd), 
                                    DIST_DPTR_TOFFSET_G(odd), 
                                    &(DIST_DIM_GEN_BLOCK_G(td,tx-1)));
      if (~F90_FLAGS_G(od) & __TEMPLATE)
        __DIST_SET_ALLOCATION(ad, i, DIST_DPTR_NO_G(odd), DIST_DPTR_PO_G(odd));
    }
    if (~F90_FLAGS_G(od) & __TEMPLATE)
      I8(__fort_finish_descriptor)(ad);

    /* reallocate and copy the old into the new */

    I8(recopy)(ad, od, ad);

    /* reset pointers to original align-target and next alignee */

    next = DIST_NEXT_ALIGNEE_G(od);
    DIST_ALIGN_TARGET_P(ad, ud);
    DIST_NEXT_ALIGNEE_P(ad, next);
    ad = next;
  }

  /* copy the new align-target descriptor into the original location */

  next = DIST_NEXT_ALIGNEE_G(ud);
  I8(__fort_copy_descriptor)(ud, td);
  DIST_ALIGN_TARGET_P(ud, ud);
  DIST_NEXT_ALIGNEE_P(ud, next);
}
