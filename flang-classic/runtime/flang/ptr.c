/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief F90 pointer
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "type.h"

#include "fort_vars.h"

/* F90 POINTER COMMON format:
 * COMMON /ptr$sym/ void *ptr, void *off, F90_desc pd
 */

/* Disassociate pointer */

static void
I8(nullify)(char *pb, F90_Desc *pd, dtype kind, __CLEN_T len)
{
  __POINT_T *off;
  char *p, **ptr;

  ptr = ((char **)pd) - 2;      /* array pointer */
  off = (__POINT_T *)(ptr + 1); /* array offset */

  if (kind == __NONE) {
    *ptr = NULL;
    *off = 0;
  } else {
    p = I8(__fort_ptr_offset)(ptr, off, pb, kind, len, NULL);
    if (p != NULL)
      __fort_abort("NULLIFY: can't nullify pointer");
  }
  F90_TAG_P(pd, __NONE);
}

void
ENTFTN(NULLIFY, nullify)(char *pb, F90_Desc *pd)
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (F90_TAG_G(pd) == __NONE)
    return; /* already disassociated */
  else if (F90_TAG_G(pd) == __DESC) {
    kind = (dtype)F90_KIND_G(pd);
    len = F90_LEN_G(pd);
  } else if (ISSCALAR(pd)) {
    kind = (dtype)F90_TAG_G(pd);
    len = GET_DIST_SIZE_OF(kind);
  } else
    __fort_abort("NULLIFY: invalid descriptor");

  I8(nullify)(pb, pd, kind, len);
}

void
ENTFTN(NULLIFY_CHARA, nullify_chara)(DCHAR(pb), F90_Desc *pd DCLEN64(pb))
{
  if (F90_TAG_G(pd) == __NONE)
    return; /* already disassociated */
  else if (F90_TAG_G(pd) == __DESC) {
    if (F90_KIND_G(pd) != __STR || (size_t)F90_LEN_G(pd) != CLEN(pb))
      __fort_abort("NULLIFY: pointer type or length error");
  } else if (!ISSCALAR(pd))
    __fort_abort("NULLIFY: invalid descriptor");

  I8(nullify)(CADR(pb), pd, __STR, CLEN(pb));
}
/* 32 bit CLEN version */
void
ENTFTN(NULLIFY_CHAR, nullify_char)(DCHAR(pb), F90_Desc *pd DCLEN(pb))
{
  ENTFTN(NULLIFY_CHARA, nullify_chara)(CADR(pb), pd, (__CLEN_T)CLEN(pb));
}

/* same as NULLIFY but pointer is passed in, not the dereferenced pointer */
void
ENTFTN(NULLIFYX, nullifyx)(char **pb, F90_Desc *pd)
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (F90_TAG_G(pd) == __NONE)
    return; /* already disassociated */
  else if (F90_TAG_G(pd) == __DESC) {
    kind = (dtype)F90_KIND_G(pd);
    len = F90_LEN_G(pd);
  } else if (ISSCALAR(pd)) {
    kind = (dtype)F90_TAG_G(pd);
    len = GET_DIST_SIZE_OF(kind);
  } else
    __fort_abort("NULLIFY: invalid descriptor");

  I8(nullify)(*pb, pd, kind, len);
}

/* Associate pointer with target (whole array only) */

static void
I8(ptr_asgn)(char *pb, F90_Desc *pd, dtype kind, __CLEN_T len, char *tb,
             F90_Desc *td, __INT_T lb[])
{
  __POINT_T *off;
  char *p, **ptr;
  __INT_T tx, toffset, ubound;

  if (F90_TAG_G(td) == __DESC) {

    /* target is whole array -- create a new descriptor with
       target bounds and adjusted template mapping */

    __DIST_INIT_DESCRIPTOR(pd, F90_RANK_G(td), kind, len, F90_FLAGS_G(td), td);
    for (tx = 1; tx <= F90_RANK_G(td); ++tx) {
      DECL_DIM_PTRS(tdd);
      SET_DIM_PTRS(tdd, td, tx - 1);
      ubound = lb[tx - 1] + DPTR_UBOUND_G(tdd) - F90_DPTR_LBOUND_G(tdd);
      toffset =
          DIST_DPTR_TSTRIDE_G(tdd) * (F90_DPTR_LBOUND_G(tdd) - lb[tx - 1]) +
          DIST_DPTR_TOFFSET_G(tdd);

      /* 
       * Added gen_block arg in set_alignment
       */

      I8(__fort_set_alignment)(pd, tx, lb[tx-1], ubound,
               DIST_DPTR_TAXIS_G(tdd), DIST_DPTR_TSTRIDE_G(tdd), toffset, 
               (DIST_DPTR_TAXIS_G(tdd) > 0) ?
                 &DIST_DIM_GEN_BLOCK_G(td,(DIST_DPTR_TAXIS_G(tdd))-1) : 
                 &DIST_DPTR_GEN_BLOCK_G(tdd));
      I8(__fort_use_allocation)(pd, tx, DIST_DPTR_NO_G(tdd), 
                                     DIST_DPTR_PO_G(tdd), td);
    }
    I8(__fort_finish_descriptor)(pd);
  } else {

    /* target is scalar */

    F90_TAG_P(pd, F90_TAG_G(td));
  }

  ptr = ((char **)pd) - 2;      /* array pointer */
  off = (__POINT_T *)(ptr + 1); /* array offset */

  p = I8(__fort_ptr_offset)(ptr, off, pb, kind, len, tb);
  if (p != tb)
    __fort_abort("PTR_ASGN: can't align ptr base with target base");
}

void
ENTFTN(PTR_ASGN, ptr_asgn)(char *pb, F90_Desc *pd, char *tb, F90_Desc *td,
                            __INT_T lb[])
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASGN: invalid descriptor");
  } else if (!ISPRESENT(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC) {
    kind = F90_KIND_G(td);
    len = F90_LEN_G(td);
  } else if (ISSCALAR(td)) {
    kind = (dtype)F90_TAG_G(td);
    len = GET_DIST_SIZE_OF(kind);
  } else {
    /*__fort_abort("PTR_ASGN: invalid target");*/
    return;
  }

  I8(ptr_asgn)(pb, pd, kind, len, tb, td, lb);
}

void
ENTFTN(PTR_ASGN_CHARA, ptr_asgn_chara)(DCHAR(pb), F90_Desc *pd, DCHAR(tb),
                                     F90_Desc *td,
                                     __INT_T lb[] DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASGN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASGN_CHAR: invalid target");*/
    return;
  }

  if (CLEN(pb) != CLEN(tb))
    __fort_abort("PTR_ASGN: target length differs from pointer");

  I8(ptr_asgn)(CADR(pb), pd, kind, len, CADR(tb), td, lb);
}
/* 32 bit CLEN version */
void
ENTFTN(PTR_ASGN_CHAR, ptr_asgn_char)(DCHAR(pb), F90_Desc *pd, DCHAR(tb),
                                     F90_Desc *td,
                                     __INT_T lb[] DCLEN(pb) DCLEN(tb))
{
  ENTFTN(PTR_ASGN_CHARA, ptr_asgn_chara)(CADR(pb), pd, CADR(tb), td, lb,
                                      (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

/* Associate pointer with target array or section */
static void
I8(ptr_assign)(char *pb, F90_Desc *pd, dtype kind, __CLEN_T len, char *tb,
               F90_Desc *td, int sectflag)
{
  DECL_DIM_PTRS(tdd);
  char **ptr;
  __INT_T i;
  __INT_T gsize;

  if (F90_TAG_G(td) == __DESC) {

    /* target has a descriptor -- pointer must have one too */

    if (sectflag) {

      /* target is section, e.g. p => t(:,:) */

      F90_FLAGS_P(pd, (F90_FLAGS_G(pd) | __SEQUENTIAL_SECTION));
      gsize = 1;
      __DIST_INIT_SECTION(pd, F90_RANK_G(td), td);
      F90_GSIZE_P(pd, 1);
      F90_GBASE_P(pd, 0);
      for (i = 1; i <= F90_RANK_G(td); ++i) {
        SET_DIM_PTRS(tdd, td, i - 1);
        __DIST_SET_SECTIONXX(pd, i, td, i, F90_DPTR_LBOUND_G(tdd),
                            DPTR_UBOUND_G(tdd), 1, 0, gsize);
        /* check to see if its a sequential (continous) section */
      }
      F90_GSIZE_P(pd, gsize); /* global section size */
      /* no longer need section stride/section offset */
      F90_GBASE_P(pd, F90_GBASE_G(td));
    } else {

      /* target is whole array -- copy entire descriptor */

      __fort_bcopy((char *)pd, (char *)td, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
      SET_F90_DIST_DESC_PTR(pd, F90_RANK_G(pd));
      /* check for align-target to self */
      if (DIST_ALIGN_TARGET_G(td) == td) {
        DIST_ALIGN_TARGET_P(pd, pd);
      }
    }

  } else {

    /* target is scalar */

    F90_TAG_P(pd, F90_TAG_G(td));
  }

  if ((__CLEN_T)F90_LEN_G(pd) != len) {
    /* final multiplier must be contiguous */
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  ptr = ((char **)pd) - 2; /* array pointer */
  /* no longer need section stride/section offset */
  *ptr = tb;
}

void
ENTFTN(PTR_ASSIGN, ptr_assign)(char *pb, F90_Desc *pd, char *tb,
                                    F90_Desc *td, __INT_T *sectflag)
{
  dtype kind;
  __CLEN_T len;
  char **ptr;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSIGN: invalid descriptor");
  } else if (!ISPRESENT(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
    F90_TAG_P(pd, __NONE);
    ptr = ((char **)pd) - 2; /* actual array pointer */
    *ptr = (char *)0;
  } else if (F90_TAG_G(td) == __DESC) {
    kind = F90_KIND_G(td);
    len = F90_LEN_G(td);
    /* target has a descriptor -- pointer must have one too */
    if (*sectflag) {
      __INT_T gsize;
      __INT_T i, rank, flags, lbase;
      DECL_F90_DIM_PTR(tdd);
      DECL_F90_DIM_PTR(pdd);
      /* target is section, e.g. p => t(:,:) */
      gsize = 1;
      rank = F90_RANK_G(td);
      flags = F90_FLAGS_G(td);
      lbase = F90_LBASE_G(td);
      /* tag, rank, kind, len, flags, gsize, lsize, gbase, lbase */
      F90_TAG_P(pd, __DESC);
      F90_RANK_P(pd, rank);
      F90_KIND_P(pd, F90_KIND_G(td));
      F90_LEN_P(pd, F90_LEN_G(td));
      F90_LSIZE_P(pd, F90_LSIZE_G(td));
      F90_GBASE_P(pd, F90_GBASE_G(td));

      SET_DIM_PTRS(tdd, td, 0);
      SET_DIM_PTRS(pdd, pd, 0);
      for (i = 0; i < rank; ++i) {
        __INT_T __extent, __myoffset, __stride;
        __extent = F90_DPTR_EXTENT_G(tdd); /* section extent */
        __myoffset = F90_DPTR_LBOUND_G(tdd) - 1;
        __stride = F90_DPTR_LSTRIDE_G(tdd);
        F90_DPTR_LBOUND_P(pdd, 1);    /* lower bound */
        DPTR_UBOUND_P(pdd, __extent); /* upper bound */
        F90_DPTR_SSTRIDE_P(pdd, 1);   /* placeholders */
        F90_DPTR_SOFFSET_P(pdd, 0);
        F90_DPTR_LSTRIDE_P(pdd, __stride);
        lbase += __myoffset * __stride;
        if (__stride != gsize)
          flags &= ~__SEQUENTIAL_SECTION;
        gsize *= __extent;
        ++F90_DIM_NAME(tdd);
        ++F90_DIM_NAME(pdd);
      }
      F90_LBASE_P(pd, lbase);
      F90_FLAGS_P(pd, flags);
      F90_GSIZE_P(pd, gsize); /* global section size */
    } else {
      /* target is whole array -- copy entire descriptor */
      __fort_bcopy((char *)pd, (char *)td, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
    }
    ptr = ((char **)pd) - 2; /* actual array pointer */
    *ptr = tb;
  } else if (ISSCALAR(td)) {
    kind = (dtype)F90_TAG_G(td);
    len = GET_DIST_SIZE_OF(kind);
    F90_TAG_P(pd, F90_TAG_G(td));
    ptr = ((char **)pd) - 2; /* actual pointer */
    *ptr = tb;
  } else {
    /*__fort_abort("PTR_ASSIGN: invalid target");*/
    return;
  }

}

void
ENTFTN(PTR_ASSIGN_CHARA, ptr_assign_chara)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
         __INT_T *sectflag DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSIGN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASSIGN_CHAR: invalid target");*/
    return;
  }

  if (CLEN(pb) != CLEN(tb))
    __fort_abort("PTR_ASSIGN: target length differs from pointer");

  I8(ptr_assign)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);
  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
}
/* 32 bit CLEN version */
void
ENTFTN(PTR_ASSIGN_CHAR, ptr_assign_char)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
         __INT_T *sectflag DCLEN(pb) DCLEN(tb))
{
  ENTFTN(PTR_ASSIGN_CHARA, ptr_assign_chara)(CADR(pb), pd, CADR(tb), td,
                             sectflag, (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

void
ENTFTN(PTR_ASSIGNX, ptr_assignx)
        (char *pb, F90_Desc *pd, char *tb, F90_Desc *td, __INT_T *sectflag,
         __INT_T *targetlen, __INT_T *targettype)
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSIGN: invalid descriptor");
  } else if (!ISPRESENT(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC) {
    kind = F90_KIND_G(td);
    len = F90_LEN_G(td);
  } else if (ISSCALAR(td)) {
    kind = (dtype)F90_TAG_G(td);
    len = GET_DIST_SIZE_OF(kind);
  } else {
    /*__fort_abort("PTR_ASSIGN: invalid target");*/
    return;
  }

  I8(ptr_assign)(pb, pd, kind, len, tb, td, *sectflag);
  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION) ||
      (targetlen && F90_LEN_G(pd) != *targetlen)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  F90_KIND_P(pd, *targettype);
}

void
ENTFTN(PTR_ASSIGN_CHARXA, ptr_assign_charxa)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag, __CLEN_T *targetlen,
          __INT_T *targettype DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSIGN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASSIGN_CHAR: invalid target");*/
    return;
  }

  if (CLEN(pb) != CLEN(tb))
    __fort_abort("PTR_ASSIGN: target length differs from pointer");

  I8(ptr_assign)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);
  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION) ||
      (targetlen && (__CLEN_T)F90_LEN_G(pd) != *targetlen)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  F90_KIND_P(pd, *targettype);
}
/* 32 bit CLEN version */
void
ENTFTN(PTR_ASSIGN_CHARX, ptr_assign_charx)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag, __INT_T *targetlen,
          __INT_T *targettype DCLEN(pb) DCLEN(tb))
{
  ENTFTN(PTR_ASSIGN_CHARXA, ptr_assign_charxa) (CADR(pb), pd, CADR(tb), td,
      sectflag, (__CLEN_T *)targetlen, targettype, (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

void
ENTFTN(PTR_ASSIGN_ASSUMESHP, ptr_assign_assumeshp)
         (char *pb, F90_Desc *pd, char *tb, F90_Desc *td, __INT_T *sectflag)
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSIGN: invalid descriptor");
  } else if (!ISPRESENT(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC) {
    kind = F90_KIND_G(td);
    len = F90_LEN_G(td);
  } else if (ISSCALAR(td)) {
    kind = (dtype)F90_TAG_G(td);
    len = GET_DIST_SIZE_OF(kind);
  } else {
    /*__fort_abort("PTR_ASSIGN: invalid target");*/
    return;
  }

  I8(ptr_assign)(pb, pd, kind, len, tb, td, *sectflag);

  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
}

void
ENTFTN(PTR_ASSIGN_CHAR_ASSUMESHPA, ptr_assign_char_assumeshpa)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSIGN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else
    __fort_abort("PTR_ASSIGN_CHAR: invalid target");

  if (CLEN(pb) != CLEN(tb))
    __fort_abort("PTR_ASSIGN: target length differs from pointer");

  I8(ptr_assign)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);

  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
}
/* 32 bit CLEN version */
void
ENTFTN(PTR_ASSIGN_CHAR_ASSUMESHP, ptr_assign_char_assumeshp)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag DCLEN(pb) DCLEN(tb))
{
  ENTFTN(PTR_ASSIGN_CHAR_ASSUMESHPA, ptr_assign_char_assumeshpa)(CADR(pb), pd, 
               CADR(tb), td, sectflag, (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

void
ENTFTN(PTR_FIX_ASSUMESHP1, ptr_fix_assumeshp1)(F90_Desc *sd, __INT_T lb1)
{
  __INT_T lbase;

  lbase = 1;
  F90_DIM_LBOUND_G(sd, 0) = lb1;
  lbase -= F90_DIM_LSTRIDE_G(sd, 0) * lb1;
  F90_LBASE_P(sd, lbase);
}

void
ENTFTN(PTR_FIX_ASSUMESHP2, ptr_fix_assumeshp2)(F90_Desc *sd, __INT_T lb1,
                                               __INT_T lb2)
{
  __INT_T lbase;

  lbase = 1;
  F90_DIM_LBOUND_G(sd, 0) = lb1;
  lbase -= F90_DIM_LSTRIDE_G(sd, 0) * lb1;
  F90_DIM_LBOUND_G(sd, 1) = lb2;
  lbase -= F90_DIM_LSTRIDE_G(sd, 1) * lb2;
  F90_LBASE_P(sd, lbase);
}

void
ENTFTN(PTR_FIX_ASSUMESHP3, ptr_fix_assumeshp3)(F90_Desc *sd, __INT_T lb1,
                                               __INT_T lb2, __INT_T lb3)
{
  __INT_T lbase;

  lbase = 1;
  F90_DIM_LBOUND_G(sd, 0) = lb1;
  lbase -= F90_DIM_LSTRIDE_G(sd, 0) * lb1;
  F90_DIM_LBOUND_G(sd, 1) = lb2;
  lbase -= F90_DIM_LSTRIDE_G(sd, 1) * lb2;
  F90_DIM_LBOUND_G(sd, 2) = lb3;
  lbase -= F90_DIM_LSTRIDE_G(sd, 2) * lb3;
  F90_LBASE_P(sd, lbase);
}

void
ENTFTN(PTR_FIX_ASSUMESHP, ptr_fix_assumeshp)(F90_Desc *sd, __INT_T rank, ...)
{
  va_list va;
  int ii;
  __INT_T lbase;
  __INT_T lb;

  va_start(va, rank);
  lbase = 1;
  for (ii = 0; ii < rank; ii++) {
    lb = va_arg(va, __INT_T);
    F90_DIM_LBOUND_G(sd, ii) = lb;
    lbase -= F90_DIM_LSTRIDE_G(sd, ii) * lb;
  }
  F90_LBASE_P(sd, lbase);
  va_end(va);
}

/* Copy in pointer argument */

static void
I8(ptr_in)(__INT_T rank, /* dummy rank */
           dtype kind,   /* dummy type-kind */
           __CLEN_T len,  /* dummy element byte length */
           char *db,     /* dummy array base address */
           F90_Desc *dd, /* dummy descriptor */
           char *ab,     /* actual array base address */
           F90_Desc *ad) /* actual descriptor */
{
  char **aptr;     /* actual cray pointer variable */
  char **dptr;     /* dummy cray pointer variable */
  __POINT_T *doff; /* dummy offset variable */

  if (!ISPRESENT(ab)) {

    /* absent pointer argument */

    dptr = ((char **)dd) - 2;
    doff = (__POINT_T *)(dptr + 1);

    (void)I8(__fort_ptr_offset)(
        dptr, doff, db, kind, len,
        (kind == __STR ? (char *)ABSENTC : (char *)ABSENT));
    F90_TAG_P(dd, __NONE);
    return;
  }

  /* pointer argument is present */

  if (F90_TAG_G(ad) == __NONE) {

    /* pointer argument is disassociated.  disassociate the
       pointer dummy */

    I8(nullify)(db, dd, kind, len);
    return;
  }

  /* pointer argument is associated.  associate the dummy pointer
     with the actual pointer's target */

  if (ISSCALAR(ad)) {

    /* pointer argument is associated with a scalar */

    if (F90_TAG_G(ad) != kind) {
      F90_TAG_G(ad) = __NONE; /* initialize so ptr_out() is ok */
      return;
    }
    if (rank != 0) {
      F90_TAG_G(ad) = __NONE; /* initialize so ptr_out() is ok */
      return;
    }
  } else if (F90_TAG_G(ad) == __DESC) {

    /* pointer argument is associated with an array */

    if (F90_RANK_G(ad) != rank) {
      F90_TAG_G(ad) = __NONE; /* initialize so ptr_out() is ok */
      return;
    }
    if (F90_KIND_G(ad) != kind) {
      F90_TAG_G(ad) = __NONE; /* initialize so ptr_out() is ok */
      return;
    }
  } else {
    /* Assume it's an uninitialized descriptor of a pointer member
     * a derived type
     */
    F90_TAG_G(ad) = __NONE; /* initialize so ptr_out() is ok */
    return;
  }

  aptr = ((char **)ad) - 2;
  I8(ptr_assign)(db, dd, kind, len, *aptr, ad, 0);
}

void
ENTFTN(PTR_INA, ptr_ina)(__INT_T *rank, __INT_T *kind, __CLEN_T *len, char *db,
                       F90_Desc *dd, char *ab, F90_Desc *ad)
{
  I8(ptr_in)(*rank, (dtype)*kind, *len, db, dd, ab, ad);
}
/* 32 bit CLEN version */
void
ENTFTN(PTR_IN, ptr_in)(__INT_T *rank, __INT_T *kind, __INT_T *len, char *db,
                       F90_Desc *dd, char *ab, F90_Desc *ad)
{
  ENTFTN(PTR_INA, ptr_ina)(rank, kind, (__CLEN_T *)len, db, dd, ab, ad);
}

void
ENTFTN(PTR_IN_CHARA, ptr_in_chara)
         (__INT_T *rank, __INT_T *kind, __CLEN_T *len, DCHAR(db), F90_Desc *dd,
          DCHAR(ab), F90_Desc *ad DCLEN64(db) DCLEN64(ab))
{
  I8(ptr_in)(*rank, (dtype)*kind, *len, CADR(db), dd, CADR(ab), ad);
}
void
ENTFTN(PTR_IN_CHAR, ptr_in_char)
         (__INT_T *rank, __INT_T *kind, __INT_T *len, DCHAR(db), F90_Desc *dd,
          DCHAR(ab), F90_Desc *ad DCLEN(db) DCLEN(ab))
{
  ENTFTN(PTR_IN_CHARA, ptr_in_chara)(rank, kind, (__CLEN_T *)len, CADR(db), dd,
                         CADR(ab), ad, (__CLEN_T)CLEN(db), (__CLEN_T)CLEN(ab));
}

/* Copy out pointer argument (actual is present) */

static void
I8(ptr_out)(char *ab,     /* actual array base address */
            F90_Desc *ad, /* actual descriptor */
            char *db,     /* dummy array base address */
            F90_Desc *dd, /* dummy descriptor */
            dtype kind,   /* dummy type-kind  */
            __CLEN_T len)      /* dummy element byte length */
{
  char **dptr;

  if (F90_TAG_G(dd) == __NONE) {

    /* dummy is disassociated.  disassociate the actual pointer */

    if (ISSCALAR(ad))
      I8(nullify)(ab, ad, (dtype)F90_TAG_G(ad),
			GET_DIST_SIZE_OF(F90_TAG_G(ad)));
    else if (F90_TAG_G(ad) == __DESC)
      I8(nullify)(ab, ad, (dtype)F90_KIND_G(ad), F90_LEN_G(ad));
    else if (F90_TAG_G(ad) != __NONE)
      __fort_abort("PTR_OUT: invalid actual descriptor");
    return;
  }
  dptr = ((char **)dd) - 2;
  I8(ptr_assign)(ab, ad, kind, len, *dptr, dd, 0);
}

/* Associate pointer with target array or section */
static void * 
I8(ptr_assn)(char *pb, F90_Desc *pd, dtype kind, __CLEN_T len, char *tb,
             F90_Desc *td, int sectflag)
{
  DECL_DIM_PTRS(tdd);
  void *res;
  __INT_T i;
  __INT_T gsize;

  if (F90_TAG_G(td) == __DESC) {

    /* target has a descriptor -- pointer must have one too */

    if (sectflag) {

      /* target is section, e.g. p => t(:,:) */

      F90_FLAGS_P(pd, (F90_FLAGS_G(pd) | __SEQUENTIAL_SECTION));
      gsize = 1;
      __DIST_INIT_SECTION(pd, F90_RANK_G(td), td);
      F90_GSIZE_P(pd, 1);
      F90_GBASE_P(pd, 0);
      for (i = 1; i <= F90_RANK_G(td); ++i) {
        SET_DIM_PTRS(tdd, td, i - 1);
        __DIST_SET_SECTIONXX(pd, i, td, i, F90_DPTR_LBOUND_G(tdd),
                            DPTR_UBOUND_G(tdd), 1, 0, gsize);
        /* check to see if its a sequential (continous) section */
      }
      F90_GSIZE_P(pd, gsize); /* global section size */
      /* no longer need section stride/section offset */
      F90_GBASE_P(pd, F90_GBASE_G(td));
    } else {

      /* target is whole array -- copy entire descriptor */

      __fort_bcopy((char *)pd, (char *)td, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
      SET_F90_DIST_DESC_PTR(pd, F90_RANK_G(pd));
      /* check for align-target to self */
      if (DIST_ALIGN_TARGET_G(td) == td) {
        DIST_ALIGN_TARGET_P(pd, pd);
      }
    }

  } else {

    /* target is scalar */

    F90_TAG_P(pd, F90_TAG_G(td));
  }

  if ((__CLEN_T)F90_LEN_G(pd) != len) {
    /* final multiplier must be contiguous */
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  /* no longer need section stride/section offset */
  res = tb;
  return res;
}

void
ENTF90(TMP_DESC,tmp_desc)(F90_Desc *nd, F90_Desc *od)
{
  /* nd is the new, temporary argument descriptor, od is original descriptor */
  dtype kind;
  __CLEN_T len;

  if (nd == NULL || od == NULL) {
    __fort_abort("TMP_DESC: invalid descriptor");
  } else if (F90_TAG_G(od) == __DESC) {
    kind = F90_KIND_G(od);
    len = F90_LEN_G(od);
    __INT_T gsize;
    __INT_T i, rank, flags, lbase;
    DECL_F90_DIM_PTR(odd);
    DECL_F90_DIM_PTR(ndd);
    gsize = 1;
    rank = F90_RANK_G(od);
    flags = F90_FLAGS_G(od);
    lbase = F90_LBASE_G(od);
    /* tag, rank, kind, len, flags, gsize, lsize, gbase, lbase */
    F90_TAG_P(nd, __DESC);
    F90_RANK_P(nd, rank);
    F90_KIND_P(nd, F90_KIND_G(od));
    F90_LEN_P(nd, F90_LEN_G(od));
    F90_LSIZE_P(nd, F90_LSIZE_G(od));
    F90_GBASE_P(nd, F90_GBASE_G(od));

    SET_DIM_PTRS(odd, od, 0);
    SET_DIM_PTRS(ndd, nd, 0);
    for (i = 0; i < rank; ++i) {
      __INT_T __extent, __myoffset, __stride;
      __extent = F90_DPTR_EXTENT_G(odd); /* section extent */
      __myoffset = F90_DPTR_LBOUND_G(odd) - 1;
      __stride = F90_DPTR_LSTRIDE_G(odd);
      F90_DPTR_LBOUND_P(ndd, 1);    /* lower bound */
      DPTR_UBOUND_P(ndd, __extent); /* upper bound */
      F90_DPTR_SSTRIDE_P(ndd, 1);   /* placeholders */
      F90_DPTR_SOFFSET_P(ndd, 0);
      F90_DPTR_LSTRIDE_P(ndd, __stride);
      lbase += __myoffset * __stride;
      if (__stride != gsize)
        flags &= ~__SEQUENTIAL_SECTION;
      gsize *= __extent;
      ++F90_DIM_NAME(odd);
      ++F90_DIM_NAME(ndd);
    }
    F90_LBASE_P(nd, lbase);
    F90_FLAGS_P(nd, flags);
    F90_GSIZE_P(nd, gsize); /* global section size */
  } else {
    __fort_abort("TMP_DESC: invalid original");
  }
}

void *
ENTFTN(PTR_ASSN, ptr_assn)(char *pb, F90_Desc *pd, char *tb, F90_Desc *td,
                            __INT_T *sectflag)
{
  dtype kind;
  __CLEN_T len;
  void *res = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENT(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
    F90_TAG_P(pd, __NONE);
    res = (char *)0;
  } else if (F90_TAG_G(td) == __DESC) {
    kind = F90_KIND_G(td);
    len = F90_LEN_G(td);
    /* target has a descriptor -- pointer must have one too */
    if (*sectflag) {
      __INT_T gsize;
      __INT_T i, rank, flags, lbase;
      DECL_F90_DIM_PTR(tdd);
      DECL_F90_DIM_PTR(pdd);
      /* target is section, e.g. p => t(:,:) */
      gsize = 1;
      rank = F90_RANK_G(td);
      flags = F90_FLAGS_G(td);
      lbase = F90_LBASE_G(td);
      /* tag, rank, kind, len, flags, gsize, lsize, gbase, lbase */
      F90_TAG_P(pd, __DESC);
      F90_RANK_P(pd, rank);
      F90_KIND_P(pd, F90_KIND_G(td));
      F90_LEN_P(pd, F90_LEN_G(td));
      F90_LSIZE_P(pd, F90_LSIZE_G(td));
      F90_GBASE_P(pd, F90_GBASE_G(td));

      SET_DIM_PTRS(tdd, td, 0);
      SET_DIM_PTRS(pdd, pd, 0);
      for (i = 0; i < rank; ++i) {
        __INT_T __extent, __myoffset, __stride;
        __extent = F90_DPTR_EXTENT_G(tdd); /* section extent */
        __myoffset = F90_DPTR_LBOUND_G(tdd) - 1;
        __stride = F90_DPTR_LSTRIDE_G(tdd);
        F90_DPTR_LBOUND_P(pdd, 1);    /* lower bound */
        DPTR_UBOUND_P(pdd, __extent); /* upper bound */
        F90_DPTR_SSTRIDE_P(pdd, 1);   /* placeholders */
        F90_DPTR_SOFFSET_P(pdd, 0);
        F90_DPTR_LSTRIDE_P(pdd, __stride);
        lbase += __myoffset * __stride;
        if (__stride != gsize)
          flags &= ~__SEQUENTIAL_SECTION;
        gsize *= __extent;
        ++F90_DIM_NAME(tdd);
        ++F90_DIM_NAME(pdd);
      }
      F90_LBASE_P(pd, lbase);
      F90_FLAGS_P(pd, flags);
      F90_GSIZE_P(pd, gsize); /* global section size */
    } else {
      /* target is whole array -- copy entire descriptor */
      __fort_bcopy((char *)pd, (char *)td, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
    }
    res = tb;
  } else if (ISSCALAR(td)) {
    kind = (dtype)F90_TAG_G(td);
    len = GET_DIST_SIZE_OF(kind);
    F90_TAG_P(pd, F90_TAG_G(td));
    res = tb;
  } else {
    /*__fort_abort("PTR_ASSN: invalid target");*/
    kind = __NONE;
    len = 0;
    return tb;
  }

  return res;
}

void *
ENTFTN(PTR_ASSN_CHARA, ptr_assn_chara)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;
  void *res;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASSN_CHAR: invalid target");*/
    return CADR(tb);
  }

  if (CLEN(pb) != CLEN(tb))
    __fort_abort("PTR_ASSN: target length differs from pointer");

  res = I8(ptr_assn)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);
  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  return res;
}
/* 32 bit CLEN version */
void *
ENTFTN(PTR_ASSN_CHAR, ptr_assn_char)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag DCLEN(pb) DCLEN(tb))
{
  return ENTFTN(PTR_ASSN_CHARA, ptr_assn_chara)(CADR(pb), pd, CADR(tb), td, sectflag,
                                      (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

void *
ENTFTN(PTR_ASSN_CHARA, ptr_assn_dchara)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;
  void *res;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASSN_CHAR: invalid target");*/
    return CADR(tb);
  }

  /*    if (CLEN(pb) != CLEN(tb))
          __fort_abort("PTR_ASSN: target length differs from pointer");
  */
  res = I8(ptr_assn)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);
  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  return res;
}
/* 32 bit CLEN version */
void *
ENTFTN(PTR_ASSN_CHAR, ptr_assn_dchar)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag DCLEN(pb) DCLEN(tb))
{
  return ENTFTN(PTR_ASSN_CHARA, ptr_assn_dchara) (CADR(pb), pd, CADR(tb), td,
                              sectflag, (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

void *
ENTFTN(PTR_ASSNXA, ptr_assnxa)
         (char *pb, F90_Desc *pd, char *tb, F90_Desc *td, __INT_T *sectflag,
          __CLEN_T *targetlen, __INT_T *targettype)
{
  dtype kind = 0;
  __CLEN_T len = 0;
  void *res;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENT(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC) {
    kind = F90_KIND_G(td);
    len = F90_LEN_G(td);
  } else if (ISSCALAR(td)) {
    kind = (dtype)F90_TAG_G(td);
    len = GET_DIST_SIZE_OF(kind);
  } else {
    /*__fort_abort("PTR_ASSN: invalid target");*/
    return 0;
  }

  res = I8(ptr_assn)(pb, pd, kind, len, tb, td, *sectflag);
  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION) ||
      (targetlen && (__CLEN_T)F90_LEN_G(pd) != *targetlen)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  F90_KIND_P(pd, *targettype);
  return res;
}
/* 32 bit CLEN version */
void *
ENTFTN(PTR_ASSNX, ptr_assnx)
         (char *pb, F90_Desc *pd, char *tb, F90_Desc *td, __INT_T *sectflag,
          __INT_T *targetlen, __INT_T *targettype)
{
  return ENTFTN(PTR_ASSNXA, ptr_assnxa) (pb, pd, tb, td, sectflag,
                                 (__CLEN_T *)targetlen, targettype);
}

void *
ENTFTN(PTR_ASSN_CHARXA,
       ptr_assn_charxa)(DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
       __INT_T *sectflag, __CLEN_T *targetlen,
      __INT_T *targettype DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;
  void *res;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASSN_CHAR: invalid target");*/
    return CADR(tb);
  }

  if (CLEN(pb) != CLEN(tb))
    __fort_abort("PTR_ASSN: target length differs from pointer");

  res = I8(ptr_assn)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);
  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION) ||
      (targetlen && (__CLEN_T)F90_LEN_G(pd) != *targetlen)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  F90_KIND_P(pd, *targettype);
  return res;
}
/* 32 bit CLEN version */
void *
ENTFTN(PTR_ASSN_CHARX,
       ptr_assn_charx)(DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
       __INT_T *sectflag, __INT_T *targetlen,
      __INT_T *targettype DCLEN(pb) DCLEN(tb))
{
  return ENTFTN(PTR_ASSN_CHARXA, ptr_assn_charxa)(CADR(pb), pd, CADR(tb), td,
                sectflag, (__CLEN_T *)targetlen, targettype, (__CLEN_T)CLEN(pb),
                (__CLEN_T)CLEN(tb));
}

void *
ENTFTN(PTR_ASSN_DCHARXA, ptr_assn_dcharxa)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td, __INT_T *sectflag,
          __CLEN_T *targetlen, __INT_T *targettype DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;
  void *res;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASSN_CHAR: invalid target");*/
    return CADR(tb);
  }

  /*    if (CLEN(pb) != CLEN(tb))
          __fort_abort("PTR_ASSN: target length differs from pointer");
  */
  res = I8(ptr_assn)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);
  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION) ||
      (targetlen && (__CLEN_T)F90_LEN_G(pd) != *targetlen)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  F90_KIND_P(pd, *targettype);
  return res;
}
/* 32 bit CLEN version */
void *
ENTFTN(PTR_ASSN_CHARX, ptr_assn_dcharx)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td, __INT_T *sectflag,
          __INT_T *targetlen, __INT_T *targettype DCLEN(pb) DCLEN(tb))
{
  return ENTFTN(PTR_ASSN_DCHARXA, ptr_assn_dcharxa) (CADR(pb), pd, CADR(tb), td,
                sectflag, (__CLEN_T *)targetlen, targettype, (__CLEN_T)CLEN(pb),
                (__CLEN_T)CLEN(tb));
}

void *
ENTFTN(PTR_ASSN_ASSUMESHP, ptr_assn_assumeshp)
         (char *pb, F90_Desc *pd, char *tb, F90_Desc *td, __INT_T *sectflag)
{
  dtype kind = 0;
  __CLEN_T len = 0;
  void *res;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENT(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC) {
    kind = F90_KIND_G(td);
    len = F90_LEN_G(td);
  } else if (ISSCALAR(td)) {
    kind = (dtype)F90_TAG_G(td);
    len = GET_DIST_SIZE_OF(kind);
  } else {
    /*__fort_abort("PTR_ASSN: invalid target");*/
    return tb;
  }

  res = I8(ptr_assn)(pb, pd, kind, len, tb, td, *sectflag);

  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  return res;
}

void *
ENTFTN(PTR_ASSN_CHAR_ASSUMESHPA, ptr_assn_char_assumeshpa)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;
  void *res;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASSN_CHAR: invalid target");*/
    return CADR(tb);
  }

  if (CLEN(pb) != CLEN(tb))
    __fort_abort("PTR_ASSN: target length differs from pointer");

  res = I8(ptr_assn)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);

  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  return res;
}
/* 32 bit CLEN version */
void *
ENTFTN(PTR_ASSN_CHAR_ASSUMESHP, ptr_assn_char_assumeshp)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
          __INT_T *sectflag DCLEN(pb) DCLEN(tb))
{
  return ENTFTN(PTR_ASSN_CHAR_ASSUMESHPA, ptr_assn_char_assumeshpa)(CADR(pb),
                pd, CADR(tb), td, sectflag, (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

void *
ENTFTN(PTR_ASSN_DCHAR_ASSUMESHPA, ptr_assn_dchar_assumeshpa)
  (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
    __INT_T *sectflag DCLEN64(pb) DCLEN64(tb))
{
  dtype kind = 0;
  __CLEN_T len = 0;
  void *res;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_ASSN: invalid descriptor");
  } else if (!ISPRESENTC(tb) || F90_TAG_G(td) == __NONE) {
    kind = __NONE;
    len = 0;
  } else if (F90_TAG_G(td) == __DESC || F90_TAG_G(td) == __STR) {
    kind = __STR;
    len = CLEN(tb);
  } else {
    /*__fort_abort("PTR_ASSN_CHAR: invalid target");*/
    return CADR(tb);
  }

  /*    if (CLEN(pb) != CLEN(tb))
          __fort_abort("PTR_ASSN: target length differs from pointer");
  */
  res = I8(ptr_assn)(CADR(pb), pd, kind, len, CADR(tb), td, *sectflag);

  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  return res;
}
/* 32 bit CLEN version */
void *
ENTFTN(PTR_ASSN_CHAR_ASSUMESHP, ptr_assn_dchar_assumeshp)
  (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td,
    __INT_T *sectflag DCLEN(pb) DCLEN(tb))
{
  return ENTFTN(PTR_ASSN_DCHAR_ASSUMESHPA, ptr_assn_dchar_assumeshpa)(CADR(pb),
                pd, CADR(tb), td, sectflag, (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

void * 
ENTFTN(PTR_SHAPE_ASSNX, ptr_shape_assnx)
       (char *pb, F90_Desc *pd, char *tb, F90_Desc *td,
       __INT_T *sectflag, __INT_T *targetlen, __INT_T *targettype,
       __INT_T *rank, ...)
/* {int lb}* */
{
  void *res = 0;
  va_list va;
  F90_Desc *new_td = 0;
  __INT_T dimflags = 0, stride[MAXDIMS], *lb = 0, *ub = 0;
  int notSet = 0;
  int i, reshape;
  __INT_T lbase = 0, tstride;
  int sz = *rank;
  int extent_diff;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_SHAPE_ASSNX: invalid descriptor");
  }

  if (rank && *rank) {

    reshape = (sz != F90_RANK_G(td));
    if (reshape && F90_RANK_G(td) != 1) {
      __fort_abort("PTR_SHAPE_ASSNX: pointer target must have a rank of 1"
                  " when pointer rank does not equal target rank");
    }

    notSet = (F90_TAG_G(pd) != __DESC);
    if (pd == td) {
      new_td = __fort_malloc(SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
      if (!new_td)
        __fort_abort("PTR_SHAPE_ASSNX: out of memory");
      __fort_bcopy((char *)new_td, (char *)td, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
      SET_F90_DIST_DESC_PTR(new_td, F90_RANK_G(new_td));
      /* check for align-target to self */
      if (DIST_ALIGN_TARGET_G(td) == td) {
        DIST_ALIGN_TARGET_P(new_td, new_td);
      }
      td = new_td;
    } else {
      __fort_bcopy((char *)pd, (char *)td, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
      SET_F90_DIST_DESC_PTR(pd, sz);
      /* check for align-target to self */
      if (DIST_ALIGN_TARGET_G(pd) == pd) {
        DIST_ALIGN_TARGET_P(pd, pd);
      }
    }
    lb = __fort_malloc(sizeof(__INT_T) * sz);
    ub = __fort_malloc(sizeof(__INT_T) * sz);
    if (!lb || !ub) {
      __fort_abort("PTR_SHAPE_ASSNX: out of memory");
    }
    va_start(va, rank);
    for (i = 0; i < sz; ++i) {
      DECL_DIM_PTRS(tdd);
      DECL_DIM_PTRS(pdd);
      SET_DIM_PTRS(tdd, td, i);
      lb[i] = *va_arg(va, __INT_T *);
      extent_diff = lb[i] - F90_DPTR_LBOUND_G(tdd);
      if (F90_RANK_G(td) >= (i + 1))
        ub[i] = DPTR_UBOUND_G(tdd) + extent_diff;
      else
        __fort_abort("PTR_SHAPE_ASSNX: invalid assumed upper bound for pointer");
      if (lb[i] > ub[i])
        stride[i] = -1;
      else
        stride[i] = 1;
      if (!reshape) {
        dimflags |= 1 << i;
        /*continue;*/
      }
      SET_DIM_PTRS(pdd, pd, i);
      F90_DPTR_LBOUND_P(pdd, lb[i]);
      DPTR_UBOUND_P(pdd, ub[i]);
      if (i == 0) {
        tstride = F90_DPTR_LSTRIDE_G(tdd);
        lbase = F90_LBASE_G(pd);
        lbase += (stride[i] * (F90_DPTR_LBOUND_G(tdd) - lb[i]) * tstride);
        F90_DPTR_LSTRIDE_P(pdd, stride[i] * tstride);
      } else {
        if (F90_RANK_G(td) == sz)
          tstride = F90_DPTR_LSTRIDE_G(tdd);
        lbase += (stride[i] * (F90_DPTR_LBOUND_G(tdd) - lb[i]) * tstride);
        F90_DPTR_LSTRIDE_P(pdd, stride[i] * tstride);
      }
      F90_LBASE_P(pd, lbase);
    }

    va_end(va);
    res = tb;
    dimflags |= __NOREINDEX;
    if (reshape) {
      F90_RANK_P(pd, sz);
      /*F90_LBASE_P(pd,lbase);*/
    } else {
    }
    if (lb)
      __fort_free(lb);
    if (ub)
      __fort_free(ub);
  } else {
    __fort_abort("PTR_SHAPE_ASSNX: invalid rank");
  }

  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION) ||
      (targetlen && F90_LEN_G(pd) != *targetlen)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  if (targettype && *targettype)
    F90_KIND_P(pd, *targettype);

  if (notSet)
    I8(__fort_finish_descriptor)(pd);

  if (new_td)
    __fort_free(new_td);

  return res;
}

void *
ENTFTN(PTR_SHAPE_ASSN, ptr_shape_assn)
         (char *pb, F90_Desc *pd, char *tb, F90_Desc *td, __INT_T *sectflag,
          __INT_T *targetlen, __INT_T *targettype, __INT_T *rank, ...)
/* {int lb, int ub}* */
{
  void *res = 0;
  va_list va;
  F90_Desc *new_td = 0;
  __INT_T dimflags = 0, stride[MAXDIMS], *lb = 0, *ub = 0;

  if (pd == NULL || td == NULL) {
    __fort_abort("PTR_SHAPE_ASSN: invalid descriptor");
  }

  if (rank && *rank) {
    int i, reshape;
    __INT_T lbase = 0, tstride, old_lbase;
    int sz = *rank;

    reshape = (sz != F90_RANK_G(td));

    if (reshape && F90_RANK_G(td) != 1) {
      __fort_abort("PTR_SHAPE_ASSN: pointer target must have a rank of 1"
                  " when pointer rank does not equal target rank");
    }

    if (pd == td) {
      new_td = __fort_malloc(SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
      if (!new_td)
        __fort_abort("PTR_SHAPE_ASSN: out of memory");
      __fort_bcopy((char *)new_td, (char *)td, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
      SET_F90_DIST_DESC_PTR(new_td, F90_RANK_G(new_td));
      /* check for align-target to self */
      if (DIST_ALIGN_TARGET_G(td) == td) {
        DIST_ALIGN_TARGET_P(new_td, new_td);
      }
      td = new_td;
    } else {
      __fort_bcopy((char *)pd, (char *)td, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(td)));
      SET_F90_DIST_DESC_PTR(pd, sz);
      /* check for align-target to self */
      if (DIST_ALIGN_TARGET_G(pd) == pd) {
        DIST_ALIGN_TARGET_P(pd, pd);
      }
      F90_RANK_P(pd, sz);
    }
    lb = __fort_malloc(sizeof(__INT_T) * sz);
    ub = __fort_malloc(sizeof(__INT_T) * sz);
    if (!lb || !ub) {
      __fort_abort("PTR_SHAPE_ASSN: out of memory");
    }
    va_start(va, rank);

    for (i = 0; i < sz; ++i) {
      DECL_DIM_PTRS(tdd);
      DECL_DIM_PTRS(pdd);
      if (i < F90_RANK_G(td)) {
        SET_DIM_PTRS(tdd, td, i);
      }
      lb[i] = *va_arg(va, __INT_T *);
      ub[i] = *va_arg(va, __INT_T *);
      if (lb[i] > ub[i])
        stride[i] = -1;
      else
        stride[i] = 1;
      if (!reshape) {
        dimflags |= 1 << i;
      }

      SET_DIM_PTRS(pdd, pd, i);
      F90_DPTR_LBOUND_P(pdd, lb[i]);
      DPTR_UBOUND_P(pdd, ub[i]);
      if (i == 0) {
        tstride = F90_DPTR_LSTRIDE_G(tdd);
        lbase = F90_LBASE_G(td);
        old_lbase = lbase;
        lbase += (stride[i] * ((F90_DPTR_LBOUND_G(tdd) - lb[i])) * tstride);
        F90_DPTR_LSTRIDE_P(pdd, stride[i] * tstride);
      } else {
        old_lbase = lbase;
        if (F90_RANK_G(td) == sz)
          tstride = F90_DPTR_LSTRIDE_G(tdd);
        else
          tstride *= 1 + (ub[i - 1] - lb[i - 1]);

        if (stride[i] < 0) {
          lbase +=
              (stride[i] * (1 + (F90_DPTR_LBOUND_G(tdd) - lb[i])) * tstride) -
              ub[i - 1];
          if (F90_RANK_G(td) != sz)
            lbase -= (1 - F90_DPTR_LBOUND_G(tdd)) + tstride;
        } else {
          lbase += (stride[i] * (F90_DPTR_LBOUND_G(tdd) - lb[i]) * tstride);
          F90_DPTR_LSTRIDE_P(pdd, stride[i] * tstride);
        }

        F90_DPTR_LSTRIDE_P(pdd, stride[i] * tstride);
      }
    }

    if (old_lbase != lbase && F90_LBASE_G(pd) == 0) {
      lbase = 1;
      for (i = 0; i < sz; ++i) {
        DECL_DIM_PTRS(pdd);
        SET_DIM_PTRS(pdd, pd, i);
        lbase += -lb[i] * (F90_DPTR_LSTRIDE_G(pdd));
      }
    } else if (sz > 1) {
      DECL_DIM_PTRS(tdd);
      SET_DIM_PTRS(tdd, td, 0);
      lbase = F90_LBASE_G(pd) +
              stride[0] * ((F90_DPTR_LBOUND_G(tdd) - lb[0])) *
                  F90_DPTR_LSTRIDE_G(tdd);
      for (i = 1; i < sz; ++i) {
        DECL_DIM_PTRS(pdd);
        SET_DIM_PTRS(pdd, pd, i);
        lbase += -lb[i] * (F90_DPTR_LSTRIDE_G(pdd));
      }
    }

    va_end(va);
    res = tb;
    dimflags |= __NOREINDEX;

    if (reshape) {
      F90_RANK_P(pd, sz);
      F90_LBASE_P(pd, lbase);
    } else {
      old_lbase = F90_LBASE_G(pd);
      switch (sz) {
      case 1:
        ENTFTN(SECT, sect)(pd, td, &lb[0], &ub[0], &stride[0], &dimflags);
        break;
      case 2:
        ENTFTN(SECT,sect)(pd,td,&lb[0],&ub[0],&stride[0],
                          &lb[1],&ub[1],&stride[1],&dimflags);
        break;
      case 3:
        ENTFTN(SECT,sect)(pd,td,&lb[0],&ub[0],&stride[0],
                          &lb[1],&ub[1],&stride[1],
			  &lb[2],&ub[2],&stride[2],&dimflags);
        break;
      case 4:
        ENTFTN(SECT,sect)(pd,td,&lb[0],&ub[0],&stride[0],
                          &lb[1],&ub[1],&stride[1],
                          &lb[2],&ub[2],&stride[2],
                          &lb[3],&ub[3],&stride[3],&dimflags);
        break;
      case 5:
        ENTFTN(SECT,sect)(pd,td,&lb[0],&ub[0],&stride[0],
                          &lb[1],&ub[1],&stride[1],
                          &lb[2],&ub[2],&stride[2],
                          &lb[3],&ub[3],&stride[3],
			  &lb[4],&ub[4],&stride[4],&dimflags);
        break;
      case 6:
        ENTFTN(SECT,sect)(pd,td,&lb[0],&ub[0],&stride[0],
                          &lb[1],&ub[1],&stride[1],
                          &lb[2],&ub[2],&stride[2],
                          &lb[3],&ub[3],&stride[3],
                          &lb[4],&ub[4],&stride[4],
                          &lb[5],&ub[5],&stride[5],&dimflags);
        break;
      case 7:
        ENTFTN(SECT,sect)(pd,td,&lb[0],&ub[0],&stride[0],
                          &lb[1],&ub[1],&stride[1],
                          &lb[2],&ub[2],&stride[2],
                          &lb[3],&ub[3],&stride[3],
                          &lb[4],&ub[4],&stride[4],
                          &lb[5],&ub[5],&stride[5],
                          &lb[6],&ub[6],&stride[6],&dimflags);
        break;
      default:
        __fort_abort("PTR_SHAPE_ASSN: invalid rank");
      }
      if (old_lbase == F90_LBASE_G(pd))
        F90_LBASE_P(pd, lbase);
    }
    if (lb)
      __fort_free(lb);
    if (ub)
      __fort_free(ub);

  } else {
    __fort_abort("PTR_SHAPE_ASSN: invalid rank");
  }

  if (!(F90_FLAGS_G(td) & __SEQUENTIAL_SECTION) ||
      (targetlen && F90_LEN_G(pd) != *targetlen)) {
    F90_FLAGS_P(pd, (F90_FLAGS_G(pd) & ~__SEQUENTIAL_SECTION));
  }
  if (targettype && *targettype)
    F90_KIND_P(pd, *targettype);

  if (new_td)
    __fort_free(new_td);

  return res;
}

void
ENTFTN(PTR_OUT, ptr_out)(char *ab, F90_Desc *ad, char *db, F90_Desc *dd)
{
  dtype kind;
  __CLEN_T len;

  if (!ISPRESENT(ab))
    return; /* actual arg is absent */

  if (!ISPRESENT(db))
    __fort_abort("PTR_OUT: unexcused dummy absence");

  if (F90_TAG_G(dd) == __DESC) {
    kind = (dtype)F90_KIND_G(dd);
    len = F90_LEN_G(dd);
  } else if (ISSCALAR(dd)) {
    kind = (dtype)F90_TAG_G(dd);
    len = GET_DIST_SIZE_OF(kind);
  } else {
    kind = __NONE;
    len = 0;
  }
  I8(ptr_out)(ab, ad, db, dd, kind, len);
}

void
ENTFTN(PTR_OUT_CHARA, ptr_out_chara)(DCHAR(ab), F90_Desc *ad, DCHAR(db),
                                   F90_Desc *dd DCLEN64(ab) DCLEN64(db))
{

  if (!ISPRESENTC(ab))
    return; /* actual arg is absent */

  if (!ISPRESENTC(db))
    __fort_abort("PTR_OUT: unexcused dummy absence");

  I8(ptr_out)(CADR(ab), ad, CADR(db), dd, __STR, CLEN(db));
}
/* 32 bit CLEN version */
void
ENTFTN(PTR_OUT_CHAR, ptr_out_char)(DCHAR(ab), F90_Desc *ad, DCHAR(db),
                                   F90_Desc *dd DCLEN(ab) DCLEN(db))
{
  ENTFTN(PTR_OUT_CHARA, ptr_out_chara)(CADR(ab), ad, CADR(db), dd,
                             (__CLEN_T)CLEN(ab), (__CLEN_T)CLEN(db));
}

/* If target is present, return .true. if pointer is associated with
   target.  otherwise, return .true. if pointer is associated. */

static int
I8(__fort_associated)(char *pb, F90_Desc *pd, char *tb, F90_Desc *td,
                     int target_present)
{
  DECL_DIM_PTRS(pdd);
  DECL_DIM_PTRS(tdd);
  char *adr;
  __INT_T i, pextent, poff, textent, toff;

      /* FS#20453 - disable FS#17427 patch below. It appears to no longer be
       * needed and this fixes UMRs in oop567 - oop570 f90_correct tests.
       */
  if (F90_TAG_G(pd) == __NONE)
    return 0;
  adr = pb;
  if (adr == NULL)
    return 0;
  if (target_present) {
    if (adr != tb)
      return 0;
    if (F90_TAG_G(pd) == __DESC) {
      if (F90_TAG_G(td) != __DESC || F90_RANK_G(pd) != F90_RANK_G(td) ||
          F90_KIND_G(pd) != F90_KIND_G(td) || F90_LEN_G(pd) != F90_LEN_G(td))
        return 0;
      poff = F90_LBASE_G(pd) - 1;
      toff = F90_LBASE_G(td) - 1;
      for (i = 0; i < F90_RANK_G(pd); ++i) {
        SET_DIM_PTRS(tdd, td, i);
        SET_DIM_PTRS(pdd, pd, i);
        pextent = F90_DPTR_EXTENT_G(pdd);
        if (pextent < 0)
          pextent = 0;
        textent = F90_DPTR_EXTENT_G(tdd);
        if (textent < 0)
          textent = 0;
        if (textent != pextent)
          return 0;
        poff += (F90_DPTR_LBOUND_G(pdd) * F90_DPTR_SSTRIDE_G(pdd) +
                 F90_DPTR_SOFFSET_G(pdd)) *
                F90_DPTR_LSTRIDE_G(pdd);
        toff += (F90_DPTR_LBOUND_G(tdd) * F90_DPTR_SSTRIDE_G(tdd) +
                 F90_DPTR_SOFFSET_G(tdd)) *
                F90_DPTR_LSTRIDE_G(tdd);
      }
      if (poff != toff)
        return 0;
    } else if (ISSCALAR(pd)) {
      if (F90_TAG_G(pd) != F90_TAG_G(td))
        return 0;
    } else
      __fort_abort("ASSOCIATED: invalid pointer descriptor");
  }
  return 1;
}

__LOG_T
ENTFTN(ASSOCIATED, associated)
         (char *pb, F90_Desc *pd, char *tb, F90_Desc *td)
{
  return I8(__fort_associated)(pb, pd, tb, td, ISPRESENT(tb))
             ? GET_DIST_TRUE_LOG
             : 0;
}

__LOG_T
ENTFTN(ASSOCIATED_T, associated_t)
         (char *pb, F90_Desc *pd, char *tb, F90_Desc *td)
{
  /*  is associated with target ??? */
  return I8(__fort_associated)(pb, pd, tb, td, 1) ? GET_DIST_TRUE_LOG : 0;
}

__LOG_T
ENTFTN(ASSOCIATED_CHARA, associated_chara)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td DCLEN64(pb) DCLEN64(tb))
{
  return I8(__fort_associated)(CADR(pb), pd, CADR(tb), td, ISPRESENTC(tb))
             ? GET_DIST_TRUE_LOG
             : 0;
}
/* 32 bit CLEN version */
__LOG_T
ENTFTN(ASSOCIATED_CHAR, associated_char)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td DCLEN(pb) DCLEN(tb))
{
  return ENTFTN(ASSOCIATED_CHARA, associated_chara) (CADR(pb), pd, CADR(tb), td,
                                        (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

__LOG_T
ENTFTN(ASSOCIATED_TCHARA, associated_tchara)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td DCLEN64(pb) DCLEN64(tb))
{
  /*  is associated with character target ??? */
  return I8(__fort_associated)(CADR(pb), pd, CADR(tb), td, 1)
             ? GET_DIST_TRUE_LOG
             : 0;
}
/* 32 bit CLEN version */
__LOG_T
ENTFTN(ASSOCIATED_TCHAR, associated_tchar)
         (DCHAR(pb), F90_Desc *pd, DCHAR(tb), F90_Desc *td DCLEN(pb) DCLEN(tb))
{
  return ENTFTN(ASSOCIATED_TCHARA, associated_tchara)(CADR(pb), pd, CADR(tb),
                                  td, (__CLEN_T)CLEN(pb), (__CLEN_T)CLEN(tb));
}

#ifndef DESC_I8

void
ENTF90(SUBCHK, subchk)(int sub, int lwb, int upb, int dim, int lineno,
                       char *arrnam, char *srcfil)
/* sub:	value of subscript */
/* lwb:	lower bound of dimension */
/* upb:	upper bound of dimension */
/* dim:	dimension */
/* lineno:	source line number */
/* *arrnam:	name of array, null-terminated string */
/* *srcfil:	name of source file, null-terminated string */
{
  static char str[200];
#ifdef DEBUG
  if (__fort_test & DEBUG_CHECK) {
    printf("%d %s-%s, sub:%d, lwb:%d, upb:%d, dim:%d, lin:%d\n", GET_DIST_LCPU,
           srcfil, arrnam, sub, lwb, upb, dim, lineno);
  }
#endif
  if (lwb > upb) {
    /* zero-sized array */
    if (sub == lwb)
      return;
  }
  if (sub < lwb || sub > upb) {
    sprintf(str,
            "Subscript out of range for array %s (%s: %d)\n"
            "    subscript=%d, lower bound=%d, upper bound=%d, dimension=%d",
            arrnam, srcfil, lineno, sub, lwb, upb, dim);
    __fort_abort(str);
  }
}

void
ENTF90(SUBCHK64, subchk64)(long sub, long lwb, long upb, int dim, int lineno,
                           char *arrnam, char *srcfil)
/* sub:	value of subscript */
/* lwb:	lower bound of dimension */
/* upb:	upper bound of dimension */
/* dim:	dimension */
/* lineno:	source line number */
/* *arrnam:	name of array, null-terminated string */
/* *srcfil:	name of source file, null-terminated string */
{
  static char str[200];
#ifdef DEBUG
  if (__fort_test & DEBUG_CHECK) {
    printf("%d %s-%s, sub:%ld, lwb:%ld, upb:%ld, dim:%d, lin:%d\n",
           GET_DIST_LCPU, srcfil, arrnam, sub, lwb, upb, dim, lineno);
  }
#endif
  if (lwb > upb) {
    /* zero-sized array */
    if (sub == lwb)
      return;
  }
  if (sub < lwb || sub > upb) {
    sprintf(str,
            "Subscript out of range for array %s (%s: %d)\n"
            "    subscript=%ld, lower bound=%ld, upper bound=%ld, dimension=%d",
            arrnam, srcfil, lineno, sub, lwb, upb, dim);
    __fort_abort(str);
  }
}

void
ENTF90(PTRCHK, ptrchk)(int ptr, int lineno, char *ptrnam, char *srcfil)
/* ptr:	value of pointer */
/* lineno:	source line number */
/* *ptrnam:	name of pointer, null-terminated string */
/* *srcfil:	name of source file, null-terminated string */
{
  static char str[200];
#ifdef DEBUG
  if (__fort_test & DEBUG_CHECK) {
    printf("%d %s-%s, ptr:%d, lin:%d\n", GET_DIST_LCPU, srcfil, ptrnam, ptr,
           lineno);
  }
#endif
  if (ptr == 0) {
    sprintf(str, "Null pointer for %s (%s: %d)\n", ptrnam, srcfil, lineno);
    __fort_abort(str);
  }
}

void
ENTF90(PTRCP, ptrcp)(void *to, void *from)
{
  ((int *)to)[0] = ((int *)from)[0];
  ((int *)to)[1] = ((int *)from)[1];
}

#endif

void
ENTF90(MOVE_ALLOC, move_alloc)(void **fp, F90_Desc *fd, void **tp, F90_Desc *td)
{
  if (fd == NULL || td == NULL) {
    __fort_abort("MOVE_ALLOC: invalid descriptor");
  }

  if (fd != td && F90_TAG_G(fd) == __DESC) {
    __fort_bcopy((char *)td, (char *)fd, SIZE_OF_RANK_n_ARRAY_DESC(F90_RANK_G(fd)));
    SET_F90_DIST_DESC_PTR(td, F90_RANK_G(td));
    /* check for align-target to self */
    if (DIST_ALIGN_TARGET_G(td) == td) {
      DIST_ALIGN_TARGET_P(td, td);
    }
  } else if (fd != td) {
    ENTF90(SET_TYPE, set_type)(td, (OBJECT_DESC *)fd);
  }

  if (*fp && !I8(__fort_allocated)((char *)*fp)) {
    *tp = NULL;
  } else {
    *tp = *fp;
  }
  *fp = NULL;
}

void
ENTF90(C_F_PTR, c_f_ptr)(void **cptr, __INT_T *rank, __INT_T *sz, void **fb,
                         F90_Desc *fd, __INT_T *ft, void *shp, __INT_T *shpt)
{
  __INT_T one;
  __INT_T ub[7];
  __INT_T flags;
  int i, n;

  *fb = *cptr;
  n = *rank;
  if (!n)
    return;
  switch (*shpt) {
  case __INT1:
  case __LOG1:
    for (i = 0; i < n; i++) {
      ub[i] = (__INT_T)(((__INT1_T *)shp)[i]);
    }
    break;
  case __INT2:
  case __LOG2:
    for (i = 0; i < n; i++) {
      ub[i] = (__INT_T)(((__INT2_T *)shp)[i]);
    }
    break;
  case __INT4:
  case __LOG4:
    for (i = 0; i < n; i++) {
      ub[i] = (__INT_T)(((__INT4_T *)shp)[i]);
    }
    break;
  case __INT8:
  case __LOG8:
    for (i = 0; i < n; i++) {
      ub[i] = (__INT_T)(((__INT8_T *)shp)[i]);
    }
    break;
  default:
    __fort_abort("C_F_POINTER: shape array error");
  }
  one = 1;
  flags = 0;
  switch (n) {
  case 1:
    ENTF90(TEMPLATE1, template1)(fd, &flags, ft, sz, &one, &ub[0]);
    break;
  case 2:
    ENTF90(TEMPLATE2,template2)(fd, &flags, ft, sz,
	    &one, &ub[0], &one, &ub[1]);
    break;
  case 3:
    ENTF90(TEMPLATE3,template3)(fd, &flags, ft, sz,
	    &one, &ub[0], &one, &ub[1], &one, &ub[2]);
    break;
  case 4:
    ENTF90(TEMPLATE,template)(fd, rank, &flags, ft, sz,
	    &one, &ub[0], &one, &ub[1], &one, &ub[2], &one, &ub[3]);
    break;
  case 5:
    ENTF90(TEMPLATE,template)(fd, rank, &flags, ft, sz,
	    &one, &ub[0], &one, &ub[1], &one, &ub[2], &one, &ub[3],
	    &one, &ub[4]);
    break;
  case 6:
    ENTF90(TEMPLATE,template)(fd, rank, &flags, ft, sz,
	    &one, &ub[0], &one, &ub[1], &one, &ub[2], &one, &ub[3],
	    &one, &ub[4], &one, &ub[5]);
    break;
  case 7:
    ENTF90(TEMPLATE,template)(fd, rank, &flags, ft, sz,
	    &one, &ub[0], &one, &ub[1], &one, &ub[2], &one, &ub[3],
	    &one, &ub[4], &one, &ub[5], &one, &ub[6] );
    break;
  }
}

#ifndef DESC_I8
void
ENTF90(C_F_PROCPTR, c_f_procptr)(void **cptr, void **fb)
{

  *fb = *cptr;
}
#endif

