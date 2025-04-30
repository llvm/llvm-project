/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Data redistribution routines
 */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "utils3f.h" /* for __cstr_free */

void
ENTFTN(TEMPLATE, template)(F90_Desc *dd, __INT_T *p_rank,
       __INT_T *p_flags, ...);

#include <string.h>
#include "fort_vars.h"
#if (defined(TARGET_LINUX_X8664) || defined (TARGET_LINUX_POWER) || defined(TARGET_OSX_X8664)   || defined(TARGET_LINUX_ARM32)  || defined(TARGET_LINUX_ARM64)) && !defined(TARGET_WIN)
#include <unistd.h>
#include <sys/wait.h>
#endif
static void store_int_kind(void *, __INT_T *, int);
static void ftn_msgcpy(char*, const char*, int);
#if defined(DEBUG)
static const char *intents[] = {"INOUT", "IN", "OUT", "??"};
#endif

/** \brief Compare alignments and local storage sequences.  Return true if all
   elements of s1 are ultimately aligned to elements of s2 that reside
   on the same processors in the same storage order.

   Descriptors always have legitimate align-target pointers.

   Sections s1 and s2 are stored alike if:

      storage strides are identical,

      extents are identical,

      both are equivalently mapped onto processor arrangements, and

      overlap allowances are identical.

   Corresponding elements of sections s1 and s2 have identical offsets.

   The sections may have different types, lengths, base addresses, and
   linearized index base offsets.

*/

int
I8(__fort_stored_alike)(F90_Desc *dd, F90_Desc *sd)
{
  DECL_DIM_PTRS(ddd);
  DECL_DIM_PTRS(sdd);
  __INT_T dim, bit;

  if (dd == sd)
    return 1;

  if (dd == NULL || sd == NULL || F90_TAG_G(dd) != F90_TAG_G(sd))
    return 0;

  if (F90_TAG_G(dd) != __DESC)
    return 1;

  if (F90_RANK_G(dd) != F90_RANK_G(sd) || F90_GSIZE_G(dd) != F90_GSIZE_G(sd) ||
      (F90_FLAGS_G(dd) | F90_FLAGS_G(sd)) & __OFF_TEMPLATE)
    return 0; /* different rank, size, or not local */

  for (dim = F90_RANK_G(dd); --dim >= 0;) {

    SET_DIM_PTRS(ddd, dd, dim);
    SET_DIM_PTRS(sdd, sd, dim);

    if (F90_DPTR_EXTENT_G(ddd) != F90_DPTR_EXTENT_G(sdd))
      return 0; /* different global extents */

    if (F90_DPTR_SSTRIDE_G(ddd) != F90_DPTR_SSTRIDE_G(sdd))
      return 0; /* different section stride */

    if (F90_DPTR_SOFFSET_G(ddd) != F90_DPTR_SOFFSET_G(sdd))
      return 0; /* different section offsets */

    if (F90_DPTR_LSTRIDE_G(ddd) != F90_DPTR_LSTRIDE_G(sdd))
      return 0; /* different local strides */

    if (DIST_DPTR_LOFFSET_G(ddd) != DIST_DPTR_LOFFSET_G(sdd))
      return 0; /* different local offsets */

    if (DIST_DPTR_NO_G(ddd) != DIST_DPTR_NO_G(sdd) ||
        DIST_DPTR_PO_G(ddd) != DIST_DPTR_PO_G(sdd))
      return 0; /* different overlaps */

    bit = 1 << dim;

    if (DIST_MAPPED_G(dd) & DIST_MAPPED_G(sd) & bit) {

      /* both partitioned... */

      if (DIST_ALIGN_TARGET_G(dd) == DIST_ALIGN_TARGET_G(sd)) {

        /* same align-target... */

        if (DIST_DPTR_TAXIS_G(ddd) != DIST_DPTR_TAXIS_G(sdd))
          return 0; /* different template axes */

        if (F90_DPTR_LBOUND_G(ddd) * DIST_DPTR_TSTRIDE_G(ddd) +
                DIST_DPTR_TOFFSET_G(ddd) !=
            F90_DPTR_LBOUND_G(sdd) * DIST_DPTR_TSTRIDE_G(sdd) +
                DIST_DPTR_TOFFSET_G(sdd))
          return 0; /* different template origins */

        if (DPTR_UBOUND_G(ddd) > F90_DPTR_LBOUND_G(ddd) &&
            DIST_DPTR_TSTRIDE_G(ddd) != DIST_DPTR_TSTRIDE_G(sdd))
          return 0; /* different template strides */
      } else {

        /* different align-target descriptors... */

        if (DIST_DPTR_PSHAPE_G(ddd) != DIST_DPTR_PSHAPE_G(sdd) ||
            DIST_DPTR_PSTRIDE_G(ddd) != DIST_DPTR_PSTRIDE_G(sdd))
          return 0; /* different processor arrangements */

        if (DIST_DPTR_BLOCK_G(sdd) * DIST_DPTR_TSTRIDE_G(ddd) !=
            DIST_DPTR_BLOCK_G(ddd) * DIST_DPTR_TSTRIDE_G(sdd))
          return 0;

        if (DIST_DPTR_BLOCK_G(sdd) *
                (F90_DPTR_LBOUND_G(ddd) * DIST_DPTR_TSTRIDE_G(ddd) +
                 DIST_DPTR_TOFFSET_G(ddd) - DIST_DPTR_TLB_G(ddd)) !=
            DIST_DPTR_BLOCK_G(ddd) *
                (F90_DPTR_LBOUND_G(sdd) * DIST_DPTR_TSTRIDE_G(sdd) +
                 DIST_DPTR_TOFFSET_G(sdd) - DIST_DPTR_TLB_G(sdd)))
          return 0;
      }
    } else if ((DIST_MAPPED_G(dd) | DIST_MAPPED_G(sd)) & bit)
      return 0; /* one partitioned and one not */
  }
  return 1;
}


typedef enum { __COPY_IN, __COPY_OUT } copy_dir;

/** \brief copy local data between global actual array and local dummy array */
static void
I8(local_copy)(char *db, F90_Desc *dd, __INT_T doffset, char *ab,
                           F90_Desc *ad, __INT_T aoffset, __INT_T dim,
                           copy_dir dir)
{
  DECL_DIM_PTRS(add);
  DECL_DIM_PTRS(ddd);
  __INT_T aoff, astr, cl, clof, cn, dstr, doff, l, alen, dlen, n, u;
  char *aptr, *dptr;

  SET_DIM_PTRS(ddd, dd, dim - 1);
  SET_DIM_PTRS(add, ad, dim - 1);

  astr = F90_DPTR_SSTRIDE_G(add) * F90_DPTR_LSTRIDE_G(add);
  dstr = F90_DPTR_LSTRIDE_G(ddd);
  doff = doffset + F90_DPTR_LBOUND_G(ddd) * dstr;

  alen = F90_LEN_G(ad);
  dlen = F90_LEN_G(dd);
  cl = DIST_DPTR_CL_G(add);
  clof = DIST_DPTR_CLOF_G(add);
  for (cn = DIST_DPTR_CN_G(add); --cn >= 0;) {
    n = I8(__fort_block_bounds)(ad, dim, cl, &l, &u);
    aoff = aoffset +
           (l * F90_DPTR_SSTRIDE_G(add) + F90_DPTR_SOFFSET_G(add) - clof) *
               F90_DPTR_LSTRIDE_G(add);
    if (dim > 1) {
      for (; n > 0; --n) {
        I8(local_copy)(db, dd, doff, ab, ad, aoff, dim - 1, dir);
        aoff += astr;
        doff += dstr;
      }
    } else if (n > 0) {
      aptr = ab + aoff * alen;
      dptr = db + doff * dlen;
      if (alen == dlen) {
        if (dir == __COPY_IN) {
          __fort_bcopysl(dptr, aptr, n, dstr, astr, alen);
        } else {
          __fort_bcopysl(aptr, dptr, n, astr, dstr, alen);
        }
      } else {
        __INT_T nn, dskip, askip;
        char *dptrx, *aptrx;
        dptrx = dptr;
        aptrx = aptr;
        dskip = dstr * dlen;
        askip = astr * alen;
        if (dir == __COPY_IN) {
          for (nn = 0; nn < n; ++nn) {
            __fort_bcopysl(dptrx, aptrx, 1, dstr, astr, dlen);
            dptrx += dskip;
            aptrx += askip;
          }
        } else {
          for (nn = 0; nn < n; ++nn) {
            __fort_bcopysl(aptrx, dptrx, 1, astr, dstr, dlen);
            dptrx += dskip;
            aptrx += askip;
          }
        }
      }
      doff += n * dstr;
    }
    cl += DIST_DPTR_CS_G(add);
    clof += DIST_DPTR_CLOS_G(add);
  }
}

/** \brief check if a descriptor is associated with a non-contiguous
 *         section.
 *
 * \param a is the descriptor we are checking.
 * \param dim is the rank of the array we are checking.
 *
 * \returns 0 if contiguous, else the dimension that is non-contiguous.
 */
__INT_T
I8(is_nonsequential_section)(F90_Desc *a, __INT_T dim)
{
  __INT_T is_nonseq_section;
  __INT_T tmp_lstride;
  int i;
  DECL_DIM_PTRS(ad);

  is_nonseq_section = 0;
  tmp_lstride = 1;
  for (i = 0; i < dim; i++) {
    SET_DIM_PTRS(ad, a, i);
    if (F90_DPTR_LSTRIDE_G(ad) != tmp_lstride || F90_DPTR_SSTRIDE_G(ad) != 1) {
      is_nonseq_section = i + 1;
      break;
    }
    tmp_lstride *= F90_DPTR_EXTENT_G(ad);
  }

  return is_nonseq_section;
}

/* copy_in actions.  REDESCRIBE means the descriptor cannot be blindly
   copied.  anything else means the data needs to be copied. */

#define REDESCRIBE 0x001
#define RETARGET 0x002
#define REFORMAT 0x004
#define REALIGN 0x008
#define RELOCATE 0x010
#define SQUEEZE 0x020
#define SHRINK 0x040
#define EXPAND 0x080
#define RECOPY 0x100

#ifdef FLANG_RDST_UNUSED
static void
invalid_flags(int flags)
{
#if defined(DEBUG)
  __fort_show_flags(flags);
  printf("\n");
#endif
  __fort_abort("COPY_IN: internal error, invalid flags");
}
#endif

static void
copy_in_abort(const char *msg)
{
  char str[120];
  sprintf(str, "COPY_IN: %s", msg);
  __fort_abort(str);
}

/* Varargs: [ [ proc *dist_target, ]
              __INT_T *isstar,
              { [__INT_T paxis, ] __INT_T *dstfmt, | (__INT_T * array,
             __INT_T * extent) }* ]
            [ F90_Desc *align_target, __INT_T *conform,
              [ __INT_T *collapse,
                { __INT_T *taxis, __INT_T *tstride, __INT_T *toffset, }*
                __INT_T *single,
                { __INT_T *coordinate, }* ] ]
            { __INT_T *lbound, [ __INT_T *ubound, ] }*
            { [ __INT_T *no, __INT_T *po, ] }*
 */
void
ENTFTN(QOPY_IN, qopy_in)(char **dptr, __POINT_T *doff, char *dbase,
                         F90_Desc *dd, char *ab, F90_Desc *ad,
                         __INT_T *p_rank, __INT_T *p_kind, __INT_T *p_len,
                         __INT_T *p_flags, ...)
{
  va_list va;
  DECL_HDR_PTRS(au); /* (a)ctual, (d)ummy, (t)arget, (u)ltimate */
  DECL_DIM_PTRS(add);
  DECL_DIM_PTRS(ddd);
  char *db, *df;

  dtype kind;
  __INT_T conform, collapse, single;
  __INT_T rank, len;
  __INT_T flags, isstar = 0;
  __INT_T lbound[MAXDIMS], ubound[MAXDIMS];
  __INT_T idx[MAXDIMS];

  _io_intent intent;
  _io_spec dist_target_spec;
  _io_spec dist_format_spec;
  _io_spec align_target_spec;

  __INT_T i, m, n;
  int action;
  int pointer_present, offset_present;
  __INT_T actual_extent[MAXDIMS], actual_size, aolb;
  __INT_T dummy_extent[MAXDIMS], dummy_size, dolb;
  void *unused;
  __INT_T wrk_rank;

  rank = *p_rank;
  kind = (dtype)*p_kind;
  len = *p_len;
  flags = *p_flags;

  if (LOCAL_MODE)
    flags |= __LOCAL;

  if (flags & __ASSUMED_SIZE)
    flags |= __SEQUENCE;

  intent = (_io_intent)(flags >> __INTENT_SHIFT & __INTENT_MASK);

  dist_target_spec =
      (_io_spec)(flags >> __DIST_TARGET_SHIFT & __DIST_TARGET_MASK);
  dist_format_spec =
      (_io_spec)(flags >> __DIST_FORMAT_SHIFT & __DIST_FORMAT_MASK);
  align_target_spec =
      (_io_spec)(flags >> __ALIGN_TARGET_SHIFT & __ALIGN_TARGET_MASK);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d COPY_IN actual ab=0x%x ad=0x%x\n", GET_DIST_LCPU, ab, ad);
    I8(__fort_describe)(ab, ad);
    printf("%d COPY_IN dummy dptr=0x%x doff=0x%x dbase=0x%x dd=0x%x\n"
           "      %s len=%d rank=%d ",
           GET_DIST_LCPU, dptr, doff, dbase, dd, GET_DIST_TYPENAMES(kind),
           len, rank);
    __fort_show_flags(flags);
    printf("\n");
  }
#endif

  if (!ISPRESENT(ab)) {
    (void)I8(__fort_ptr_offset)(dptr, doff, dbase, kind, len, ab);
    I8(__fort_copy_descriptor)(dd, ad);
    return;
  }

  if (ISSCALAR(ad) || ISSEQUENCE(ad)) {
    if (!(flags & (__SEQUENCE | __NO_OVERLAPS)))
      copy_in_abort("scalar passed to nonsequential dummy");
    if (flags & __ASSUMED_SHAPE)
      copy_in_abort("scalar passed to assumed-shape dummy");

    au = NULL; /* no descriptors */
  } else if (F90_TAG_G(ad) == __DESC) {

    /* actual argument must match nonsequential or assumed-shape
       dummy type, kind, and rank */

    if (flags & __ASSUMED_SHAPE || !(flags & (__SEQUENCE | __NO_OVERLAPS))) {
      if (F90_RANK_G(ad) != rank)
        copy_in_abort("actual argument rank differs from dummy");
      else if (F90_KIND_G(ad) != kind)
        copy_in_abort("actual argument type differs from dummy");
      else if (F90_LEN_G(ad) != len)
        copy_in_abort("actual argument length differs from dummy");
    }

    au = ad;
  } else if (F90_TAG_G(ad) == 0)
    /* tpr2467: allocable never allocated; the descriptor tag has been
     * initialized to 0.
     */
    return;
  else
    copy_in_abort("invalid actual argument descriptor"
                  " (missing or incorrect\ninterface block in caller,"
                  " or wrong number of arguments?)");

  conform = 0;
  collapse = 0;
  single = 0;
  isstar = 0;
  action = 0;

  va_start(va, p_flags);

  if (align_target_spec) {
    /* consume args not used by F90 */
    unused = va_arg(va, F90_Desc *);
    conform = *va_arg(va, __INT_T *);
    if (!(flags & __IDENTITY_MAP)) {
      collapse = *va_arg(va, __INT_T *);
      for (i = 0; i < rank; ++i) {
        if (!(collapse >> i & 1)) {
          unused = va_arg(va, __INT_T *);
          unused = va_arg(va, __INT_T *);
          unused = va_arg(va, __INT_T *);
        }
      }

      /* single alignments */

      single = *va_arg(va, __INT_T *);
      for (i = 0; single >> i; ++i) {
        if (single >> i & 1)
          unused = va_arg(va, __INT_T *);
      }
    }
  } else {
    /* distribution target spec, consume unused args */
    if ((dist_target_spec == __PRESCRIPTIVE) || (dist_target_spec == __DESCRIPTIVE)) {
      unused = va_arg(va, proc *);
    }

    if ((dist_format_spec == __PRESCRIPTIVE) || (dist_format_spec == __DESCRIPTIVE)) {
      isstar = *va_arg(va, __INT_T *);

      for (i = 0; i < rank; ++i) {
        if (isstar >> i & 1) {
          if (flags & __DUMMY_COLLAPSE_PAXIS) {
            unused = va_arg(va, __INT_T *);
            unused = va_arg(va, __INT_T *);
          }
        } else {
          if (flags & __DIST_TARGET_AXIS) {
            unused = va_arg(va, __INT_T *);
          }

          if (!(isstar & EXTENSION_BLOCK_MASK) ||
              !(((isstar & EXTENSION_BLOCK_MASK) >> (7 + 3 * i)) & 0x01)) {
            unused = va_arg(va, __INT_T *);
          } else {
            unused = va_arg(va, __INT_T *);
            if (!(flags & __ASSUMED_GB_EXTENT))
              unused = va_arg(va, __INT_T *);
          }
        }
      }
    }
  }

  /* compute actual extents and length */

  if (F90_TAG_G(ad) != __DESC) {
    /* actual arg is scalar or sequential */

    actual_size = 1;
    for (i = rank; --i >= 0;)
      actual_extent[i] = 1;
  } else {
    /* normal case; actual extents are global extents */

    actual_size = F90_GSIZE_G(ad);
    for (i = F90_RANK_G(ad); --i >= 0;) {
      actual_extent[i] = F90_DIM_EXTENT_G(ad, i);
    }
  }

  /* get array bounds; handle assumed-shape or size */

  if (flags & __ASSUMED_SHAPE) {
    dummy_size = actual_size;
    for (i = 0; i < rank; ++i) {
      lbound[i] = *va_arg(va, __INT_T *);
      ubound[i] = lbound[i] + actual_extent[i] - 1;
      dummy_extent[i] = actual_extent[i];
    }
  } else if (flags & __ASSUMED_SIZE) {
    dummy_size = 1;
    for (i = 1; i < rank; ++i) {
      lbound[i - 1] = *va_arg(va, __INT_T *);
      ubound[i - 1] = *va_arg(va, __INT_T *);
      n = ubound[i - 1] - lbound[i - 1] + 1;
      if (n < 0)
        n = 0;
      dummy_extent[i - 1] = n;
      dummy_size *= n;
    }
    if (dummy_size <= 0)
      n = 0;
    else if (F90_TAG_G(ad) != __DESC)
      n = 1;
    else {
      n = (actual_size * F90_LEN_G(ad)) / (dummy_size * len);
      if (n < 0)
        n = 0;
    }
    lbound[rank - 1] = *va_arg(va, __INT_T *);
    ubound[rank - 1] = lbound[rank - 1] + n - 1;
    dummy_extent[rank - 1] = n;
    dummy_size *= n;
  } else { /* explicit-shape dummy */
    dummy_size = 1;
    for (i = 0; i < rank; ++i) {
      lbound[i] = *va_arg(va, __INT_T *);
      ubound[i] = *va_arg(va, __INT_T *);
      n = ubound[i] - lbound[i] + 1;
      if (n < 0)
        n = 0;
      dummy_extent[i] = n;
      dummy_size *= n;
    }

    if (!(flags & (__SEQUENCE | __NO_OVERLAPS | __LOCAL)) &&
        (dummy_size > actual_size))
      copy_in_abort("argument shape conformance error");
  }

  va_end(va);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("  dim star   px dfmt  cnf  col   tx  str  off"
           " lbnd ubnd   no   po\n");
    for (i = 0; i < rank; ++i) {
      printf("%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d\n", i + 1, 1, 0, 0, 0, 0,
             0, 0, 0, lbound[i], ubound[i], 0, 0);
    }
  }
#endif

  /* determine copy_in action. force copying if index offset
     required and base variables can't be aligned */

  pointer_present = ISPRESENT(dptr);
  offset_present = ISPRESENT(doff);

  if (offset_present && !I8(__fort_ptr_aligned)(dbase, kind, len, ab))
    action |= RELOCATE;

  if (F90_TAG_G(ad) != __DESC) {
    if (action & RELOCATE)
      copy_in_abort("unable to align dummy base with sequential arg");

    /* actual argument is sequential.  make new descriptor */

    action |= REDESCRIBE;
  } else {

    /* make new descriptors if lower bounds are different */

    for (i = rank; --i >= 0;) {
      if (F90_DIM_LBOUND_G(ad, i) != lbound[i] ||
          F90_DIM_SOFFSET_G(ad, i) != 0) {
        action |= REDESCRIBE;
        break;
      }
    }

    /* make new descriptors for assumed-size dummy or if rank
       changes (sequential dummy) */

    if (flags & __ASSUMED_SIZE || rank != F90_RANK_G(ad))
      action |= REDESCRIBE;

    /* squeeze if the actual argument is discontiguous (ignoring
       overlaps) */

    if (I8(is_nonsequential_section)(ad, F90_RANK_G(ad)))
      action |= SQUEEZE;

    /* copy if the local shape is different */

    if (dummy_size > 0 && !(action & ~REDESCRIBE)) {
      m = 1;
      for (i = 0; i < rank; ++i) {
        SET_DIM_PTRS(add, ad, i);
        if (dummy_extent[i] > 1 &&
            (F90_DPTR_SSTRIDE_G(add) * F90_DPTR_LSTRIDE_G(add) != m ||
             dummy_extent[i] != actual_extent[i])) {
#if defined(DEBUG)
          if (__fort_test & DEBUG_RDST)
            printf("%d nonsequential->local squeeze\n", GET_DIST_LCPU);
#endif
          action |= SQUEEZE;
          break;
        }
        m *= dummy_extent[i];
      }
    }
  }

  /* make new descriptors if actual argument isn't ultimately
     aligned to self */

  if (au != ad)
    action |= REDESCRIBE;

  if (action) {
    __DIST_INIT_DESCRIPTOR(dd, rank, kind, len, flags, NULL);
    for (i = 0; i < rank; ++i) {
      SET_DIM_PTRS(ddd, dd, i);
      F90_DPTR_LBOUND_P(ddd, lbound[i]);
      DPTR_UBOUND_P(ddd, ubound[i]);
      F90_DPTR_SSTRIDE_P(ddd, 1); /* section stride */
      F90_DPTR_SOFFSET_P(ddd, 0); /* section offset */
      F90_DPTR_LSTRIDE_P(ddd, 0);
    }

    F90_FLAGS_P(dd, F90_FLAGS_G(dd) & ~__TEMPLATE);
    I8(__fort_finish_descriptor)(dd);
  } else {
    I8(__fort_copy_descriptor)(dd, ad);

    aolb = F90_LBASE_G(dd) - 1;
    for (i = F90_RANK_G(dd); --i >= 0;) {
      SET_DIM_PTRS(ddd, dd, i);
      aolb += F90_DPTR_LSTRIDE_G(ddd) * F90_DPTR_LBOUND_G(ddd);
    }
    *dptr = ab + aolb * F90_LEN_G(dd);
    F90_LBASE_P(dd, F90_LBASE_G(dd) - aolb);

    F90_FLAGS_P(dd, F90_FLAGS_G(dd) | __NOT_COPIED);
    return;
  }

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST && action)
    printf("%d%s%s%s%s%s%s%s%s%s\n", GET_DIST_LCPU,
           action & REDESCRIBE ? " redescribe" : "",
           action & RETARGET ? " retarget" : "",
           action & REFORMAT ? " reformat" : "",
           action & REALIGN ? " realign" : "",
           action & RELOCATE ? " relocate" : "",
           action & SQUEEZE ? " squeeze" : "", action & SHRINK ? " shrink" : "",
           action & EXPAND ? " expand" : "", action & RECOPY ? " recopy" : "");
#endif

  if (action & ~REDESCRIBE) {

    /* copying needed for all actions except REDESCRIBE */

    DECL_HDR_VARS(c);
    DECL_HDR_PTRS(cd) = dd; /* descriptor to use for copy */

    if (!(pointer_present | offset_present))
      copy_in_abort("cannot copy actual arg to sequential dummy");

    /* allocate space and copy the array */

    F90_FLAGS_P(dd, F90_FLAGS_G(dd) & ~__NOT_COPIED);

    if (flags & (__SEQUENCE | __NO_OVERLAPS)) {

      /* sequential dummy may not have same shape as the actual
         argument, in which case we need to create a descriptor
         with the actual argument shape just to do copy */

      if (F90_RANK_G(ad) != rank)
        cd = c;
      else {
        for (i = rank; --i >= 0;) {
          if (actual_extent[i] != dummy_extent[i]) {
            cd = c;
            break;
          }
        }
      }
      if (cd != dd) {
        __DIST_INIT_DESCRIPTOR(cd, F90_RANK_G(ad), F90_KIND_G(ad), F90_LEN_G(ad),
                              flags, NULL);
        wrk_rank = F90_RANK_G(ad);
        for (i = 1; i <= wrk_rank; ++i) {
          SET_DIM_PTRS(ddd, dd, i - 1);
          F90_DPTR_LBOUND_P(ddd, 1);
          DPTR_UBOUND_P(ddd, actual_extent[i - 1]);
          F90_DPTR_SSTRIDE_P(ddd, 1); /* section stride */
          F90_DPTR_SOFFSET_P(ddd, 0); /* section offset */
          F90_DPTR_LSTRIDE_P(ddd, 0);
        }
        F90_FLAGS_P(cd, F90_FLAGS_G(cd) & ~__TEMPLATE);
        I8(__fort_finish_descriptor)(cd);
      }
    }

    if (__fort_test & DEBUG_COPYIN_LOCAL)
      __fort_tracecall("local argument copied");

    db = I8(__fort_local_allocate)(F90_LSIZE_G(cd), F90_KIND_G(cd),
                                  F90_LEN_G(cd), dbase, dptr, doff);
    if (intent != __OUT) {
      I8(local_copy)(db, cd, F90_LBASE_G(dd)-1, ab, ad, 
                           F90_LBASE_G(ad)-1, rank, __COPY_IN);
    }
  } else {

/* the array does not need to be copied.  if there is local
   data, then both the dummy and the actual should have the
   same first local element.  the base address and offset of
   the dummy argument needs to be adjusted for unaccounted
   single alignments, scalar subscripts, and overlap allowance
   differences. */

#if defined(DEBUG)
    if (__fort_test & DEBUG_RDST) {
      printf("%d COPY_IN initial ab=0x%x dd=0x%x\n", GET_DIST_LCPU, ab, dd);
    }
#endif

    F90_FLAGS_P(dd, F90_FLAGS_G(dd) | __NOT_COPIED);
    if (F90_TAG_G(ad) == __DESC && ~F90_FLAGS_G(dd) & __OFF_TEMPLATE &&
        F90_GSIZE_G(dd) > 0) {
      for (i = F90_RANK_G(ad); --i >= 0;)
        idx[i] = F90_DIM_LBOUND_G(ad, i);
      aolb = I8(__fort_local_offset)(ad, idx);

      for (i = rank; --i >= 0;) {
        F90_DIM_SOFFSET_P(dd, i, 0);
        idx[i] = F90_DIM_LBOUND_G(dd, i);
      }
      dolb = I8(__fort_local_offset)(dd, idx);

      if (flags & (__SEQUENCE | __NO_OVERLAPS)) {

        /* Because a sequential dummy may be passed to an f77
           extrinsic, the dummy address (db) must point
           precisely at the actual first local element and the
           dummy descriptor must produce a zero offset when
           its first element is indexed */

        db = ab + aolb * F90_LEN_G(ad);
        F90_LBASE_P(dd, F90_LBASE_G(dd) - dolb);
      } else { /* nonsequential dummy */
        db = ab;

#if defined(DEBUG)
        if (aolb != dolb) {
          I8(__fort_describe)(db, dd);
          printf("aolb=%d dolb=%d\n", aolb, dolb);
          copy_in_abort("nonsequential arg offsets differ");
        }
#endif
      }
    } else
      db = ab;

    df = I8(__fort_ptr_offset)(dptr, doff, dbase, kind, len, db);
    if (df != db || (df != ab && !(pointer_present | offset_present)))
      copy_in_abort("cannot align dummy base with actual arg");
  }

  if (flags & __NO_OVERLAPS)
    /* coerced to sequential */
    F90_FLAGS_P(dd, F90_FLAGS_G(dd) | __SEQUENCE);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d COPY_IN final db=0x%x dd=0x%x\n", GET_DIST_LCPU, db, dd);
    I8(__fort_describe)(db, dd);
  }
#endif
}

/* copy dummy array back to actual argument */

#if defined(DEBUG)
static void
copy_out_abort(char *msg)
{
  char str[120];
  sprintf(str, "COPY_OUT: %s", msg);
  __fort_abort(str);
}
#endif

void I8(__fort_copy_out)(void *ab,     /**< actual base address */
                        void *db,     /**< dummy base address */
                        F90_Desc *ad, /**< actual descriptor */
                        F90_Desc *dd, /**< dummy descriptor */
                        __INT_T flags)
{
  DECL_HDR_VARS(c);
  DECL_HDR_PTRS(cd); /* descriptor to use for copy */
  __INT_T actual_extent[MAXDIMS], dummy_extent, i, intent;
  __INT_T wrk_rank;

  if (!ISPRESENT(ab))
    return; /* actual arg is absent */

  intent = (flags >> __INTENT_SHIFT & __INTENT_MASK);

  if (dd && F90_TAG_G(dd) == 0)
    /* tpr2467: allocable never allocated; the descriptor tag has been
     * initialized to 0.
     */
    return;

#if defined(DEBUG)
  if (!ISPRESENT(db))
    copy_out_abort("dummy not present");
  if (dd == NULL || F90_TAG_G(dd) != __DESC)
    copy_out_abort("invalid dummy descriptor");
  if (F90_TAG_G(DIST_ALIGN_TARGET_G(dd)) != __DESC)
    copy_out_abort("invalid dummy align-target descriptor");
  if (__fort_test & DEBUG_RDST)
    printf("%d COPY_OUT ab=%x db=%x INTENT(%s)%s\n", GET_DIST_LCPU, ab, db,
           intents[intent],
           F90_FLAGS_G(dd) & __NOT_COPIED ? " NOT COPIED" : "");
#endif

  if (F90_FLAGS_G(dd) & __NOT_COPIED)
    return;

  cd = dd;
  if (F90_FLAGS_G(dd) & (__SEQUENCE | __NO_OVERLAPS)) {

    /* sequential dummy may not have same shape as the actual
       argument, in which case we need to create a descriptor
       with the actual argument shape just to do copy */

    for (i = F90_RANK_G(ad); --i >= 0;) {
      actual_extent[i] = F90_DIM_EXTENT_G(ad, i);
    }

    if (F90_RANK_G(ad) != F90_RANK_G(dd))
      cd = c;
    else {
      for (i = F90_RANK_G(ad); --i >= 0;) {
        dummy_extent = F90_DIM_EXTENT_G(dd, i);
        if (actual_extent[i] != dummy_extent) {
          cd = c;
          break;
        }
      }
    }
    if (cd != dd) {
      __DIST_INIT_DESCRIPTOR(cd, F90_RANK_G(ad), F90_KIND_G(ad), F90_LEN_G(ad),
                            F90_FLAGS_G(dd), DIST_DIST_TARGET_G(dd));
      wrk_rank = F90_RANK_G(ad);
      for (i = 1; i <= wrk_rank; ++i) {
        __DIST_SET_DISTRIBUTION(cd, i, 1, actual_extent[i - 1], 0, NULL);
      }
      F90_FLAGS_P(cd, F90_FLAGS_G(cd) & ~__TEMPLATE);
      I8(__fort_finish_descriptor)(cd);
    }
  }

  if (intent != __IN) {
    I8(__fort_cycle_bounds)(ad);
    I8(local_copy)(db, cd, F90_LBASE_G(dd)-1, ab, ad, F90_LBASE_G(ad)-1,
		       F90_RANK_G(ad), __COPY_OUT);
  }
  I8(__fort_local_deallocate)(db);
}

void
ENTFTN(COPY_OUT, copy_out)(void *ab, void *db, F90_Desc *ad, F90_Desc *dd,
                            __INT_T *intent)
{
  if (*intent & __F77_LOCAL_DUMMY)
    F90_FLAGS_P(dd, F90_FLAGS_G(dd) | __F77_LOCAL_DUMMY);

  I8(__fort_copy_out)(ab, db, ad, dd, *intent << __INTENT_SHIFT);
}

void
ENTFTN(CHECK_BLOCK_SIZE, check_block_size)(void *ab, F90_Desc *ad)
{

  /* used to check block size variable argument for
   * block(k) and cyclic(k) distributions.
   *
   * The compiler will generate a call to this routine when
   * user is using a variable argument for a block(k) or
   * cyclic(k) distribution.
   */

  int kind;

  if (!ISSCALAR(ad)) {
    __fort_abort(
        "check_block_size: block(k)/cyclic(k) size argument must be scalar");
  }

  kind = *((int *)ad);

  switch (kind) {

  case __INT1:
    if (*((__INT1_T *)ab) >= 1)
      return;
    break;
  case __INT2:
    if (*((__INT2_T *)ab) >= 1)
      return;
    break;
  case __INT4:
    if (*((__INT4_T *)ab) >= 1)
      return;
    break;

  case __INT8:
    if (*((__INT8_T *)ab) >= 1)
      return;
    break;

  default:
    __fort_abort(
        "check_block_size: invalid data type for block(k)/cyclic(k) size");
  }

  __fort_abort("check_block_size: block(k)/cyclic(k) size must be >= 1");
}

/** \brief General routine to create a template (alignment or distribution)
 * descriptor
 *
 *<pre>
 * Varargs: [ [ proc *dist_target, ]
 *            __INT_T *isstar,
 *            { [__INT_T paxis, ] ( __INT_T *dstfmt, | __INT_T * genBlockArray,
 *              __INT_T * genBlockArrayExtent,) }* ]
 *          [ F90_Desc *align_target, __INT_T *conform,
 *            [ __INT_T *collapse,
 *              { __INT_T *taxis, __INT_T *tstride, __INT_T *toffset, }*
 *              __INT_T *single,
 *              { __INT_T *coordinate, }* ] ]
 *          { __INT_T *lbound, __INT_T *ubound, }*
 *</pre>
 */
void
ENTFTN(TEMPLATE, template)
        (F90_Desc *dd, __INT_T *p_rank, __INT_T *p_flags, ...)
{
  va_list va;
  DECL_HDR_PTRS(td);
  DECL_HDR_PTRS(tu);
  DECL_DIM_PTRS(tdd);
  proc *tp;

  __INT_T rank, flags, isstar = 0;
  /* isstar format: bits 0..6 set for each */
  /*                collapsed dimension.   */
  /*                bits 7..27 dist. format*/
  /*                for each non-collapsed */
  /*                dim. (3-bit fields:)   */
  /*                                       */
  /*                0 - block,block(k),    */
  /*                    cyclic, cyclic(k)  */
  /*                1 - gen_block          */
  /*                2 - indirect (if we    */
  /*                    ever support it)   */
  /*             3..7 - future expansion.  */

  __INT_T collapse, single;
  __INT_T coordinate[MAXDIMS];
  __INT_T lbound[MAXDIMS], ubound[MAXDIMS];

  _io_spec dist_target_spec;
  _io_spec dist_format_spec;
  _io_spec align_target_spec;

  __INT_T i, m;
  void *unused;

  rank = *p_rank;
  flags = *p_flags;

  if (LOCAL_MODE)
    flags |= __LOCAL;

  dist_target_spec =
      (_io_spec)(flags >> __DIST_TARGET_SHIFT & __DIST_TARGET_MASK);
  dist_format_spec =
      (_io_spec)(flags >> __DIST_FORMAT_SHIFT & __DIST_FORMAT_MASK);
  align_target_spec =
      (_io_spec)(flags >> __ALIGN_TARGET_SHIFT & __ALIGN_TARGET_MASK);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d TEMPLATE %x ", GET_DIST_LCPU, dd);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (dd == NULL)
    __fort_abort("TEMPLATE: invalid descriptor");
  if ((align_target_spec && (dist_target_spec | dist_format_spec)) ||
      flags & __INHERIT) {
    __fort_abort("TEMPLATE: invalid flags");
  }
#endif

  va_start(va, p_flags);

  if (align_target_spec) {

#if defined(DEBUG)
    if (align_target_spec != __PRESCRIPTIVE &&
        align_target_spec != __DESCRIPTIVE)
      __fort_abort("TEMPLATE: bad align-target flags");
#endif

    /* get align-target spec */

    td = va_arg(va, F90_Desc *);

#if defined(DEBUG)
    if (td == NULL || F90_TAG_G(td) != __DESC)
      __fort_abort("TEMPLATE: invalid align-target descriptor");
#endif

    tu = DIST_ALIGN_TARGET_G(td);

#if defined(DEBUG)
    if (tu == NULL || F90_TAG_G(tu) != __DESC)
      __fort_abort("TEMPLATE: invalid ultimate align-target");
#endif

    unused = va_arg(va, __INT_T *);

    if (flags & __ASSUMED_SHAPE) {

      /* template assumes shape of ultimate align-target but
         bounds are shifted to lbound == 1 */

      collapse = 0;
      td = tu;
      for (i = 1; i <= rank; ++i) {
        SET_DIM_PTRS(tdd, td, i - 1);
        lbound[i - 1] = 1;
        ubound[i - 1] = F90_DPTR_EXTENT_G(tdd);
      }
      single = 0;
    } else {

      if (flags & __IDENTITY_MAP) {
        collapse = 0;
        single = 0;
#if defined(DEBUG)
        if (F90_RANK_G(td) != rank)
          __fort_abort("TEMPLATE: align-target rank mismatch");
#endif
      } else {
        collapse = *va_arg(va, __INT_T *);
        for (i = 0; i < rank; ++i) {
          if (!(collapse >> i & 1)) {
            unused = va_arg(va, __INT_T *);
            unused = va_arg(va, __INT_T *);
            unused = va_arg(va, __INT_T *);
          }
        }
        single = *va_arg(va, __INT_T *);
        if (single >> F90_RANK_G(td))
          __fort_abort("TEMPLATE: invalid single alignment axis");
        m = single;
        for (i = 0; m > 0; ++i, m >>= 1) {
          if (m & 1)
            coordinate[i] = *va_arg(va, __INT_T *);
        }
      }

      /* get array bounds */

      for (i = 1; i <= rank; ++i) {
        lbound[i - 1] = *va_arg(va, __INT_T *);
        ubound[i - 1] = (i == rank && flags & __ASSUMED_SIZE)
                            ? lbound[i - 1]
                            : *va_arg(va, __INT_T *);
      }
    }

    /* create alignment descriptor */
    __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, tu);

    for (i = 1; i <= rank; ++i) {
      I8(__fort_set_alignment)(dd, i, lbound[i - 1], ubound[i - 1], 0, 0, 0);
    }

  } else {

#if defined(DEBUG)
    if (flags & __ASSUMED_SHAPE)
      __fort_abort("TEMPLATE: assumed shape needs align-target");
#endif

    /* get distribution target spec --- for F90 consume var args */

    switch (dist_target_spec) {

    case __PRESCRIPTIVE:
    case __DESCRIPTIVE:
      tp = va_arg(va, proc *);
      break;

    case __OMITTED:
      tp = NULL;
      break;

    case __TRANSCRIPTIVE:
      __fort_abort("TEMPLATE: bad dist-target flags");
    }

    /* get distribution format spec */

    switch (dist_format_spec) {

    case __PRESCRIPTIVE:
    case __DESCRIPTIVE:

      isstar = *va_arg(va, __INT_T *);

      for (i = 0; i < rank; ++i) {
        if (isstar >> i & 1) {
          if (flags & __DUMMY_COLLAPSE_PAXIS) {
            unused = va_arg(va, __INT_T *);
            unused = va_arg(va, __INT_T *);
          }
        } else {

          if (flags & __DIST_TARGET_AXIS) {
            unused = va_arg(va, __INT_T *);
          }

          if (!(isstar & EXTENSION_BLOCK_MASK) ||
              !(((isstar & EXTENSION_BLOCK_MASK) >> (7 + 3 * i)) & 0x01)) {
            unused = va_arg(va, __INT_T *);
          } else {
            unused = va_arg(va, __INT_T *);

            if (!(flags & __ASSUMED_GB_EXTENT))
              unused = va_arg(va, __INT_T *);
          }
        }
      } /*end for i<rank*/
      break;

    case __OMITTED:
      break;

    case __TRANSCRIPTIVE:
      __fort_abort("TEMPLATE: bad dist-format flags");
    }

    /* get array bounds */

    for (i = 1; i <= rank; ++i) {
      lbound[i - 1] = *va_arg(va, __INT_T *);
      ubound[i - 1] = (i == rank && flags & __ASSUMED_SIZE)
                          ? lbound[i - 1]
                          : *va_arg(va, __INT_T *);
    }

    if (tp == NULL) {
      tp = __fort_localproc();
    } else if (tp->tag != __PROC)
      __fort_abort("TEMPLATE: invalid dist-target");

    /* create distribution descriptor */

    __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, tp);

    /* Set gen_block copy for each dimension and set distribution for
     * each dimension...
     */

    for (i = 1; i <= rank; ++i) {
      __DIST_SET_DISTRIBUTION(dd, i, lbound[i - 1], ubound[i - 1], 0, NULL);
    }
  }
  va_end(va);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("TEMPLATE: %d dd=%x align_target=%lx dist_target=%lx\n",
           GET_DIST_LCPU, dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
}
/** \brief Optimized routine to create a template 
 *
 *  Varargs: { __INT_T *lbound, __INT_T *ubound, }*
 */
void
ENTF90(TEMPLATE, template)(F90_Desc *dd, __INT_T *p_rank, __INT_T *p_flags,
                           __INT_T *p_kind, __INT_T *p_len, ...)
{
  va_list va;

  __INT_T rank, flags, len;
  dtype kind;
  __INT_T i;
  __INT_T gsize, lbase;

  rank = *p_rank;
  flags = *p_flags;
  kind = (dtype)*p_kind;
  len = *p_len;

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d TEMPLATE %x ", GET_DIST_LCPU, dd);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (dd == NULL)
    __fort_abort("TEMPLATE: invalid descriptor");
#endif

  va_start(va, p_len);

#if defined(DEBUG)
  if (flags & __ASSUMED_SHAPE)
    __fort_abort("TEMPLATE: assumed shape unsupported");
#endif

  /* create distribution descriptor */

  __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, NULL);

  gsize = lbase = 1;
  for (i = 1; i <= rank; ++i) {
    __INT_T __extent, l, u;
    DECL_DIM_PTRS(_dd);
    SET_DIM_PTRS(_dd, dd, i - 1);
    l = *va_arg(va, __INT_T *);
    u = (i == rank && flags & __ASSUMED_SIZE) ? l : *va_arg(va, __INT_T *);
    if (u >= l) {
      __extent = u - l + 1;
    } else {
      __extent = 0;
      u = l - 1;
    }
    F90_DPTR_LBOUND_P(_dd, l);
    DPTR_UBOUND_P(_dd, u);
    F90_DPTR_SSTRIDE_P(_dd, 1);
    F90_DPTR_SOFFSET_P(_dd, 0);
    lbase -= gsize * l;
    F90_DPTR_LSTRIDE_P(_dd, gsize);
    gsize *= __extent;
  }
  F90_LBASE_P(dd, lbase);
  F90_LSIZE_P(dd, gsize);
  F90_GSIZE_P(dd, gsize);
  F90_KIND_P(dd, kind);
  F90_LEN_P(dd, len);
  va_end(va);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("TEMPLATE: %d dd=%x align_target=%lx dist_target=%lx\n",
           GET_DIST_LCPU, dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
} /* f90_template */

/* versions optimized for one/two/three-dimensions without varargs, not
 * assumed-size */
#define TEMPLATE(dd, i, lb, ub, lbase, gsize)                                  \
  {                                                                            \
    __INT_T __extent, u, l;                                                    \
    DECL_DIM_PTRS(_dd);                                                        \
    SET_DIM_PTRS(_dd, dd, i - 1);                                              \
    l = lb;                                                                    \
    u = ub;                                                                    \
    if (u >= l) {                                                              \
      __extent = u - l + 1;                                                    \
    } else {                                                                   \
      __extent = 0;                                                            \
      u = l - 1;                                                               \
    }                                                                          \
    F90_DPTR_LBOUND_P(_dd, l);                                                 \
    DPTR_UBOUND_P(_dd, u);                                                     \
    F90_DPTR_SSTRIDE_P(_dd, 1);                                                \
    F90_DPTR_SOFFSET_P(_dd, 0);                                                \
    lbase -= gsize * l;                                                        \
    F90_DPTR_LSTRIDE_P(_dd, gsize);                                            \
    gsize *= __extent;                                                         \
  }

/** \brief Optimized routine to create a template for 1 dimension array */
void
ENTF90(TEMPLATE1, template1)(F90_Desc *dd, __INT_T *p_flags, __INT_T *p_kind,
                             __INT_T *p_len, __INT_T *p_l1, __INT_T *p_u1)
{
  __INT_T rank, flags, len;
  dtype kind;
  __INT_T gsize, lbase;

  rank = 1;
  flags = *p_flags;
  kind = (dtype)*p_kind;
  len = *p_len;

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d TEMPLATE %x ", GET_DIST_LCPU, dd);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (dd == NULL)
    __fort_abort("TEMPLATE: invalid descriptor");
#endif

#if defined(DEBUG)
  if (flags & __ASSUMED_SHAPE)
    __fort_abort("TEMPLATE: assumed shape unsupported");
#endif

  /* create distribution descriptor */

  __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, NULL);

  gsize = lbase = 1;
  TEMPLATE(dd, 1, *p_l1, *p_u1, lbase, gsize);
  F90_LBASE_P(dd, lbase);
  F90_LSIZE_P(dd, gsize);
  F90_GSIZE_P(dd, gsize);
  F90_KIND_P(dd, kind);
  F90_LEN_P(dd, len);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("TEMPLATE: %d dd=%x align_target=%lx dist_target=%lx\n",
           GET_DIST_LCPU, dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
} /* f90_template1 */

/** \brief Optimized routine to create a template for 2 dimesion array */
void
ENTF90(TEMPLATE2, template2)(F90_Desc *dd, __INT_T *p_flags,
                                  __INT_T *p_kind, __INT_T *p_len,
                                  __INT_T *p_l1, __INT_T *p_u1, __INT_T *p_l2,
                                  __INT_T *p_u2)
{
  __INT_T rank, flags, len;
  dtype kind;
  __INT_T gsize, lbase;

  rank = 2;
  flags = *p_flags;
  kind = (dtype)*p_kind;
  len = *p_len;

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d TEMPLATE %x ", GET_DIST_LCPU, dd);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (dd == NULL)
    __fort_abort("TEMPLATE: invalid descriptor");
#endif

#if defined(DEBUG)
  if (flags & __ASSUMED_SHAPE)
    __fort_abort("TEMPLATE: assumed shape unsupported");
#endif

  /* create distribution descriptor */

  __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, NULL);

  gsize = lbase = 1;
  TEMPLATE(dd, 1, *p_l1, *p_u1, lbase, gsize);
  TEMPLATE(dd, 2, *p_l2, *p_u2, lbase, gsize);
  F90_LBASE_P(dd, lbase);
  F90_LSIZE_P(dd, gsize);
  F90_GSIZE_P(dd, gsize);
  F90_KIND_P(dd, kind);
  F90_LEN_P(dd, len);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("TEMPLATE: %d dd=%x align_target=%lx dist_target=%lx\n",
           GET_DIST_LCPU, dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
} /* f90_template2 */

/** \brief Optimized routine to create a template for 3 dimesion array */
void
ENTF90(TEMPLATE3, template3)(F90_Desc *dd, __INT_T *p_flags,
                             __INT_T *p_kind, __INT_T *p_len,
                             __INT_T *p_l1, __INT_T *p_u1, __INT_T *p_l2,
                             __INT_T *p_u2, __INT_T *p_l3, __INT_T *p_u3)
{
  __INT_T rank, flags, len;
  dtype kind;
  __INT_T gsize, lbase;

  rank = 3;
  flags = *p_flags;
  kind = (dtype)*p_kind;
  len = *p_len;

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d TEMPLATE %x ", GET_DIST_LCPU, dd);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (dd == NULL)
    __fort_abort("TEMPLATE: invalid descriptor");
#endif

#if defined(DEBUG)
  if (flags & __ASSUMED_SHAPE)
    __fort_abort("TEMPLATE: assumed shape unsupported");
#endif

  /* create distribution descriptor */

  __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, NULL);

  gsize = lbase = 1;
  TEMPLATE(dd, 1, *p_l1, *p_u1, lbase, gsize);
  TEMPLATE(dd, 2, *p_l2, *p_u2, lbase, gsize);
  TEMPLATE(dd, 3, *p_l3, *p_u3, lbase, gsize);
  F90_LBASE_P(dd, lbase);
  F90_LSIZE_P(dd, gsize);
  F90_GSIZE_P(dd, gsize);
  F90_KIND_P(dd, kind);
  F90_LEN_P(dd, len);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("TEMPLATE: %d dd=%x align_target=%lx dist_target=%lx\n",
           GET_DIST_LCPU, dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
} /* f90_template3 */

/** \brief Optimized routine to create a template from value arguments */
void ENTF90(TEMPLATE1V, template1v)(F90_Desc *dd, __INT_T flags, dtype kind,
                                    __INT_T len, __INT_T l1, __INT_T u1)
{
  __INT_T rank;
  __INT_T gsize, lbase;

  rank = 1;

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d TEMPLATE %x ", GET_DIST_LCPU, dd);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (dd == NULL)
    __fort_abort("TEMPLATE: invalid descriptor");
#endif

#if defined(DEBUG)
  if (flags & __ASSUMED_SHAPE)
    __fort_abort("TEMPLATE: assumed shape unsupported");
#endif

  /* create distribution descriptor */

  __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, NULL);

  gsize = lbase = 1;
  TEMPLATE(dd, 1, l1, u1, lbase, gsize);
  F90_LBASE_P(dd, lbase);
  F90_LSIZE_P(dd, gsize);
  F90_GSIZE_P(dd, gsize);
  F90_KIND_P(dd, kind);
  F90_LEN_P(dd, len);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("TEMPLATE: %d dd=%x align_target=%lx dist_target=%lx\n",
           GET_DIST_LCPU, dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
} /* f90_template1v */

/** \brief Optimized routine to create a template for 2 dim array from
 * value arguments
 */
void
ENTF90(TEMPLATE2V, template2v)(F90_Desc *dd, __INT_T flags, __INT_T kind,
                               __INT_T len, __INT_T l1, __INT_T u1,
                               __INT_T l2, __INT_T u2)
{
  __INT_T rank;
  __INT_T gsize, lbase;

  rank = 2;

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d TEMPLATE %x ", GET_DIST_LCPU, dd);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (dd == NULL)
    __fort_abort("TEMPLATE: invalid descriptor");
#endif

#if defined(DEBUG)
  if (flags & __ASSUMED_SHAPE)
    __fort_abort("TEMPLATE: assumed shape unsupported");
#endif

  /* create distribution descriptor */

  __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, NULL);

  gsize = lbase = 1;
  TEMPLATE(dd, 1, l1, u1, lbase, gsize);
  TEMPLATE(dd, 2, l2, u2, lbase, gsize);
  F90_LBASE_P(dd, lbase);
  F90_LSIZE_P(dd, gsize);
  F90_GSIZE_P(dd, gsize);
  F90_KIND_P(dd, kind);
  F90_LEN_P(dd, len);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("TEMPLATE: %d dd=%x align_target=%lx dist_target=%lx\n",
           GET_DIST_LCPU, dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
} /* f90_template2v */

/** \brief Optimized routine to create a template for 3 dim array from
 * value arguments
 */
void
ENTF90(TEMPLATE3V, template3v)(F90_Desc *dd, __INT_T flags, __INT_T kind,
                               __INT_T len, __INT_T l1, __INT_T u1,
                               __INT_T l2, __INT_T u2, __INT_T l3,
                               __INT_T u3)
{
  __INT_T rank;
  __INT_T gsize, lbase;

  rank = 3;

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d TEMPLATE %x ", GET_DIST_LCPU, dd);
    __fort_show_flags(flags);
    printf("\n");
  }
  if (dd == NULL)
    __fort_abort("TEMPLATE: invalid descriptor");
#endif

#if defined(DEBUG)
  if (flags & __ASSUMED_SHAPE)
    __fort_abort("TEMPLATE: assumed shape unsupported");
#endif

  /* create distribution descriptor */

  __DIST_INIT_DESCRIPTOR(dd, rank, __NONE, 0, flags, NULL);

  gsize = lbase = 1;
  TEMPLATE(dd, 1, l1, u1, lbase, gsize);
  TEMPLATE(dd, 2, l2, u2, lbase, gsize);
  TEMPLATE(dd, 3, l3, u3, lbase, gsize);
  F90_LBASE_P(dd, lbase);
  F90_LSIZE_P(dd, gsize);
  F90_GSIZE_P(dd, gsize);
  F90_KIND_P(dd, kind);
  F90_LEN_P(dd, len);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("TEMPLATE: %d dd=%x align_target=%lx dist_target=%lx\n",
           GET_DIST_LCPU, dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
} /* f90_template3v */
#undef TEMPLATE

/* Varargs: { [ __INT_T *no, __INT_T *po, ] }* */
void
ENTFTN(INSTANCE, instance)(F90_Desc *dd, F90_Desc *td, __INT_T *p_kind,
                           __INT_T *p_len, __INT_T *p_collapse, ...)
{
  DECL_HDR_PTRS(tu);
  DECL_DIM_PTRS(tdd);

  dtype kind;
  __INT_T i, len;

#if defined(DEBUG)
  if (dd == NULL)
    __fort_abort("INSTANCE: invalid descriptor");
  if (td == NULL || F90_TAG_G(td) != __DESC)
    __fort_abort("INSTANCE: invalid template descriptor");
  if (td == dd && ~F90_FLAGS_G(td) & __TEMPLATE)
    __fort_abort("INSTANCE: descriptor is not a template");
#endif

  kind = (dtype)*p_kind;
  len = *p_len;

  if (td == dd) {

    /* convert template to array descriptor in place */

    F90_KIND_P(dd, kind);
    F90_LEN_P(dd, len);
  } else {

    /* create new array descriptor aligned to template */

    tu = DIST_ALIGN_TARGET_G(td);
    __DIST_INIT_DESCRIPTOR(dd, F90_RANK_G(td), kind, len, F90_FLAGS_G(td), tu);

    for (i = 1; i <= F90_RANK_G(td); ++i) {
      SET_DIM_PTRS(tdd, td, i - 1);
      I8(__fort_set_alignment)(dd, i, F90_DPTR_LBOUND_G(tdd), 
                                    DPTR_UBOUND_G(tdd), 0, 0, 0);
    }
  }
  F90_FLAGS_P(dd, F90_FLAGS_G(dd) & ~__TEMPLATE);
  I8(__fort_finish_descriptor)(dd);

#if defined(DEBUG)
  if (__fort_test & DEBUG_RDST) {
    printf("%d INSTANCE %x align-target=%lx dist-target=%lx\n", GET_DIST_LCPU,
           dd, DIST_ALIGN_TARGET_G(dd), DIST_DIST_TARGET_G(dd));
  }
#endif
}

void
ENTFTN(FREE, free)(F90_Desc *d)
{
  DECL_HDR_PTRS(t);
  DECL_HDR_PTRS(p);
  DECL_HDR_PTRS(n);

  if (d == NULL || F90_TAG_G(d) != __DESC)
    __fort_abort("FREE: invalid descriptor (already freed?)");

  t = DIST_ALIGN_TARGET_G(d);
  if (t == NULL || F90_TAG_G(t) != __DESC) {
    if (DIST_NEXT_ALIGNEE_G(d) != NULL)
      __fort_abort("FREE: alignee has invalid align-target");
  } else if (t != d) {

    /* freeing an alignee. */

    if (F90_FLAGS_G(t) & __DYNAMIC) {

      /* find descriptor in align-target's alignees list */

      p = t;
      n = DIST_NEXT_ALIGNEE_G(t);
      while (n != NULL && n != d) {
        p = n;
        n = DIST_NEXT_ALIGNEE_G(n);
      }
      if (n != d)
        __fort_abort("FREE: alignee not in alignees list");

      /* unlink descriptor from alignees list */

      DIST_NEXT_ALIGNEE_P(p, DIST_NEXT_ALIGNEE_G(d));
      DIST_NEXT_ALIGNEE_P(d, NULL);
    }
  } else {

    /* freeing an ultimate align-target.  the compiler
       sometimes frees align-targets first, so we break the
       links from all accessible alignees. */

    n = d;
    while (n != NULL) {
      p = n;
      n = DIST_NEXT_ALIGNEE_G(n);
      DIST_ALIGN_TARGET_P(p, NULL);
      DIST_NEXT_ALIGNEE_P(p, NULL);
    }
  }
  F90_TAG_P(d, __NONE);
}

void
ENTFTN(FREEN, freen)(__INT_T *cnt, ...)
{
  va_list va;
  int n;

  va_start(va, cnt);
  for (n = *cnt; n > 0; --n)
    ENTFTN(FREE, free)(va_arg(va, void *));
  va_end(va);
}

void
ENTF90(ADDR_1_DIM_1ST_ELEM, addr_1_dim_1st_elem)
        (char *ab, char *ad, char **addr)
{
  *addr = ab ? ad : ab;
}

/* no longer needed after 5.2 */
void
ENTF90(COPY_F77_ARG, copy_f77_arg)(char **ab, F90_Desc *ad, char **db_ptr,
                                   int *copy_in)
{
  DECL_HDR_VARS(c);
  DECL_HDR_PTRS(cd) = c;
  DECL_DIM_PTRS(cdd);
  DECL_DIM_PTRS(add);
  __INT_T wrk_rank;
  __INT_T extent;
  __INT_T nbr_elem;
  __INT_T i;

  if (F90_FLAGS_G(ad) & __SEQUENTIAL_SECTION) {
    if (*copy_in) {
      *db_ptr = (char *)F90_GBASE_G(ad);
    }
    return;
  }

  if (*ab == 0) {
    *db_ptr = 0;
    return;
  }

  /* This does not need to be a complete descriptor.  It Only needs to
   * be complete enough to accommodate the array copy rtn.
   */
  __DIST_INIT_DESCRIPTOR(cd, F90_RANK_G(ad), F90_KIND_G(ad), F90_LEN_G(ad),
                        F90_FLAGS_G(cd), NULL);
  nbr_elem = 1;
  wrk_rank = F90_RANK_G(ad);
  for (i = 1; i <= wrk_rank; ++i) {
    SET_DIM_PTRS(cdd, cd, i - 1);
    SET_DIM_PTRS(add, ad, i - 1);
    F90_DPTR_LBOUND_P(cdd, 1);
    extent = F90_DPTR_EXTENT_G(add);
    DPTR_UBOUND_P(cdd, extent);
    F90_DPTR_SSTRIDE_P(cdd, 1); /* section stride */
    F90_DPTR_SOFFSET_P(cdd, 0); /* section offset */
    F90_DPTR_LSTRIDE_P(cdd, 0);
    nbr_elem *= extent;
  }
  if (nbr_elem > 0) {
    I8(__fort_finish_descriptor)(cd);

    if (*copy_in) {
      (void)I8(__fort_alloc)(nbr_elem, F90_KIND_G(cd), F90_LEN_G(cd), 0,
                            (char **)db_ptr, 0, 0, 0,
                            __fort_malloc_without_abort);

      I8(local_copy)(*db_ptr, cd, F90_LBASE_G(cd)-1, *ab, ad, 
                        F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_IN);
    } else {
      I8(local_copy)(*db_ptr, cd, F90_LBASE_G(cd)-1, *ab, ad, 
                        F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_OUT);

      I8(__fort_dealloc)(*db_ptr, 0, __fort_gfree);
    }
  }
}

/** \brief Copy argument if necessary
 *
 * when passing a pointer array section to F77 dummy,
 * if we don't know whether it is sequential (contiguous),
 * call this routine; this is the same as the one above, but this one works
 * the previous routine should vanish in the next release
 * copy_in == 1 means copy-in
 * copy_in == 0 means copy-out intent(inout)
 * copy_in == 2 means copy-out intent(out), no need to copy values
 */
void
ENTF90(COPY_F77_ARGW, copy_f77_argw)(char **ab, F90_Desc *ad, char *afirst,
                                     char **db_ptr, int *copy_in)
{
  DECL_HDR_VARS(c);
  DECL_HDR_PTRS(cd) = c;
  DECL_DIM_PTRS(cdd);
  DECL_DIM_PTRS(add);
  __INT_T wrk_rank;
  __INT_T extent;
  __INT_T nbr_elem;
  __INT_T i;

  if (F90_FLAGS_G(ad) & __SEQUENTIAL_SECTION) {
    if (*copy_in == 1) {
      *db_ptr = afirst;
    }
    return;
  }

  if (ab == NULL || *ab == NULL) {
    *db_ptr = 0;
    return;
  }

  /* This does not need to be a complete descriptor.  It only needs to
   * be complete enough to accommodate the array copy rtn.
   */
  __DIST_INIT_DESCRIPTOR(cd, F90_RANK_G(ad), F90_KIND_G(ad), F90_LEN_G(ad),
                        F90_FLAGS_G(cd), NULL);
  nbr_elem = 1;
  wrk_rank = F90_RANK_G(ad);
  for (i = 1; i <= wrk_rank; ++i) {
    SET_DIM_PTRS(cdd, cd, i - 1);
    SET_DIM_PTRS(add, ad, i - 1);
    F90_DPTR_LBOUND_P(cdd, 1);
    extent = F90_DPTR_EXTENT_G(add);
    DPTR_UBOUND_P(cdd, extent);
    F90_DPTR_SSTRIDE_P(cdd, 1); /* section stride */
    F90_DPTR_SOFFSET_P(cdd, 0); /* section offset */
    F90_DPTR_LSTRIDE_P(cdd, 0);
    nbr_elem *= extent;
  }
  if (nbr_elem > 0) {
    I8(__fort_finish_descriptor)(cd);

    if (*copy_in == 1) {
      (void)I8(__fort_alloc)(nbr_elem, F90_KIND_G(cd), F90_LEN_G(cd), 0,
                            (char **)db_ptr, 0, 0, 0,
                            __fort_malloc_without_abort);

      I8(local_copy)(*db_ptr, cd, F90_LBASE_G(cd)-1, *ab, ad, 
                       F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_IN);
    } else {
      if (*copy_in == 0) {
        I8(local_copy)(*db_ptr, cd, F90_LBASE_G(cd)-1, *ab, ad, 
                           F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_OUT);
      }
      I8(__fort_dealloc)(*db_ptr, 0, __fort_gfree);
    }
  }
}

/** \brief Copy argument if necessary
 *
 * when passing a pointer array section to F77 dummy,
 * if we don't know whether it is sequential (contiguous),
 * call this routine; this is the same as the one above, but this one works
 * the previous routine should vanish in the next release
 * copy_in == 1 means copy-in
 * copy_in == 0 means copy-out intent(inout)
 * copy_in == 2 means copy-out intent(out), no need to copy values
 */
void
ENTF90(COPY_F77_ARGL, copy_f77_argl)(char **ab, F90_Desc *ad, char *afirst,
                                     char **db_ptr, int *copy_in, int *len)
{
  DECL_HDR_VARS(c);
  DECL_HDR_PTRS(cd) = c;
  DECL_DIM_PTRS(cdd);
  DECL_DIM_PTRS(add);
  __INT_T wrk_rank;
  __INT_T extent;
  __INT_T nbr_elem;
  __INT_T i;

  if ((F90_FLAGS_G(ad) & __SEQUENTIAL_SECTION) && F90_LEN_G(ad) == *len) {
    if (*copy_in == 1) {
      *db_ptr = afirst;
    }
    return;
  }

  if (ab == NULL || *ab == NULL) {
    *db_ptr = 0;
    return;
  }

  /* This does not need to be a complete descriptor.  It only needs to
   * be complete enough to accommodate the array copy rtn.
   */
  __DIST_INIT_DESCRIPTOR(cd, F90_RANK_G(ad), F90_KIND_G(ad), *len,
                        F90_FLAGS_G(ad), NULL);
  nbr_elem = 1;
  wrk_rank = F90_RANK_G(ad);
  for (i = 1; i <= wrk_rank; ++i) {
    SET_DIM_PTRS(cdd, cd, i - 1);
    SET_DIM_PTRS(add, ad, i - 1);
    F90_DPTR_LBOUND_P(cdd, 1);
    extent = F90_DPTR_EXTENT_G(add);
    DPTR_UBOUND_P(cdd, extent);
    F90_DPTR_SSTRIDE_P(cdd, 1); /* section stride */
    F90_DPTR_SOFFSET_P(cdd, 0); /* section offset */
    F90_DPTR_LSTRIDE_P(cdd, 1);
    nbr_elem *= extent;
  }
  if (nbr_elem > 0) {
    I8(__fort_finish_descriptor)(cd);

    if (*copy_in == 1) {
      (void)I8(__fort_alloc)(nbr_elem, F90_KIND_G(cd), F90_LEN_G(cd), 0,
                            (char **)db_ptr, 0, 0, 0,
                            __fort_malloc_without_abort);

      I8(local_copy)(*db_ptr, cd, F90_LBASE_G(cd)-1, *ab, ad, 
                       F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_IN);
    } else {
      if (*copy_in == 0) {
        I8(local_copy)(*db_ptr, cd, F90_LBASE_G(cd)-1, *ab, ad, 
                           F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_OUT);
      }
      I8(__fort_dealloc)(*db_ptr, 0, __fort_gfree);
    }
  }
}

/** \brief Copy argument if necessary
 *
 * when passing an assumed-shape array section to F77 dummy,
 * we don't know whether it is sequential (contiguous)
 * (we only know leftmost dimension is stride-1),
 * call this routine; this is almost the same as the one above
 * copy_in == 1 means copy-in
 * copy_in == 0 means copy-out intent(inout)
 * copy_in == 2 means copy-out intent(out), no need to copy values
 */
void
ENTF90(COPY_F77_ARGSL, copy_f77_argsl)
         (char *ab, F90_Desc *ad, char *afirst, char **db_ptr, int *copy_in,
          int *len)
{
  DECL_HDR_VARS(c);
  DECL_HDR_PTRS(cd) = c;
  DECL_DIM_PTRS(cdd);
  DECL_DIM_PTRS(add);
  __INT_T wrk_rank;
  __INT_T extent;
  __INT_T nbr_elem;
  __INT_T i;

  if ((F90_FLAGS_G(ad) & __SEQUENTIAL_SECTION) && F90_LEN_G(ad) == *len) {
    /* incoming argument is sequential, no need to copy */
    if (*copy_in == 1) {
      *db_ptr = afirst;
    }
    return;
  }

  if (!ab) {
    *db_ptr = 0;
    return;
  }

  /* This does not need to be a complete descriptor.  It only needs to
   * be complete enough to accommodate the array copy rtn.
   */
  __DIST_INIT_DESCRIPTOR(cd, F90_RANK_G(ad), F90_KIND_G(ad), *len,
                        F90_FLAGS_G(ad), NULL);
  nbr_elem = 1;
  wrk_rank = F90_RANK_G(ad);
  for (i = 1; i <= wrk_rank; ++i) {
    SET_DIM_PTRS(cdd, cd, i - 1);
    SET_DIM_PTRS(add, ad, i - 1);
    F90_DPTR_LBOUND_P(cdd, 1);
    extent = F90_DPTR_EXTENT_G(add);
    DPTR_UBOUND_P(cdd, extent);
    F90_DPTR_SSTRIDE_P(cdd, 1); /* section stride */
    F90_DPTR_SOFFSET_P(cdd, 0); /* section offset */
    F90_DPTR_LSTRIDE_P(cdd, 1);
    nbr_elem *= extent;
  }
  if (nbr_elem > 0) {
    I8(__fort_finish_descriptor)(cd);

    if (*copy_in == 1) {
      (void)I8(__fort_alloc)(nbr_elem, F90_KIND_G(cd), F90_LEN_G(cd), 0,
                            (char **)db_ptr, 0, 0, 0,
                            __fort_malloc_without_abort);

      I8(local_copy)(*db_ptr, cd, F90_LBASE_G(cd)-1, ab, ad, 
                       F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_IN);
    } else {
      if (*copy_in == 0) {
        I8(local_copy)(*db_ptr, cd, F90_LBASE_G(cd)-1, ab, ad, 
                           F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_OUT);
      }
      I8(__fort_dealloc)(*db_ptr, 0, __fort_gfree);
    }
  }
}

static void
init_unassociated_pointer_desc(F90_Desc *d)
{
  __DIST_INIT_DESCRIPTOR(d, 0, 0, 0, 0, 0);
  F90_DIM_LBOUND_P(d, 0, 0);
  F90_DIM_EXTENT_P(d, 0, 0);
  F90_DIM_SSTRIDE_P(d, 0, 0);
  F90_DIM_SOFFSET_P(d, 0, 0);
  F90_DIM_LSTRIDE_P(d, 0, 0);
}

/** \brief Copy argument if necessary
 *
 * when passing an array section to an assumed-shape dummy argument,
 * if we don't know the actual strides (such as with a pointer),
 * if the leftmost dimension stride is actually one, we don't have to do a copy
 * copy_in == 1 means copy-in
 * copy_in == 0 means copy-out intent(inout)
 * copy_in == 2 means copy-out intent(out), no need to copy values
 */
void
ENTF90(COPY_F90_ARG, copy_f90_arg)
         (char **ab, F90_Desc *ad, char **db, F90_Desc *dd, int *copy_in)
{
  DECL_DIM_PTRS(cdd);
  DECL_DIM_PTRS(add);
  __INT_T wrk_rank;
  __INT_T extent, ubound;
  __INT_T nbr_elem;
  __INT_T i;

  if (!*ab) {
    init_unassociated_pointer_desc(dd);
    return;
  }

  if (*copy_in == 1) {
    if (F90_DIM_SSTRIDE_G(ad, 0) == 1 && F90_DIM_LSTRIDE_G(ad, 0) == 1) {
      __INT_T running_lstride;
      *db = *ab;
      /*
       * create a descriptor to be passed in
       */
      __DIST_INIT_DESCRIPTOR(dd, F90_RANK_G(ad), F90_KIND_G(ad), F90_LEN_G(ad),
                            F90_FLAGS_G(ad), NULL);
      /* __DIST_INIT_DESCRIPTOR always sets __SEQUENTIAL_SECTION */
      if (F90_LEN_G(dd) != GET_DIST_SIZE_OF(F90_KIND_G(dd)))
        F90_FLAGS_P(dd, (F90_FLAGS_G(dd) & ~__SEQUENTIAL_SECTION));
      wrk_rank = F90_RANK_G(ad);
      running_lstride = 1;
      for (i = 0; i < wrk_rank; ++i) {
        SET_DIM_PTRS(cdd, dd, i);
        SET_DIM_PTRS(add, ad, i);
        F90_DPTR_LBOUND_P(cdd, F90_DPTR_LBOUND_G(add));
        ubound = F90_DPTR_UBOUND_G(add);
        DPTR_UBOUND_P(cdd, ubound);
        F90_DPTR_SSTRIDE_P(cdd, F90_DPTR_SSTRIDE_G(add));
        F90_DPTR_SOFFSET_P(cdd, F90_DPTR_SOFFSET_G(add));
        F90_DPTR_LSTRIDE_P(cdd, F90_DPTR_LSTRIDE_G(add));
        /* check to see if its a sequential (continous) section */
        if (F90_DPTR_SSTRIDE_G(cdd) != 1 ||
            F90_DPTR_LSTRIDE_G(cdd) != running_lstride) {
          F90_FLAGS_P(dd, (F90_FLAGS_G(dd) & ~__SEQUENTIAL_SECTION));
        }
        running_lstride *= F90_DPTR_EXTENT_G(add);
      }
      /*__fort_finish_descriptor(dd);*/
      F90_DIST_DESC_P(dd, F90_DIST_DESC_G(ad)); /* TYPE_DESC pointer */
      F90_GBASE_P(dd, F90_GBASE_G(ad));
      F90_LBASE_P(dd, F90_LBASE_G(ad));
      F90_GSIZE_P(dd, F90_GSIZE_G(ad));
      F90_LSIZE_P(dd, F90_LSIZE_G(ad));
    } else {
      /*
       * create a descriptor to be passed in
       */
      __DIST_INIT_DESCRIPTOR(dd, F90_RANK_G(ad), F90_KIND_G(ad), F90_LEN_G(ad),
                            F90_FLAGS_G(ad), NULL);
      if (!(F90_FLAGS_G(ad) & __SEQUENTIAL_SECTION))
        F90_FLAGS_P(dd, (F90_FLAGS_G(dd) & ~__SEQUENTIAL_SECTION));
      nbr_elem = 1;
      wrk_rank = F90_RANK_G(ad);
      for (i = 0; i < wrk_rank; ++i) {
        SET_DIM_PTRS(cdd, dd, i);
        SET_DIM_PTRS(add, ad, i);
        F90_DPTR_LBOUND_P(cdd, 1);
        extent = F90_DPTR_EXTENT_G(add);
        DPTR_UBOUND_P(cdd, extent);
        F90_DPTR_SSTRIDE_P(cdd, 1); /* section stride */
        F90_DPTR_SOFFSET_P(cdd, 0); /* section offset */
        F90_DPTR_LSTRIDE_P(cdd, 0);
        nbr_elem *= extent;
      }
      I8(__fort_finish_descriptor)(dd);
      F90_DIST_DESC_P(dd, F90_DIST_DESC_G(ad)); /* TYPE_DESC pointer */

      /* if this is the copy-in, allocate the array, copy the data */
      (void)I8(__fort_alloc)(nbr_elem, F90_KIND_G(dd), F90_LEN_G(dd), 0,
                            (char **)db, 0, 0, 0, __fort_malloc_without_abort);

      I8(local_copy)(*db, dd, F90_LBASE_G(dd)-1, *ab, ad, 
                            F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_IN);
    }
  } else {
    if (F90_DIM_SSTRIDE_G(ad, 0) == 1 && F90_DIM_LSTRIDE_G(ad, 0) == 1) {
    } else {
      /* if this is the copy-out, copy the data base, deallocate array */
      if (*copy_in == 0) {
        I8(local_copy)(*db, dd, F90_LBASE_G(dd)-1, *ab, ad, 
                               F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_OUT);
      }

      I8(__fort_dealloc)(*db, 0, __fort_gfree);
    }
  }

} /* copy_f90_arg */

/** \brief Copy argument if necessary
 *
 * When passing an array section to an assumed-shape dummy argument,
 * if we don't know the actual strides (such as with a pointer),
 * if the leftmost dimension stride is actually one, we don't have to do a copy
 * copy_in == 1 means copy-in
 * copy_in == 0 means copy-out intent(inout)
 * copy_in == 2 means copy-out intent(out), no need to copy values
 * the len argument is to assure the 'len' matches the datatype length
 */
void
ENTF90(COPY_F90_ARGL, copy_f90_argl)(char **ab, F90_Desc *ad, char **db,
                                     F90_Desc *dd, int *copy_in, int *len)
{
  DECL_DIM_PTRS(cdd);
  DECL_DIM_PTRS(add);
  __INT_T wrk_rank;
  __INT_T extent, ubound;
  __INT_T nbr_elem;
  __INT_T i;
  __INT_T nlbase;

  if (!*ab) {
    init_unassociated_pointer_desc(dd);
    return;
  }

  if (*copy_in == 1) {
    if (F90_DIM_SSTRIDE_G(ad, 0) == 1 && F90_DIM_LSTRIDE_G(ad, 0) == 1 &&
        F90_LEN_G(ad) == *len) {
      __INT_T running_lstride;
      *db = *ab;

      /*
       * create a descriptor to be passed in
       */
      __DIST_INIT_DESCRIPTOR(dd, F90_RANK_G(ad), F90_KIND_G(ad), F90_LEN_G(ad),
                            F90_FLAGS_G(ad), NULL);
      /* __DIST_INIT_DESCRIPTOR always sets __SEQUENTIAL_SECTION */
      if (F90_LEN_G(dd) != GET_DIST_SIZE_OF(F90_KIND_G(dd)))
        F90_FLAGS_P(dd, (F90_FLAGS_G(dd) & ~__SEQUENTIAL_SECTION));
      wrk_rank = F90_RANK_G(ad);
      running_lstride = 1;
      nlbase = F90_LBASE_G(ad) - 1;
      for (i = 0; i < wrk_rank; ++i) {
        SET_DIM_PTRS(cdd, dd, i);
        SET_DIM_PTRS(add, ad, i);
        nlbase += F90_DPTR_LSTRIDE_G(add) * F90_DPTR_LBOUND_G(add);
        F90_DPTR_LBOUND_P(cdd, F90_DPTR_LBOUND_G(add));
        ubound = F90_DPTR_UBOUND_G(add);
        DPTR_UBOUND_P(cdd, ubound);
        F90_DPTR_SSTRIDE_P(cdd, F90_DPTR_SSTRIDE_G(add));
        F90_DPTR_SOFFSET_P(cdd, F90_DPTR_SOFFSET_G(add));
        F90_DPTR_LSTRIDE_P(cdd, F90_DPTR_LSTRIDE_G(add));
        /* check to see if its a sequential (continous) section */
        if (F90_DPTR_SSTRIDE_G(cdd) != 1 ||
            F90_DPTR_LSTRIDE_G(cdd) != running_lstride) {
          F90_FLAGS_P(dd, (F90_FLAGS_G(dd) & ~__SEQUENTIAL_SECTION));
        }
        running_lstride *= F90_DPTR_EXTENT_G(add);
      }
      /*__fort_finish_descriptor(dd);*/
      F90_GBASE_P(dd, F90_GBASE_G(ad));
      F90_LBASE_P(dd, F90_LBASE_G(ad) - nlbase);
      *db = *ab + nlbase * F90_LEN_G(ad);
      F90_GSIZE_P(dd, F90_GSIZE_G(ad));
      F90_LSIZE_P(dd, F90_LSIZE_G(ad));
      F90_DIST_DESC_P(dd, F90_DIST_DESC_G(ad)); /* TYPE_DESC pointer */
    } else {
      /*
       * create a descriptor to be passed in
       */
      __DIST_INIT_DESCRIPTOR(dd, F90_RANK_G(ad), F90_KIND_G(ad), *len,
                            F90_FLAGS_G(ad), NULL);
      nlbase = F90_LBASE_G(dd) - 1;
      nbr_elem = 1;
      wrk_rank = F90_RANK_G(ad);
      for (i = 0; i < wrk_rank; ++i) {
        SET_DIM_PTRS(cdd, dd, i);
        SET_DIM_PTRS(add, ad, i);
        nlbase = F90_DPTR_LSTRIDE_G(add) * F90_DPTR_LBOUND_G(add);
        F90_DPTR_LBOUND_P(cdd, 1);
        extent = F90_DPTR_EXTENT_G(add);
        DPTR_UBOUND_P(cdd, extent);
        F90_DPTR_SSTRIDE_P(cdd, 1); /* section stride */
        F90_DPTR_SOFFSET_P(cdd, 0); /* section offset */
        F90_DPTR_LSTRIDE_P(cdd, 0);
        nbr_elem *= extent;
      }
      I8(__fort_finish_descriptor)(dd);
      nlbase = F90_LBASE_G(dd) - 1;
      F90_DIST_DESC_P(dd, F90_DIST_DESC_G(ad)); /* TYPE_DESC pointer */

      /* if this is the copy-in, allocate the array, copy the data */
      (void)I8(__fort_alloc)(nbr_elem, F90_KIND_G(dd), *len, 0, (char **)db, 0,
                            0, 0, __fort_malloc_without_abort);

      I8(local_copy)(*db, dd, F90_LBASE_G(dd)-1, *ab, ad, 
                            F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_IN);
    }
  } else {
    if (F90_DIM_SSTRIDE_G(ad, 0) == 1 && F90_DIM_LSTRIDE_G(ad, 0) == 1 &&
        (!len || F90_LEN_G(ad) == *len)) {
    } else {
      /* if this is the copy-out, copy the data base, deallocate array */
      if (*copy_in == 0) {
        I8(local_copy)(*db, dd, F90_LBASE_G(dd)-1, *ab, ad, 
                               F90_LBASE_G(ad)-1, F90_RANK_G(ad), __COPY_OUT);
      }

      I8(__fort_dealloc)(*db, 0, __fort_gfree);
    }
  }

} /* copy_f90_argl */

int
ENTF90(CONFORMABLE_DD, conformable_dd)(char *db, F90_Desc *dd, F90_Desc *sd)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int ndim;
  int i;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  ndim = F90_RANK_G(dd);
  for (i = 0; i < ndim; i++) {
    if (F90_DIM_EXTENT_G(dd, i) != F90_DIM_EXTENT_G(sd, i)) {
      conformable = -1;
      break;
    }
  }

  if (conformable != 1 && F90_GSIZE_G(dd) >= F90_GSIZE_G(sd)) {
    conformable = 0;
  }

  return conformable;
}

/* Varargs(pointers pass): src_size, rank, src_extnt1, ... ,srcextntn */
int
ENTF90(CONFORMABLE_DN, conformable_dn)(char *db, F90_Desc *dd, ...)
{
  va_list va;
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int gsize;
  int extnt;
  int ndim;
  int i;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  va_start(va, dd);
  ndim = *va_arg(va, __INT_T *);
  gsize = 1;
  for (i = 0; i < ndim; i++) {
    extnt = *va_arg(va, __INT_T *);
    gsize *= extnt;
    if (F90_DIM_EXTENT_G(dd, i) != extnt) {
      conformable = -1;
    }
  }
  va_end(va);

  if (conformable != 1 && F90_GSIZE_G(dd) >= gsize) {
    conformable = 0;
  }

  return conformable;
}

int
ENTF90(CONFORMABLE_D1V, conformable_d1v)(char *db, F90_Desc *dd, 
	__INT_T extnt0)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  if (F90_DIM_EXTENT_G(dd, 0) != extnt0) {
    conformable = -1;
  }

  if (conformable != 1 && F90_GSIZE_G(dd) >= extnt0) {
    conformable = 0;
  }

  return conformable;
} 

int
ENTF90(CONFORMABLE_D2V, conformable_d2v)(char *db, F90_Desc *dd, 
	__INT_T extnt0, __INT_T extnt1)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int gsize;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  gsize = extnt0 * extnt1;
  if (F90_DIM_EXTENT_G(dd, 0) != extnt0 || 
       F90_DIM_EXTENT_G(dd, 1) != extnt1) {
    conformable = -1;
  }

  if (conformable != 1 && F90_GSIZE_G(dd) >= gsize) {
    conformable = 0;
  }

  return conformable;
}

int
ENTF90(CONFORMABLE_D3V, conformable_d3v)(char *db, F90_Desc *dd, 
	__INT_T extnt0, __INT_T extnt1, __INT_T extnt2)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int gsize;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  gsize = extnt0 * extnt1 * extnt2;
  if (F90_DIM_EXTENT_G(dd, 0) != extnt0 || 
       F90_DIM_EXTENT_G(dd, 1) != extnt1 ||
       F90_DIM_EXTENT_G(dd, 2) != extnt2) {
    conformable = -1;
  }

  if (conformable != 1 && F90_GSIZE_G(dd) >= gsize) {
    conformable = 0;
  }

  return conformable;
}

/* Varargs(value only pass): src_size, rank, src_extnt1, ... ,srcextntn */
int
ENTF90(CONFORMABLE_DNV, conformable_dnv)(char *db, F90_Desc *dd, ...)
{
  va_list va;
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int gsize;
  int extnt;
  int ndim;
  int i;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  va_start(va, dd);
  ndim = va_arg(va, __INT_T);
  gsize = 1;
  for (i = 0; i < ndim; i++) {
    extnt = va_arg(va, __INT_T);
    gsize *= extnt;
    if (F90_DIM_EXTENT_G(dd, i) != extnt) {
      conformable = -1;
    }
  }
  va_end(va);

  if (conformable != 1 && F90_GSIZE_G(dd) >= gsize) {
    conformable = 0;
  }

  return conformable;
}

/* Varargs(pointer pass): dest_size, rank, dest_extnt1, ... ,dest_extntn */
int
ENTF90(CONFORMABLE_ND, conformable_nd)(char *db, F90_Desc *sd, ...)
{
  va_list va;
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int gsize;
  int extnt;
  int ndim;
  int i;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  va_start(va, sd);
  ndim = *va_arg(va, __INT_T *);
  gsize = 1;
  for (i = 0; i < ndim; i++) {
    extnt = *va_arg(va, __INT_T *);
    gsize *= extnt;
    if (extnt != F90_DIM_EXTENT_G(sd, i)) {
      conformable = -1;
    }
  }
  va_end(va);

  if (conformable != 1 && gsize >= F90_GSIZE_G(sd)) {
    conformable = 0;
  }

  return conformable;
}

int
ENTF90(CONFORMABLE_1DV, conformable_1dv)(char *db, F90_Desc *sd, __INT_T extnt0)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  if (extnt0 != F90_DIM_EXTENT_G(sd, 0)) {
    conformable = -1;
  }

  if (conformable != 1 && extnt0 >= F90_GSIZE_G(sd)) {
    conformable = 0;
  }

  return conformable;
}

int
ENTF90(CONFORMABLE_2DV, conformable_2dv)(char *db, F90_Desc *sd, 
	__INT_T extnt0, __INT_T extnt1)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int gsize;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  gsize = extnt0 * extnt1;
  if (extnt0 != F90_DIM_EXTENT_G(sd, 0) ||
      extnt1 != F90_DIM_EXTENT_G(sd, 1)) {
    conformable = -1;
  }

  if (conformable != 1 && gsize >= F90_GSIZE_G(sd)) {
    conformable = 0;
  }

  return conformable;
}

int
ENTF90(CONFORMABLE_3DV, conformable_3dv)(char *db, F90_Desc *sd, 
	__INT_T extnt0, __INT_T extnt1, __INT_T extnt2)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int gsize;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  gsize = extnt0 * extnt1 * extnt2;
  if (extnt0 != F90_DIM_EXTENT_G(sd, 0) ||
      extnt1 != F90_DIM_EXTENT_G(sd, 1) ||
      extnt2 != F90_DIM_EXTENT_G(sd, 2)) {
    conformable = -1;
  }

  if (conformable != 1 && gsize >= F90_GSIZE_G(sd)) {
    conformable = 0;
  }

  return conformable;
}

/* Varargs(value pass): dest_size, rank, dest_extnt1, ... ,dest_extntn */
int
ENTF90(CONFORMABLE_NDV, conformable_ndv)(char *db, F90_Desc *sd, ...)
{
  va_list va;
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int gsize;
  int extnt;
  int ndim;
  int i;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  va_start(va, sd);
  ndim = va_arg(va, __INT_T);
  gsize = 1;
  for (i = 0; i < ndim; i++) {
    extnt = va_arg(va, __INT_T);
    gsize *= extnt;
    if (extnt != F90_DIM_EXTENT_G(sd, i)) {
      conformable = -1;
    }
  }
  va_end(va);

  if (conformable != 1 && gsize >= F90_GSIZE_G(sd)) {
    conformable = 0;
  }

  return conformable;
}

/* Varargs(pointer pass): rank, dest_extnt1, src_extnt1, ... ,dest_extntn, src_extntn */
int
ENTF90(CONFORMABLE_NN, conformable_nn)(char *db, ...)
{
  va_list va;
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int dgsize;
  int sgsize;
  int dextnt;
  int sextnt;
  int ndim;
  int i;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  va_start(va, db);
  dgsize = 1;
  sgsize = 1;
  ndim = *va_arg(va, __INT_T *);

  for (i = 0; i < ndim; i++) {
    dextnt = *va_arg(va, __INT_T *);
    dgsize *= dextnt;
    sextnt = *va_arg(va, __INT_T *);
    sgsize *= sextnt;
    if (dextnt != sextnt) {
      conformable = -1;
    }
  }
  va_end(va);

  if (conformable != 1 && dgsize >= sgsize) {
    conformable = 0;
  }

  return conformable;
}

int
ENTF90(CONFORMABLE_11V, conformable_11v)(char *db, __INT_T dextnt0, __INT_T sextnt0)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  if (dextnt0 != sextnt0) {
    conformable = -1;
  }

  if (conformable != 1 && dextnt0 >= sextnt0) {
    conformable = 0;
  }

  return conformable;
}

int
ENTF90(CONFORMABLE_22V, conformable_22v)(char *db, 
	__INT_T dextnt0, __INT_T sextnt0,
	__INT_T dextnt1, __INT_T sextnt1)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int dgsize;
  int sgsize;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  dgsize = dextnt0 * dextnt1;
  sgsize = sextnt0 * sextnt1;

  if (dextnt0 != sextnt0 || 
	dextnt1 != sextnt1) {
    conformable = -1;
  }

  if (conformable != 1 && dgsize >= sgsize) {
    conformable = 0;
  }

  return conformable;
}

int
ENTF90(CONFORMABLE_33V, conformable_33v)(char *db, 
	__INT_T dextnt0, __INT_T sextnt0,
	__INT_T dextnt1, __INT_T sextnt1,
	__INT_T dextnt2, __INT_T sextnt2)
{
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int dgsize;
  int sgsize;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  dgsize = dextnt0 * dextnt1 * dextnt2;
  sgsize = sextnt0 * sextnt1 * sextnt2;

  if (dextnt0 != sextnt0 || 
	dextnt1 != sextnt1 ||
	dextnt2 != sextnt2) {
    conformable = -1;
  }

  if (conformable != 1 && dgsize >= sgsize) {
    conformable = 0;
  }

  return conformable;
}

/* Varargs(value pass): rank, dest_extnt1, src_extnt1, ... ,dest_extntn, src_extntn */
int
ENTF90(CONFORMABLE_NNV, conformable_nnv)(char *db, ...)
{
  va_list va;
  int conformable = 1; /*  1 ==> conformable
                        *  0 ==> not conformable but big enough
                        * -1 --> not conformable, no big enough */
  int dgsize;
  int sgsize;
  int dextnt;
  int sextnt;
  int ndim;
  int i;

  if (!I8(__fort_allocated)(db)) {
    return -1;
  }

  va_start(va, db);
  dgsize = 1;
  sgsize = 1;
  ndim = va_arg(va, __INT_T);

  for (i = 0; i < ndim; i++) {
    dextnt = va_arg(va, __INT_T);
    dgsize *= dextnt;
    sextnt = va_arg(va, __INT_T);
    sgsize *= sextnt;
    if (dextnt != sextnt) {
      conformable = -1;
    }
  }
  va_end(va);

  if (conformable != 1 && dgsize >= sgsize) {
    conformable = 0;
  }

  return conformable;
}

__INT_T
ENTF90(IS_CONTIGUOUS, is_contiguous)(char *ab, F90_Desc *ad)
{
  if (!ab || !ad || I8(is_nonsequential_section)(ad, F90_RANK_G(ad)))
    return 0;
  return GET_DIST_TRUE_LOG;
}

/** \brief Print a contiguous error message and abort.
 *
 * This function will also call is_nonsequential_section() to get the
 * first dimension of the array that is non-contiguous and include it in the
 * error message.
 *
 * \param ptr is the pointer we are checking.
 * \param pd is the descriptor we are checking.
 * \param lineno is the source line number we are checking. 
 * \param ptrnam is the name of pointer, null-terminated string.
 * \param srcfil is the name of source file, null-terminated string.
 * \param flags is currently 1 when ptr is an optional argument, else 0.
 */
void
ENTF90(CONTIGERROR, contigerror)(void *ptr, F90_Desc *pd, __INT_T lineno,
                                 char *ptrnam, char *srcfil, __INT_T flags)
{
    char str[200];
    int dim;

    if (flags == 1 && ptr == NULL) {
      /* ignore non-present optional argument */
      return;
    }
    dim = I8(is_nonsequential_section)(pd, F90_RANK_G(pd));
    sprintf(str, "Runtime Error at %s, line %d: Pointer assignment of "
                 "noncontiguous target (dimension %d) to CONTIGUOUS pointer "
                 "%s\n", srcfil, lineno, dim, ptrnam); 
    __fort_abort(str);
}

/** \brief Check whether a pointer is associated with a contiguous array object.
 *
 * If the pointer is not associated with a contiguous array object, then a 
 * message is printed to stderr and the user program aborts.
 * 
 * \param ptr is the pointer we are checking.
 * \param pd is the descriptor we are checking.
 * \param lineno is the source line number we are checking. 
 * \param ptrnam is the name of pointer, null-terminated string.
 * \param srcfil is the name of source file, null-terminated string.
 * \param flags is currently 1 when ptr is an optional argument, else 0.
 */
void
ENTF90(CONTIGCHK, contigchk)(void *ptr, F90_Desc *pd, __INT_T lineno, 
                             char *ptrnam, char *srcfil, __INT_T flags)
{
  if (flags == 1 && ptr == NULL) {
    /* ignore non-present optional argument */
    return;
  }

  if (!(ENTF90(IS_CONTIGUOUS, is_contiguous)(ptr, pd))) {
    ENTF90(CONTIGERROR, contigerror)(ptr, pd, lineno, ptrnam, srcfil, flags);
  }
}

/** \brief Execute a command line.
  * 
  * \param command is the command to be executed.
  * \param wait controls to execute command synchronously or asynchronously.
  * \param exitstatus is the value of exit status.
  * \param cmdstat shows the status of command execution.
  * \param cmdmsg is the assigned explanatory message.
  * \param exitstat_int_kind is the integer kind for the exitstat.
  * \param cmdstat_int_kind is the integer kind for the cmdstat.
  * \param DCLEN64(command) is generated by compiler which contains the length
  *        of the command string.
  * \param DCLEN64(cmdmsg) is generated by compiler which contains the length
           of the cmdmsg string.
  */
void
ENTF90(EXECCMDLINE, execcmdline)(DCHAR(command), __LOG_T *wait,
                                 __INT_T *exitstatus,
                                 __INT_T *cmdstat, DCHAR (cmdmsg),
                                 __INT_T *exitstat_int_kind, 
                                 __INT_T *cmdstat_int_kind 
                                 DCLEN64(command) DCLEN64(cmdmsg)) {
  char *cmd, *cmdmes;
  int cmdmes_len;
#if (defined(TARGET_LINUX_X8664) || defined(TARGET_OSX_X8664) || defined(TARGET_LINUX_POWER) || defined(TARGET_LINUX_ARM32) || defined(TARGET_LINUX_ARM64)) && !defined(TARGET_WIN)
  int stat;
#endif
  int cmdflag = 0;
  enum CMD_ERR{NO_SUPPORT_ERR=-1, FORK_ERR=1, EXECL_ERR=2, SIGNAL_ERR=3};
  
  cmd = __fstr2cstr(CADR(command), CLEN(command));
  cmdmes = (char*) CADR(cmdmsg);
  cmdmes_len = CLEN(cmdmsg);

  if (cmdstat)
    store_int_kind(cmdstat, cmdstat_int_kind, 0);
#if (defined(TARGET_LINUX_X8664) || defined(TARGET_OSX_X8664) || defined(TARGET_LINUX_POWER) || defined(TARGET_LINUX_ARM32) || defined(TARGET_LINUX_ARM64)) && !defined(TARGET_WIN)
  pid_t pid, w;
  int wstatus, ret;
  
  /* If WAIT is present with the value false, and the processor supports
   * asynchronous execution of the command, the command is executed
   * asynchronously; otherwise it is executed synchronously.
   */
  pid = fork();
  if (pid < 0) {
    if (cmdmes)
      ftn_msgcpy(cmdmes, "Fork failed", cmdmsg_len); 
    if (cmdstat)
      store_int_kind(cmdstat, cmdstat_int_kind, FORK_ERR);
  } else if (pid == 0) {
    ret = execl("/bin/sh", "sh", "-c", cmd, (char *) NULL);
    exit(ret);
  } else {
    // either wait is not specified or wait is true, then synchronous mode
    if ( !wait || *wait == -1) {
#if DEBUG
      printf("either wait is not specified or Wait = .true.\n");
      printf("Synchronous execution mode!\n");
#endif
      /* code executed by parent, wait for children */
      w = waitpid(pid, &wstatus, WUNTRACED | WCONTINUED);
      if (w == -1)
        cmdflag = EXECL_ERR;
    
      if (WIFEXITED(wstatus)) { 
        stat = WEXITSTATUS(wstatus);

        if (exitstatus)
          store_int_kind(exitstatus, exitstat_int_kind, stat);
      }

      if (WIFSIGNALED(wstatus))
        cmdflag = SIGNAL_ERR;

      if (cmdstat && cmdflag > 0)
        store_int_kind(cmdstat, cmdstat_int_kind, cmdflag);

      if (cmdmes) {
        switch (cmdflag) {
        case EXECL_ERR:
          ftn_msgcpy(cmdmes, "Excel failed", cmdmsg_len); 
          break;
        case SIGNAL_ERR:
          ftn_msgcpy(cmdmes, "Signal error", cmdmsg_len);
          break;
        }
      }

      /* If a condition occurs that would assign a nonzero value to CMDSTAT 
         but the CMDSTAT variable is not present, error termination is
         initiated.
       */
      if (!cmdstat && cmdflag > 0) {
        fprintf(__io_stderr(), "ERROR STOP ");
        exit(cmdflag);
      }
        
#if DEBUG
      if (WIFEXITED(wstatus)) {
        printf("exited, status=%d\n", WEXITSTATUS(wstatus));
      } else if (WIFSIGNALED(wstatus)) {
        printf("killed by signal %d\n", WTERMSIG(wstatus));
      } else if (WIFSTOPPED(wstatus)) {
        printf("stopped by signal %d\n", WSTOPSIG(wstatus));
      } else if (WIFCONTINUED(wstatus)) {
        printf("continued\n");
      }
#endif
    } // end else
  }
#else // defined(TARGET_WIN)
  // Windows runtime work to be continued.
  cmdflag = NO_SUPPORT_ERR;
  if (cmdmes)
    ftn_msgcpy(cmdmes, "No Windows support", cmdmsg_len); 
  if (cmdstat)
    store_int_kind(cmdstat, cmdstat_int_kind, cmdflag);
  else
    __fort_abort("execute_command_line: not yet supported on Windows\n");
#endif
   __cstr_free(cmd);
}  

// TODO: Code restructure needed to reduce redundant codes.
/*
 * helper function to store an int/logical value into a varying int/logical
 */
static void
store_int_kind(void *b, __INT_T *int_kind, int v)
{
  switch (*int_kind) {
  case 1:
    *(__INT1_T *)b = (__INT1_T)v;
    break;
  case 2:
    *(__INT2_T *)b = (__INT2_T)v;
    break;
  case 4:
    *(__INT4_T *)b = (__INT4_T)v;
    break;
  case 8:
    *(__INT8_T *)b = (__INT8_T)v;
    break;
  default:
    __fort_abort("store_int_kind: unexpected int kind");
  }
}

// TODO: Code restructure needed to reduce redundant codes.
/** \brief Copy msg string to statmsg and padding with blank space at the end.
  * 
  * \param statmsg is the Fortran string we want to assign values.
  * \param msg is the string contains error message.
  * \param len is the length of statmsg. 
  */
static void 
ftn_msgcpy(char *statmsg, const char *msg, int len) {
  int i;
  for (i=0; i<len; ++i) {
    statmsg[i] = *msg ? *msg++ : ' ';
  }
}
