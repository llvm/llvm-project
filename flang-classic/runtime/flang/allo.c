/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * Runtime memory allocation routines
 */

#include <string.h>
#include <stdlib.h>
#include "stdioInterf.h"
#include "fioMacros.h"
#include "type.h"
#define DEBUG 1
#include "llcrit.h"
#include "f90alloc.h"

MP_SEMAPHORE(static, sem);

#include "fort_vars.h"

/* always attempt to quad align */

#define ASZ 16

#define AUTOASZ 16

#undef USE_MEMALIGN
#define USE_MEMALIGN

/* use pointer arithmetic to get above the next bytes boundary, then
 * mask out the bits where they reside in CRAY pointers. Note: ignores
 * bit offset bits, assumes len is a power of two.
 */

#define MASKP(len) ((len)-1)

#define ALIGNP(p, len)                                                         \
  ((char *)(((__POINT_T)(((char *)p) + (len)-1)) & ~MASKP(len)))

/*
 * The implementation of a normal allocte pre-dates memalign; to accomodate:
 * + NULL is 0,
 * + we want to align the area to improve cache utiltization
 * the allocate mallocs a large enough space to whose address is massaged
 * to give the desired alignment and the pointer to the malloc area is
 * stored 1 pointer-element back from the returned address.
 *
 * USE_MEMALIGN says we're going to use memalign to support the use of our
 * ALIGN= extension in ALLOCATE. This method coexists with the 'nornmal'
 * method; therefore, a list is needed to keep track of ALIGN'd allocaale
 * The allocation via memalign gives us the pointer returned to the user;
 * we DO NOT use the 1 pointer-element idea.  For alloc/deallocate, we
 * must always be sure that the list is always checked/set before continuing
 * to the normal method.
 *
 * This can all be simplified to just use memalign and never randomize the
 * alignment of allocations (could impact apps the have automatic arrsys,
 * e.g, bwaves.
 */

/* header structure on every allocated block */

typedef struct ALLO_HDR ALLO_HDR;
struct ALLO_HDR {
  ALLO_HDR *next; /* list of allocated blocks */
  char *area;     /* address of area returned to caller */
                  /* NOTE:
                   * 01/04/2010 -- area may be overwritten by a pointer which
                   * locates the header.  The address of the header is stored
                   * in the pointer-size location immediately preceeding the
                   * address returned to the caller.  The basic idea is to
                   * deallocate without having to first search a list. */
};

/* define an ALLO_HDR structure to be the head of the list */

#define SEL_HDR(area) ((((__POINT_T)area >> 7)) & (num_hdrs - 1))
#define NUM_HDRS 1024 * 64

#define XYZZY(a) ((void **)a)[-1]
#define XYZZYP(a, p) XYZZY(a) = p

static ALLO_HDR *allo_list;
static long num_hdrs = NUM_HDRS;

/* these are used with -ta=tesla:managed and -ta=tesla:pin */
#define __man_malloc malloc
#define __man_callocx calloc
#define __man_free free
#define __pin_malloc malloc
#define __pin_callocx calloc
#define __pin_free free

/* insure allo_list is allocated */

#define ALLHDR()

#ifdef FLANG_ALLO_UNUSED
/** \brief
 * Allocate ALLO_HDR list */
static void
allhdr()
{
  char *p, *q;
  long n;

  MP_P(sem);
  if (allo_list != (ALLO_HDR *)0) { /* check again */
    MP_P(sem);
    return;
  }
  p = getenv("F90_ALLOCATE_HDRS"); /* check environment */
  if (p != NULL) {
    num_hdrs = strtol(p, &q, 0);
    if ((*q == 'k') || (*q == 'K')) {
      num_hdrs *= 1024;
    } else if ((*q == 'm') || (*q == 'M')) {
      num_hdrs *= 1024 * 1024;
    } else if ((*q == 'g') || (*q == 'G')) {
      num_hdrs *= 1024 * 1024 * 1024;
    }
  }
  n = 8; /* must be power of 2 */
  while (n < num_hdrs) {
    n = n << 1;
  }
  num_hdrs = n; /* allocate headers */
  allo_list = (ALLO_HDR *)calloc(num_hdrs, sizeof(ALLO_HDR));
  if (allo_list == (ALLO_HDR *)0) {
    __abort(1, "No memory for allocate headers");
  }

  MP_V(sem);
}
#endif

/** \brief
 * Return nonzero if addresses p1 and p2 are aligned with respect to a
 * multiple of the length of the data type.
 */
int
I8(__fort_ptr_aligned)(char *p1, dtype kind, int len, char *p2)
{
  __POINT_T diff, off;

  off = diff = p1 - p2;
  if (kind != __STR && kind != __DERIVED)
    off >>= GET_DIST_SHIFTS(kind);
  else
    off /= len;
  return (off * len == diff);
}

/** \brief
 * Compute pointer-sized index offset to replace Cray pointer, return
 * aligned address. offset assumes 1-based arrays.
 */
char *
I8(__fort_ptr_offset)(char **pointer, __POINT_T *offset, char *base,
                      dtype kind, __CLEN_T len, char *area)
{
  __POINT_T off;
  char *aligned;

  if (ISPRESENT(offset)) {
    if (ISPRESENT(pointer) && *pointer == base) {

      /* offset is present but cray pointer is being used */

      *offset = 0;
      *pointer = area;
      return area;
    }

    /* offset is present and being used */

    off = (area >= base) ? area - base + len - 1 : base - area;
    if (kind != __STR && kind != __DERIVED)
      off >>= GET_DIST_SHIFTS(kind);
    else
      off /= len;

    if (area < base)
      off = -off;

    *offset = off + 1;
    aligned = base + off * len;
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d ptr_offset: area %p base %p + (%d - 1)*%lu = %p\n",
             GET_DIST_LCPU, area, base, (int)*offset, len, aligned);
#endif
  } else
    aligned = area;
  if (ISPRESENT(pointer))
    *pointer = aligned;
  return aligned;
}

/** \brief
 * Compute pointer-sized index offset to replace Cray pointer.  offset
 * assumes 1-based arrays.  DOES NOT WORK FOR CHARACTER DATA TYPE
 */
void
ENTFTN(PTR_OFFSET, ptr_offset)(__POINT_T *offset, char **ptr, char *base,
                               __INT_T *kind)
{
  char *area;

#if defined(DEBUG)
  if (*kind == __STR || *kind == __DERIVED)
    __fort_abort("PTR_OFFSET: cannot handle character or derived type");
#endif
  area = ISPRESENT(ptr) ? *ptr : (char *)ABSENT;
  *offset = ((area - base) >> GET_DIST_SHIFTS(*kind)) + 1;
}

/** \brief
 * allocate space using given 'malloc' function. if present, set
 * pointer and pointer-sized integer offset from base address. offset
 * is in units of the data type length.
 */
char *
I8(__fort_alloc)(__INT_T nelem, dtype kind, size_t len, __STAT_T *stat,
                 char **pointer, __POINT_T *offset, char *base, int check,
                 void *(*mallocfn)(size_t))
{
  ALLO_HDR *p;
  char *area;
  size_t need, size, slop, sizeof_hdr;
  __POINT_T off;
  char msg[80];
  char *p_env;

#if (defined(_WIN64))
#define ALN_LARGE
#else
#undef ALN_LARGE
#endif
/* always use the larger padding   - used be for just win, but
 * now it makes a bug difference on linux with newer (since Jul 2007)
 * processor.
 */
#define ALN_LARGE

  size_t ALN_MINSZ = 128000;
  size_t ALN_UNIT = 64;
  size_t ALN_MAXADJ = 4096;

#define ALN_THRESH (ALN_MAXADJ / ALN_UNIT)
  static size_t aln_n = 0;
  static int env_checked = 0;
  size_t myaln;

  sizeof_hdr = AUTOASZ;

  if (!env_checked) {
    env_checked = 1;

    p_env = getenv("F90_ALN_MINSZ");
    if (p_env != NULL)
      ALN_MINSZ = atol(p_env);

    p_env = getenv("F90_ALN_UNIT");
    if (p_env != NULL)
      ALN_UNIT = atol(p_env);

    p_env = getenv("F90_ALN_MAXADJ");
    if (p_env != NULL)
      ALN_MAXADJ = atol(p_env);
  }

  ALLHDR();

  if (!ISPRESENT(stat))
    stat = NULL;
  if (!ISPRESENT(pointer))
    pointer = NULL;
  if (!ISPRESENT(offset))
    offset = NULL;
  need = (nelem <= 0) ? 0 : (size_t)nelem * len;
  slop = 0;
  /* SLOPPY COMMENT:
   * here, we add some slop so we can align the eventual
   * allocated address after adding the size of the ALLO_HDR.
   * We are going to align to ASZ.  Since we know that the malloc
   * function returns something at least 8-byte aligned, we only
   * should have to add (ASZ-8) slop.
   * Actually, we shouldn't even have to do this if we know
   * that the malloc function returns aligned results and
   * skipping ALLO_HDR will keep it aligned */
  if (nelem > 1 || need > 2 * sizeof_hdr)
    slop = (offset && len > (ASZ - 8)) ? len : (ASZ - 8);
  size = (sizeof_hdr + slop + need + ASZ - 1) & ~(ASZ - 1);
  MP_P(sem);
  if (size > ALN_MINSZ) {
    myaln = aln_n;
    size += ALN_UNIT * myaln;
    if (aln_n < ALN_THRESH)
      aln_n++;
    else
      aln_n = 0;
  }
  p = (size < need) ? NULL : (ALLO_HDR *)mallocfn(size);
  MP_V(sem);
  if (p == NULL) {
    if (pointer)
      *pointer = NULL;
    if (offset)
      *offset = 1;
    if (stat) {
      *stat = 1;
      return NULL;
    }
    MP_P_STDIO;
    sprintf(msg, "ALLOCATE: %lu bytes requested; not enough memory", need);
    MP_V_STDIO;
    __fort_abort(msg);
  }
  if (stat)
    *stat = 0;
  area = (char *)p + sizeof_hdr;
  if (offset) {
    off = area - base + len - 1;
    if (kind != __STR && kind != __DERIVED)
      off >>= GET_DIST_SHIFTS(kind);
    else
      off /= len;

    *offset = off + 1;
    area = base + off * len;
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d alloc: need %lu size %lu p %p area %p end %p"
             " base %p offset %ld len %lu\n",
             GET_DIST_LCPU, need, size, p, area, (char *)p + size - 1, base,
             *offset, len);
#endif
  } else {
    /* see SLOPPY COMMENT above */
    if (nelem > 1 || need > 2 * sizeof_hdr)
      area = ALIGNP(area, ASZ);
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d alloc: need %lu size %lu p %p area %p end %p\n",
             GET_DIST_LCPU, need, size, p, area, (char *)p + size - 1);
#endif
  }
  if (size > ALN_MINSZ)
    area += ALN_UNIT * myaln;
  XYZZYP(area, p);
  if (pointer)
    *pointer = area;
  return area;
}

/** \brief
 * allocate space using given 'malloc' function. if present, set
 * pointer and pointer-sized integer offset from base address. offset
 * is in units of the data type length.
 */
static char *
I8(__alloc04)(__NELEM_T nelem, dtype kind, size_t len,
               __STAT_T *stat, char **pointer, __POINT_T *offset,
               char *base, int check, void *(*mallocfn)(size_t),
               size_t align, char *errmsg, int errlen)
{
  ALLO_HDR *p;
  char *area;
  size_t need, size, slop, sizeof_hdr;
  __POINT_T off;
  char msg[80];
  char *p_env;

  if (!ISPRESENT(stat))
    stat = NULL;
  if (!ISPRESENT(pointer))
    pointer = NULL;
  if (!ISPRESENT(offset))
    offset = NULL;
  if (!ISPRESENT(errmsg))
    errmsg = NULL;

  if (*pointer && I8(__fort_allocated)(*pointer)
      && ISPRESENT(stat) && *stat == 2) {
    int i;
    const char *mp = "array already allocated";
    MP_P_STDIO;
    for (i = 0; i < errlen; i++)
      errmsg[i] = (*mp ? *mp++ : ' ');
    MP_V_STDIO;
  }

#if (defined(_WIN64))
#define ALN_LARGE
#else
#undef ALN_LARGE
#endif
/* always use the larger padding   - used be for just win, but
 * now it makes a bug difference on linux with newer (since Jul 2007)
 * processor.
 */
#define ALN_LARGE

  size_t ALN_MINSZ = 128000;
  size_t ALN_UNIT = 64;
  size_t ALN_MAXADJ = 4096;

#define ALN_THRESH (ALN_MAXADJ / ALN_UNIT)
  static size_t aln_n = 0;
  static int env_checked = 0;
  size_t myaln;

  sizeof_hdr = AUTOASZ;

  if (!env_checked) {
    env_checked = 1;

    p_env = getenv("F90_ALN_MINSZ");
    if (p_env != NULL)
      ALN_MINSZ = atol(p_env);

    p_env = getenv("F90_ALN_UNIT");
    if (p_env != NULL)
      ALN_UNIT = atol(p_env);

    p_env = getenv("F90_ALN_MAXADJ");
    if (p_env != NULL)
      ALN_MAXADJ = atol(p_env);
  }

  ALLHDR();
  need = (nelem <= 0) ? 0 : (size_t)nelem * len;
  if (!need) /* should this be size < ASZ ?? */
    need = ASZ;
  slop = 0;
  /* SLOPPY COMMENT:
   * here, we add some slop so we can align the eventual
   * allocated address after adding the size of the ALLO_HDR.
   * We are going to align to ASZ.  Since we know that the malloc
   * function returns something at least 8-byte aligned, we only
   * should have to add (ASZ-8) slop.
   * Actually, we shouldn't even have to do this if we know
   * that the malloc function returns aligned results and
   * skipping ALLO_HDR will keep it aligned */
  if (nelem > 1 || need > 2 * sizeof_hdr)
    slop = (offset && len > (ASZ - 8)) ? len : (ASZ - 8);
  size = (sizeof_hdr + slop + need + ASZ - 1) & ~(ASZ - 1);
  if (size > ALN_MINSZ) {
    myaln = aln_n;
    size += ALN_UNIT * myaln;
    if (aln_n < ALN_THRESH)
      aln_n++;
    else
      aln_n = 0;
  }
  p = (size < need) ? NULL : (ALLO_HDR *)mallocfn(size);
  if (p == NULL) {
    if (pointer)
      *pointer = NULL;
    if (offset)
      *offset = 1;
    if (stat) {
      *stat = 1;
      if (errmsg) {
        int i;
        char *mp;
        MP_P_STDIO;
        sprintf(msg, "Not enough memory to allocate %lu bytes", need);
        mp = msg;
        for (i = 0; i < errlen; i++)
          errmsg[i] = (*mp ? *mp++ : ' ');
        MP_V_STDIO;
      }
      return NULL;
    }
    MP_P_STDIO;
    sprintf(msg, "ALLOCATE: %lu bytes requested; not enough memory", need);
    MP_V_STDIO;
    __fort_abort(msg);
  }
  area = (char *)p + sizeof_hdr;
  if (offset) {
    off = area - base + len - 1;
    if (kind != __STR && kind != __DERIVED)
      off >>= GET_DIST_SHIFTS(kind);
    else
      off /= len;

    *offset = off + 1;
    area = base + off * len;
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d alloc: need %lu size %lu p %p area %p end %p"
             " base %p offset %ld len %lu\n",
             GET_DIST_LCPU, need, size, p, area, (char *)p + size - 1, base,
             *offset, len);
#endif
  } else {
    /* see SLOPPY COMMENT above */
    if (nelem > 1 || need > 2 * sizeof_hdr)
      area = ALIGNP(area, ASZ);
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d alloc: need %lu size %lu p %p area %p end %p\n",
             GET_DIST_LCPU, need, size, p, area, (char *)p + size - 1);
#endif
  }
  if (size > ALN_MINSZ)
    area += ALN_UNIT * myaln;
  XYZZYP(area, p);
  if (pointer)
    *pointer = area;
  return area;
}

/** \brief
 * allocate space using given 'malloc' function. if present, set
 * pointer and pointer-sized integer offset from base address. offset
 * is in units of the data type length.
 */
char *
I8(__fort_kalloc)(__INT8_T nelem, dtype kind, size_t len, __STAT_T *stat,
                  char **pointer, __POINT_T *offset, char *base, int check,
                  void *(*mallocfn)(size_t))
{
  ALLO_HDR *p;
  char *area;
  size_t need, size, slop, sizeof_hdr;
  __POINT_T off;
  char msg[80];

  ALLHDR();

  if (!ISPRESENT(stat))
    stat = NULL;
  if (!ISPRESENT(pointer))
    pointer = NULL;
  if (!ISPRESENT(offset))
    offset = NULL;

  sizeof_hdr = AUTOASZ;

  need = (nelem <= 0) ? 0 : (size_t)nelem * len;
  /* see SLOPPY COMMENT above */
  slop = 0;
  if (nelem > 1 || need > 2 * sizeof_hdr)
    slop = (offset && len > (ASZ / 2)) ? len : (ASZ / 2);
  size = (sizeof_hdr + slop + need + ASZ - 1) & ~(ASZ - 1);
  MP_P(sem);
  p = (size < need) ? NULL : (ALLO_HDR *)mallocfn(size);
  MP_V(sem);
  if (p == NULL) {
    if (pointer)
      *pointer = NULL;
    if (offset)
      *offset = 1;
    if (stat) {
      *stat = 1;
      return NULL;
    }
    MP_P_STDIO;
    sprintf(msg, "ALLOCATE: %lu bytes requested; not enough memory", need);
    MP_V_STDIO;
    __fort_abort(msg);
  }
  if (stat)
    *stat = 0;
  area = (char *)p + sizeof_hdr;
  if (offset) {
    off = area - base + len - 1;
    if (kind != __STR && kind != __DERIVED)
      off >>= GET_DIST_SHIFTS(kind);
    else
      off /= len;

    *offset = off + 1;
    area = base + off * len;
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d alloc: need %lu size %lu p %p area %p end %p"
             " base %p offset %ld len %lu\n",
             GET_DIST_LCPU, need, size, p, area, (char *)p + size - 1, base,
             *offset, len);
#endif
  } else {
    /* see SLOPPY COMMENT above */
    if (nelem > 1 || need > 2 * sizeof_hdr)
      area = ALIGNP(area, ASZ);
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d alloc: need %lu size %lu p %p area %p end %p\n",
             GET_DIST_LCPU, need, size, p, area, (char *)p + size - 1);
#endif
  }
  if (pointer)
    *pointer = area;
  return area;
}

/** \brief
 * Is array allocated?
 */
int
I8(__fort_allocated)(char *area)
{
  ALLHDR();

  if (area) {
    return 1;
  }
  return 0;
}

__LOG_T
I8(ftn_allocated)(char *area)
{
  return I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

__LOG_T
ENTF90(ALLOCATED, allocated)(char *area)
{
  return I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

__LOG_T
ENTF90(ALLOCATED2, allocated2)(char *area)
{
  return I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

__LOG_T
ENTF90(ALLOCATED_LHS, allocated_lhs)(char *area)
{
  return I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

__LOG_T
ENTF90(ALLOCATED_LHS2, allocated_lhs2)(char *area)
{
  return I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

/* -i8 variant of ALLOCATED */
__LOG8_T
ENTF90(KALLOCATED, kallocated)(char *area)
{
  return (__LOG8_T)I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

/* -i8 variant of ALLOCATED2 */
__LOG8_T
ENTF90(KALLOCATED2, kallocated2)(char *area)
{
  return (__LOG8_T)I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

/* -i8 variant of ALLOCATED_LHS */
__LOG8_T
ENTF90(KALLOCATED_LHS, kallocated_lhs)(char *area)
{
  return (__LOG8_T)I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

/* -i8 variant of ALLOCATED_LHS2 */
__LOG8_T
ENTF90(KALLOCATED_LHS2, kallocated_lhs2)(char *area)
{
  return (__LOG8_T)I8(__fort_allocated)(area) ? GET_DIST_TRUE_LOG : 0;
}

/** \brief
 * F77 allocate statement -- don't check allocated status
 */
char *
I8(ftn_allocate)(int size, __STAT_T *stat)
{
  return I8(__fort_alloc)(size, __CHAR, 1, stat, NULL, NULL, NULL, 0,
                         LOCAL_MODE ? __fort_malloc_without_abort
                                    : __fort_gmalloc_without_abort);
}

/** \brief
 * F77 allocate statement -- check allocated status
 */
char *
I8(ftn_alloc)(int size, __STAT_T *stat, char *area)
{
  return I8(__fort_alloc)(size, __CHAR, 1, stat, NULL, NULL, area, 1,
                         LOCAL_MODE ? __fort_malloc_without_abort
                                    : __fort_gmalloc_without_abort);
}

typedef struct {
  __INT8_T len;
  int valid; /* 0 == not allocated; 1 == allocated in use;
              * -1 = allocated ready for reuse
              */
  char *pointer;
} SAL;

/*
 * Possibly use TLS to manange the 'savedalloc' lists if enabled; obviates the
 * need for a semaphore.
 */
#define TLS_DECL
MP_SEMAPHORE(static, sem_allo);
#define MP_P_ALLO _mp_p(&sem_allo)
#define MP_V_ALLO _mp_v(&sem_allo)

/*
 * savedalloc is currently disabled. Using semaphores causes a significant hit
 * to performance when the app uses OpenMP; using TLS  helps but there is still
 * noticeable hit for cam & pop.  A better approach would be to improve the
 * compiler to use the LHS as the temporary for expressions in the RHS.  The
 * fundamental problem (e.g. fatigue) is that we allocate a temp for the RHS
 * for certain assignments -- better analysis to use the LHS is far more
 * general.
 */
/* use this initialization to completely disable the allocation optimization */
static SAL savedalloc = {0, -99, (char *)0};

#if !defined(DESC_I8)
/** \brief
 * F90 allocate statement -- check allocated status
 */
void
__f90_allo_term(void)
{
  if (savedalloc.valid != -99) {
    MP_P_ALLO;
    if (savedalloc.valid == -1) {
      char *area;
      int memaligned;
      area = savedalloc.pointer;
      savedalloc.valid = 0;
      savedalloc.pointer = NULL;
      savedalloc.len = 0;
      memaligned = 0;
      if (!memaligned)
        __fort_free(XYZZY(area));
    }
    MP_V_ALLO;
  }
}
#endif

static void
save_alloc(__POINT_T nelem, __INT_T len, char **pointer)
{
  if (savedalloc.valid >= 0) {
    /* we aren't saving some old space here */
    __INT8_T l;
    l = nelem;
    MP_P_ALLO;
    if (savedalloc.valid >= 0 && l > 0) { /* check still valid */
      /* now, save the most recently allocated space for later reuse */
      l *= (len);
      savedalloc.valid = 1;
      savedalloc.pointer = *pointer;
      savedalloc.len = l;
    }
    MP_V_ALLO;
  }
}

static void *
use_alloc(__POINT_T nelem, __INT_T len)
{
  if (savedalloc.valid == -1) {
    void *salp;
    /* for allocate/free in a loop, see if we have a recently
     * allocated and freed space that we can immediately reuse;
     * don't do this with stat argument
     */
    __INT8_T l;
    l = nelem;
    if (l > 0)
      l *= (len);
    /* l holds the length, see fi the space is long enough, but not too long */
    MP_P_ALLO;
    if (savedalloc.valid != -1) { /* still valid (in case of threading) */
      MP_V_ALLO;
    } else {
      if (l <= savedalloc.len && l > (savedalloc.len >> 1)) {
        salp = savedalloc.pointer;
        savedalloc.valid = 1; /* in use */
        /* success, it's just long enough, use it */
        MP_V_ALLO;
        return salp;
      } else {
        char *pp;
        pp = savedalloc.pointer;
        savedalloc.valid = 0; /* not allocated */
        savedalloc.pointer = NULL;
        savedalloc.len = 0;
        MP_V_ALLO; /* get out of the critical section */
        /* failure; just free the space we had saved */
        (void)I8(__fort_dealloc)(pp, (__STAT_T *)(ENTCOMN(0, 0)), __fort_free);
      }
    }
  }
  return NULL;
}

static void *
reuse_alloc(__STAT_T *stat, char *area)
{
  if (savedalloc.pointer == area && savedalloc.pointer != NULL) {
    MP_P_ALLO;
    /* now test again inside the critical region */
    if (savedalloc.pointer == area && savedalloc.pointer != NULL) {
      if (!ISPRESENT(stat)) {
        /* if this was the 'recently allocated' space, mark it as ready for
         * reuse */
        savedalloc.valid = -1; /* ready for reuse */
        MP_V_ALLO;
        return area;
      }
      savedalloc.valid = 0; /* will be free-ed */
      savedalloc.pointer = NULL;
      savedalloc.len = 0;
    }
    MP_V_ALLO;
  }
  return NULL;
}

void
ENTF90(ALLOCA, alloca)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                       __STAT_T *stat, char **pointer, __POINT_T *offset,
                       DCHAR(base) DCLEN64(base))
{

  ALLHDR();

  if (!ISPRESENT(stat)) {
    void *salp;
    salp = use_alloc(*nelem, *len);
    if (salp) {
      *pointer = salp;
      return;
    }
  }
  (void)I8(__fort_alloc)(
      *nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset, CADR(base), 1,
      LOCAL_MODE ? __fort_malloc_without_abort : __fort_gmalloc_without_abort);
  if (!ISPRESENT(stat)) {
    save_alloc(*nelem, *len, pointer);
  }
}

/* 32 bit CLEN version */
void
ENTF90(ALLOC, alloc)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                     __STAT_T *stat, char **pointer, __POINT_T *offset,
                     DCHAR(base) DCLEN(base))
{
  ENTF90(ALLOCA, alloca)(nelem, kind, len, stat, pointer, offset, CADR(base),
                         (__CLEN_T)CLEN(base));
}

void
ENTF90(ALLOC03A, alloc03a)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, DCHAR(errmsg) DCLEN64(errmsg))
{
  ALLHDR();

  if (ISPRESENT(stat) && *firsttime)
    *stat = 0;

  if (!ISPRESENT(stat)) {
    void *salp;
    salp = use_alloc(*nelem, *len);
    if (salp) {
      *pointer = salp;
      return;
    }
  }
  (void)I8(__alloc04)(*nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset,
                      0, 1, LOCAL_MODE ? __fort_malloc_without_abort
                                       : __fort_gmalloc_without_abort,
                      0, CADR(errmsg), CLEN(errmsg));
  if (!ISPRESENT(stat)) {
    save_alloc(*nelem, *len, pointer);
  }
}

/* 32 bit CLEN version */
void
ENTF90(ALLOC03, alloc03)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(ALLOC03A, alloc03a)(nelem, kind, len, stat, pointer, offset,
                             firsttime, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(ALLOC03_CHKA, alloc03_chka)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, DCHAR(errmsg) DCLEN64(errmsg))
{

  if (*pointer && I8(__fort_allocated)(*pointer)) {
    if (ISPRESENT(stat)) {
      *stat = 2;
    } else {
      __fort_abort("ALLOCATE: array already allocated");
    }
  } else if (ISPRESENT(stat) && *firsttime) {
    *stat = 0;
  }
  ENTF90(ALLOC03,alloc03)(nelem, kind, len, stat, pointer, offset,
                            firsttime,CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(ALLOC03_CHK, alloc03_chk)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(ALLOC03_CHKA, alloc03_chka)(nelem, kind, len,
                         stat, pointer, offset,
                         firsttime, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(REALLOC_ARR_IN_IMPLIED_DO, realloc_arr_in_impiled_do)(char **ptr,
                         F90_Desc *ad, F90_Desc *dd)
{
  int i, total_extent;
  char *tmp = NULL;
  __NELEM_T old_size;

  if (F90_LSIZE_G(dd) * F90_LEN_G(dd) <= 0)
    return; /* no need to realloc */

  old_size = F90_LSIZE_G(ad) * F90_LEN_G(ad);
  total_extent = 1;
  for (i = 0; i < F90_RANK_G(dd) ; i++) {
    total_extent *= F90_DIM_EXTENT_G(dd, i);
  }

  F90_LSIZE_G(ad) += F90_LSIZE_G(dd);
  F90_GSIZE_G(ad) += F90_GSIZE_G(dd);

  ad->dim[0].extent += total_extent;

  (void)I8(__fort_allocate)(F90_LSIZE_G(ad), F90_KIND_G(ad), F90_LEN_G(ad), 0,
                            &tmp, 0);
  if (old_size > 0)
    __fort_bcopy(tmp, *ptr, old_size);

  I8(__fort_deallocate)(*ptr);
  *ptr = tmp;
}

void
ENTF90(ALLOC04A, alloc04a)(__NELEM_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, __NELEM_T *align,
                         DCHAR(errmsg) DCLEN64(errmsg))
{
  ALLHDR();

  if (ISPRESENT(stat) && *firsttime && *stat != 2)
    *stat = 0;

  if (!ISPRESENT(stat) && !*align) {
    void *salp;
    salp = use_alloc(*nelem, *len);
    if (salp) {
      *pointer = salp;
      return;
    }
  }
  (void)I8(__alloc04)(*nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset,
                      0, 1, LOCAL_MODE ? __fort_malloc_without_abort
                                       : __fort_gmalloc_without_abort,
                      *align, CADR(errmsg), CLEN(errmsg));
  if (!ISPRESENT(stat)) {
    save_alloc(*nelem, *len, pointer);
  }
}

/* 32 bit CLEN version */
void
ENTF90(ALLOC04, alloc04)(__NELEM_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, __NELEM_T *align,
                         DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(ALLOC04A, alloc04a)(nelem, kind, len, stat, pointer, offset, firsttime, 
			   align, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(ALLOC04_CHKA, alloc04_chka)(__NELEM_T *nelem, __INT_T *kind,
                                 __INT_T *len, __STAT_T *stat,
                                 char **pointer, __POINT_T *offset,
                                 __INT_T *firsttime, __NELEM_T *align,
                                 DCHAR(errmsg) DCLEN64(errmsg))
{

  if (*pointer && I8(__fort_allocated)(*pointer)) {
    if (ISPRESENT(stat)) {
      *stat = 2;
    } else {
      __fort_abort("ALLOCATE: array already allocated");
    }
  } else if (ISPRESENT(stat) && *firsttime) {
    *stat = 0;
  }
  ENTF90(ALLOC04,alloc04)(nelem, kind, len, stat, pointer, offset, firsttime,
           align, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(ALLOC04_CHK, alloc04_chk)(__NELEM_T *nelem, __INT_T *kind,
                                 __INT_T *len, __STAT_T *stat,
                                 char **pointer, __POINT_T *offset,
                                 __INT_T *firsttime, __NELEM_T *align,
                                 DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(ALLOC04_CHKA, alloc04_chka)(nelem, kind, len, stat, pointer, offset,
                                     firsttime, align, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(KALLOC, kalloc)(__INT8_T *nelem, __INT_T *kind, __INT_T *len,
                       __STAT_T *stat, char **pointer, __POINT_T *offset,
                       DCHAR(base) DCLEN(base))
{
  if (!ISPRESENT(stat)) {
    void *salp;
    salp = use_alloc(*nelem, *len);
    if (salp) {
      *pointer = salp;
      return;
    }
  }
  (void)I8(__fort_kalloc)(
      *nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset, CADR(base), 1,
      LOCAL_MODE ? __fort_malloc_without_abort : __fort_gmalloc_without_abort);
  if (!ISPRESENT(stat)) {
    save_alloc(*nelem, *len, pointer);
  }
}

void
ENTF90(CALLOC, calloc)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                       __STAT_T *stat, char **pointer, __POINT_T *offset,
                       DCHAR(base) DCLEN(base))
{
  (void)I8(__fort_alloc)(
      *nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset, CADR(base), 1,
      LOCAL_MODE ? __fort_calloc_without_abort : __fort_gcalloc_without_abort);
}

void
ENTF90(CALLOC03A, calloc03a)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                           __STAT_T *stat, char **pointer,
                           __POINT_T *offset, __INT_T *firsttime,
                           DCHAR(errmsg) DCLEN64(errmsg))
{
  if (ISPRESENT(stat) && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(*nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset,
                      0, 1, LOCAL_MODE ? __fort_calloc_without_abort
                                       : __fort_gcalloc_without_abort,
                      0, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(CALLOC03, calloc03)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                           __STAT_T *stat, char **pointer,
                           __POINT_T *offset, __INT_T *firsttime,
                           DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(CALLOC03A, calloc03a)(nelem, kind, len,
                           stat, pointer,
                           offset, firsttime,
                           CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(CALLOC04A, calloc04a)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                           __STAT_T *stat, char **pointer,
                           __POINT_T *offset, __INT_T *firsttime,
                           __NELEM_T *align, DCHAR(errmsg) DCLEN64(errmsg))
{
  if (ISPRESENT(stat) && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(*nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset,
                      0, 1, LOCAL_MODE ? __fort_calloc_without_abort
                                       : __fort_gcalloc_without_abort,
                      *align, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(CALLOC04, calloc04)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                           __STAT_T *stat, char **pointer,
                           __POINT_T *offset, __INT_T *firsttime,
                           __NELEM_T *align, DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(CALLOC04A, calloc04a)(nelem, kind, len,
                           stat, pointer,
                           offset, firsttime,
                           align, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(KCALLOC, kcalloc)(__INT8_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         DCHAR(base) DCLEN(base))
{
  (void)I8(__fort_kalloc)(
      *nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset, CADR(base), 1,
      LOCAL_MODE ? __fort_calloc_without_abort : __fort_gcalloc_without_abort);
}

/** \brief
 * F90 allocate statement -- don't check allocated status
 */
void
ENTF90(PTR_ALLOCA, ptr_alloca)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                             __STAT_T *stat, char **pointer,
                             __POINT_T *offset, DCHAR(base) DCLEN64(base))
{
  (void)I8(__fort_alloc)(
      *nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset, CADR(base), 0,
      LOCAL_MODE ? __fort_malloc_without_abort : __fort_gmalloc_without_abort);
}

/* 32 bit CLEN version */
void
ENTF90(PTR_ALLOC, ptr_alloc)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                             __STAT_T *stat, char **pointer,
                             __POINT_T *offset, DCHAR(base) DCLEN(base))
{
  ENTF90(PTR_ALLOCA, ptr_alloca)(nelem, kind, len,
                             stat, pointer,
                             offset, CADR(base), (__CLEN_T)CLEN(base));
}

/** \brief
 * F90 allocate statement -- don't check allocated status
 */
void
ENTF90(PTR_ALLOC03A, ptr_alloc03a)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, DCHAR(errmsg) DCLEN64(errmsg))
{
  if (ISPRESENT(stat) && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(*nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset,
                      0, 0, LOCAL_MODE ? __fort_malloc_without_abort
                                       : __fort_gmalloc_without_abort,
                      0, CADR(errmsg), CLEN(errmsg));
}
/* 32 bit CLEN version */
void
ENTF90(PTR_ALLOC03, ptr_alloc03)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(PTR_ALLOC03A, ptr_alloc03a)(nelem, kind, len,
                         stat, pointer, offset,
                         firsttime, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(PTR_ALLOC04A, ptr_alloc04a)(__NELEM_T *nelem, __INT_T *kind,
                                 __INT_T *len, __STAT_T *stat,
                                 char **pointer, __POINT_T *offset,
                                 __INT_T *firsttime, __NELEM_T *align,
                                 DCHAR(errmsg) DCLEN64(errmsg))
{
  if (ISPRESENT(stat) && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(*nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset,
                      0, 0, LOCAL_MODE ? __fort_malloc_without_abort
                                       : __fort_gmalloc_without_abort,
                      *align, CADR(errmsg), CLEN(errmsg));
}

void
ENTF90(PTR_ALLOC04, ptr_alloc04)(__NELEM_T *nelem, __INT_T *kind,
                                 __INT_T *len, __STAT_T *stat,
                                 char **pointer, __POINT_T *offset,
                                 __INT_T *firsttime, __NELEM_T *align,
                                 DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(PTR_ALLOC04A, ptr_alloc04a)(nelem, kind,
                                 len, stat,
                                 pointer, offset,
                                 firsttime, align,
                                 CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

/** \brief
 *  Same as ptr_alloc03 above, except it's used with sourced allocation where
 *  the destination or source argument is polymorphic. So, we neeed to get
 *  the size from the source descriptor and allocate the source size if it's
 *  greater than the destination size passed in *len.
 */
void
ENTF90(PTR_SRC_ALLOC03A, ptr_src_alloc03a)(F90_Desc *sd, __INT_T *nelem,
                             __INT_T *kind, __INT_T *len, __STAT_T *stat,
                             char **pointer, __POINT_T *offset,
                             __INT_T *firsttime, DCHAR(errmsg) DCLEN64(errmsg))
{
  __INT_T src_len, max_len;

  src_len = ENTF90(GET_OBJECT_SIZE, get_object_size)(sd);
  if (sd && sd->tag == __DESC && sd->lsize > 1)
    src_len *= sd->lsize;
  else if (nelem && *nelem > 1) {
    src_len *= *nelem;
  }
  max_len = (len && nelem) ? (*len * *nelem) : 0;
  if (max_len < src_len)
    max_len = src_len;

  if (ISPRESENT(stat) && firsttime && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(1, (dtype)*kind, (size_t)max_len, stat, pointer,
                      offset, 0, 0, LOCAL_MODE ? __fort_malloc_without_abort
                                               : __fort_gmalloc_without_abort,
                      0, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(PTR_SRC_ALLOC03, ptr_src_alloc03)(F90_Desc *sd, __INT_T *nelem,
                             __INT_T *kind, __INT_T *len, __STAT_T *stat,
                             char **pointer, __POINT_T *offset,
                             __INT_T *firsttime, DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(PTR_SRC_ALLOC03A, ptr_src_alloc03a)(sd, nelem,
                             kind, len, stat,
                             pointer, offset,
                             firsttime, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(PTR_SRC_CALLOC03A, ptr_src_calloc03a)(F90_Desc *sd, __INT_T *nelem,
                              __INT_T *kind, __INT_T *len, __STAT_T *stat,
                              char **pointer, __POINT_T *offset,
                             __INT_T *firsttime, DCHAR(errmsg) DCLEN64(errmsg))
{
  __INT_T src_len, max_len;

  src_len = ENTF90(GET_OBJECT_SIZE, get_object_size)(sd);
  if (sd && sd->tag == __DESC && sd->lsize > 1) {
    src_len *= sd->lsize;
  } else if (nelem && *nelem > 1) {
    src_len *= *nelem; 
  }
  max_len = (len && nelem) ? (*len * *nelem) : 0;
  if (max_len < src_len)
    max_len = src_len;

  if (ISPRESENT(stat) && firsttime && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(1, (dtype)*kind, (size_t)max_len, stat, pointer,
                      offset, 0, 0, LOCAL_MODE ? __fort_calloc_without_abort
                                               : __fort_gcalloc_without_abort,
                      0, CADR(errmsg), CLEN(errmsg));
}

void
ENTF90(PTR_SRC_CALLOC03, ptr_src_calloc03)(F90_Desc *sd, __INT_T *nelem,
                              __INT_T *kind, __INT_T *len, __STAT_T *stat,
                              char **pointer, __POINT_T *offset,
                             __INT_T *firsttime, DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(PTR_SRC_CALLOC03A, ptr_src_calloc03a)(sd, nelem,
                              kind, len, stat,
                              pointer, offset,
                             firsttime, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(PTR_SRC_ALLOC04A, ptr_src_alloc04a)(F90_Desc *sd, __NELEM_T *nelem,
                             __INT_T *kind, __INT_T *len, __STAT_T *stat,
                             char **pointer, __POINT_T *offset,
                             __INT_T *firsttime, __NELEM_T *align,
                             DCHAR(errmsg) DCLEN64(errmsg))
{
  __INT_T src_len, max_len;

  src_len = ENTF90(GET_OBJECT_SIZE, get_object_size)(sd);
  if (sd && sd->tag == __DESC && sd->lsize > 1)
    src_len *= sd->lsize;
  else if (nelem && *nelem > 1) {
    src_len *= *nelem;
  }
  max_len = (len && nelem) ? (*len * *nelem) : 0;
  if (max_len < src_len)
    max_len = src_len;

  if (ISPRESENT(stat) && firsttime && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(1, (dtype)*kind, (size_t)max_len, stat, pointer,
                      offset, 0, 0, LOCAL_MODE ? __fort_malloc_without_abort
                                               : __fort_gmalloc_without_abort,
                      *align, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(PTR_SRC_ALLOC04, ptr_src_alloc04)(F90_Desc *sd, __NELEM_T *nelem,
                             __INT_T *kind, __INT_T *len, __STAT_T *stat,
                             char **pointer, __POINT_T *offset,
                             __INT_T *firsttime, __NELEM_T *align,
                             DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(PTR_SRC_ALLOC04A, ptr_src_alloc04a)(sd, nelem,
                             kind, len, stat,
                             pointer, offset,
                             firsttime, align,
                             CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(PTR_SRC_CALLOC04A, ptr_src_calloc04a)
                             (F90_Desc *sd, __NELEM_T *nelem, __INT_T *kind,
                              __INT_T *len, __STAT_T *stat, char **pointer,
                              __POINT_T *offset, __INT_T *firsttime,
                              __NELEM_T *align, DCHAR(errmsg) DCLEN64(errmsg))
{
  __INT_T src_len, max_len;

  src_len = ENTF90(GET_OBJECT_SIZE, get_object_size)(sd);
  if (sd && sd->tag == __DESC) {
    if (sd->lsize > 1) {
      src_len *= sd->lsize;
    } else if (!sd->rank && !sd->lsize && !sd->gsize && sd->len > 0 &&
               sd->kind > 0 && sd->kind <= __NTYPES) {
      src_len = sd->len;
    }
  } else if (nelem && *nelem > 1) {
    src_len *= *nelem; 
  }
  max_len = (len && nelem) ? (*len * *nelem) : 0;
  if (max_len < src_len)
    max_len = src_len;

  if (ISPRESENT(stat) && firsttime && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(1, (dtype)*kind, (size_t)max_len, stat, pointer,
                      offset, 0, 0, LOCAL_MODE ? __fort_calloc_without_abort
                                               : __fort_gcalloc_without_abort,
                      *align, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(PTR_SRC_CALLOC04, ptr_src_calloc04)
                             (F90_Desc *sd, __NELEM_T *nelem, __INT_T *kind,
                              __INT_T *len, __STAT_T *stat, char **pointer,
                              __POINT_T *offset, __INT_T *firsttime,
                              __NELEM_T *align, DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(PTR_SRC_CALLOC04A, ptr_src_calloc04a)(sd, nelem, kind,
                              len, stat, pointer,
                              offset, firsttime,
                              align, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

/** \brief
 * 64 bit F90 allocate statement -- don't check allocated status
 */
void
ENTF90(PTR_KALLOC, ptr_kalloc)(__INT8_T *nelem, __INT_T *kind,
                               __INT_T *len, __STAT_T *stat,
                               char **pointer, __POINT_T *offset,
                               DCHAR(base) DCLEN(base))
{
  (void)I8(__fort_kalloc)(
      *nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset, CADR(base), 0,
      LOCAL_MODE ? __fort_malloc_without_abort : __fort_gmalloc_without_abort);
}

void
ENTF90(PTR_CALLOC, ptr_calloc)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                               __STAT_T *stat, char **pointer,
                               __POINT_T *offset, DCHAR(base) DCLEN(base))
{
  (void)I8(__fort_alloc)(
      *nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset, CADR(base), 0,
      LOCAL_MODE ? __fort_calloc_without_abort : __fort_gcalloc_without_abort);
}

void
ENTF90(PTR_CALLOC03A, ptr_calloc03a)
                         (__INT_T *nelem, __INT_T *kind, __INT_T *len,
                          __STAT_T *stat, char **pointer, __POINT_T *offset,
                          __INT_T *firsttime, DCHAR(errmsg) DCLEN64(errmsg))
{
  if (ISPRESENT(stat) && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(*nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset,
                      0, 0, LOCAL_MODE ? __fort_calloc_without_abort
                                       : __fort_gcalloc_without_abort,
                      0, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(PTR_CALLOC03, ptr_calloc03)
                         (__INT_T *nelem, __INT_T *kind, __INT_T *len,
                          __STAT_T *stat, char **pointer, __POINT_T *offset,
                          __INT_T *firsttime, DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(PTR_CALLOC03A, ptr_calloc03a)(nelem, kind, len,
                          stat, pointer, offset,
                          firsttime, CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(PTR_CALLOC04A, ptr_calloc04a)(__NELEM_T *nelem, __INT_T *kind,
                                   __INT_T *len, __STAT_T *stat,
                                   char **pointer, __POINT_T *offset,
                                   __INT_T *firsttime, __NELEM_T *align,
                                   DCHAR(errmsg) DCLEN64(errmsg))
{
  if (ISPRESENT(stat) && *firsttime)
    *stat = 0;

  (void)I8(__alloc04)(*nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset,
                      0, 0, LOCAL_MODE ? __fort_calloc_without_abort
                                       : __fort_gcalloc_without_abort,
                      *align, CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(PTR_CALLOC04, ptr_calloc04)(__NELEM_T *nelem, __INT_T *kind,
                                   __INT_T *len, __STAT_T *stat,
                                   char **pointer, __POINT_T *offset,
                                   __INT_T *firsttime, __NELEM_T *align,
                                   DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(PTR_CALLOC04A, ptr_calloc04a)(nelem, kind,
                                   len, stat,
                                   pointer, offset,
                                   firsttime, align,
                                   CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(PTR_KCALLOC, ptr_kcalloc)(__INT8_T *nelem, __INT_T *kind,
                                 __INT_T *len, __STAT_T *stat,
                                 char **pointer, __POINT_T *offset,
                                 DCHAR(base) DCLEN(base))
{
  (void)I8(__fort_kalloc)(
      *nelem, (dtype)*kind, (size_t)*len, stat, pointer, offset, CADR(base), 0,
      LOCAL_MODE ? __fort_calloc_without_abort : __fort_gcalloc_without_abort);
}

/** \brief
 * Allocate global array (must be same size on all processors), return
 * pointer and pointer-sized integer offset from base address.  offset
 * is in units of the data type length. don't check allocated status
 */
char *
I8(__fort_allocate)(int nelem, dtype kind, size_t len, char *base,
                    char **pointer, __POINT_T *offset)
{
  return I8(__fort_alloc)(nelem, kind, len, NULL, pointer, offset, base, 0,
                         __fort_gmalloc_without_abort);
}

/** \brief
 * Allocate local array (may be different size on each processor),
 * return pointer and pointer-sized integer offset from base address.
 * offset is in units of the data type length. don't check allocated
 * status
 */
char *
I8(__fort_local_allocate)(int nelem, dtype kind, size_t len, char *base,
                          char **pointer, __POINT_T *offset)
{
  return I8(__fort_alloc)(nelem, kind, len, NULL, pointer, offset, base, 0,
                         __fort_malloc_without_abort);
}

/** \brief
 * Allocate global array (must be same size on all processors), return
 * pointer and pointer-sized integer offset from base address.  offset
 * is in units of the data type length. don't check allocated status
 */
char *
I8(__fort_kallocate)(long nelem, dtype kind, size_t len, char *base,
                     char **pointer, __POINT_T *offset)
{
  return I8(__fort_kalloc)(nelem, kind, len, NULL, pointer, offset, base, 0,
                          __fort_gmalloc_without_abort);
}

/** \brief
 * Allocate local array (may be different size on each processor),
 * return pointer and pointer-sized integer offset from base address.
 * offset is in units of the data type length. don't check allocated
 * status
 */

char *
I8(__fort_local_kallocate)(long nelem, dtype kind, size_t len, char *base,
                           char **pointer, __POINT_T *offset)
{
  return I8(__fort_kalloc)(nelem, kind, len, NULL, pointer, offset, base, 0,
                          __fort_malloc_without_abort);
}

/** \brief
 * Deallocate array using given 'free' function
 */
char *
I8(__fort_dealloc)(char *area, __STAT_T *stat, void (*freefn)(void *))
{
  ALLO_HDR *p = NULL;
  char msg[80];

  ALLHDR();

  if (!ISPRESENT(stat))
    stat = NULL;
  if (!ISPRESENT(area))
    area = NULL;
  if (area) {
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d dealloc p %p area %p\n", GET_DIST_LCPU, p, area);
#endif
    freefn(XYZZY(area));
    if (stat)
      *stat = 0;
    return area;
  }
  if (stat)
    *stat = 1;
  else {
    MP_P_STDIO;
    sprintf(msg, "DEALLOCATE: memory at %p not allocated", area);
    MP_V_STDIO;
    __fort_abort(msg);
  }
  return NULL;
}

static char *
I8(__fort_dealloc03)(char *area, __STAT_T *stat, void (*freefn)(void *),
                     char *errmsg, int errlen)
{
  ALLO_HDR *p = NULL;
  char msg[80];

  ALLHDR();

  if (!ISPRESENT(stat))
    stat = NULL;
  if (!ISPRESENT(area))
    area = NULL;
  if (!ISPRESENT(errmsg))
    errmsg = NULL;
  if (area) {
#if defined(DEBUG)
    if (__fort_test & DEBUG_ALLO)
      printf("%d dealloc p %p area %p\n", GET_DIST_LCPU, p, area);
#endif
    freefn(XYZZY(area));
    return area;
  }
  if (stat) {
    *stat = 1;
    if (errmsg) {
      int i;
      char *mp;
      MP_P_STDIO;
      sprintf(msg, "Memory at %p not allocated", area);
      mp = msg;
      for (i = 0; i < errlen; i++)
        errmsg[i] = (*mp ? *mp++ : ' ');
      MP_V_STDIO;
    }
  } else {
    MP_P_STDIO;
    sprintf(msg, "DEALLOCATE: memory at %p not allocated", area);
    MP_V_STDIO;
    __fort_abort(msg);
  }
  return NULL;
}

/** \brief
 * F77 deallocate statement
 */
void
I8(ftn_deallocate)(char *area, __STAT_T *stat)
{
  (void)I8(__fort_dealloc)(area, stat, LOCAL_MODE ? __fort_free : __fort_gfree);
}

char *
I8(ftn_dealloc)(char *area, __STAT_T *stat)
{
  return I8(__fort_dealloc)(area, stat, LOCAL_MODE ? __fort_free : __fort_gfree);
}

/** \brief
 * f90 deallocate statement
 */
void
ENTF90(DEALLOCA, dealloca)(__STAT_T *stat, DCHAR(area) DCLEN64(area))
{
  if (reuse_alloc(stat, CADR(area)))
    return;
  (void)I8(__fort_dealloc)(CADR(area), stat,
                          LOCAL_MODE ? __fort_free : __fort_gfree);
}

/* 32 bit CLEN version */
void
ENTF90(DEALLOC, dealloc)(__STAT_T *stat, DCHAR(area) DCLEN(area))
{
  ENTF90(DEALLOCA, dealloca)(stat, CADR(area), (__CLEN_T)CLEN(area));
}

void
ENTF90(DEALLOC03A, dealloc03a)(__STAT_T *stat, char *area,
                             __INT_T *firsttime,
                             DCHAR(errmsg) DCLEN64(errmsg))
{
  if (ISPRESENT(stat) && *firsttime)
    *stat = 0;
  if (reuse_alloc(stat, area))
    return;
  (void)I8(__fort_dealloc03)(area, stat, LOCAL_MODE ? __fort_free : __fort_gfree,
                            CADR(errmsg), CLEN(errmsg));
}

/* 32 bit CLEN version */
void
ENTF90(DEALLOC03, dealloc03)(__STAT_T *stat, char *area,
                             __INT_T *firsttime,
                             DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(DEALLOC03A, dealloc03a)(stat, area,
                             firsttime,
                             CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(DEALLOC_MBR, dealloc_mbr)(__STAT_T *stat, DCHAR(area) DCLEN(area))
{

  if (I8(__fort_allocated)(CADR(area))) {
    ENTF90(DEALLOC, dealloc)(stat, CADR(area), CLEN(area));
  }
}

void
ENTF90(DEALLOC_MBR03A, dealloc_mbr03a)(__STAT_T *stat, char *area,
                                          __INT_T *firsttime,
                                          DCHAR(errmsg) DCLEN64(errmsg))
{
  if (I8(__fort_allocated)(area)) {
    ENTF90(DEALLOC03,dealloc03)(stat, area, firsttime,
                CADR(errmsg), CLEN(errmsg));
  }
}

/* 32 bit CLEN version */
void
ENTF90(DEALLOC_MBR03, dealloc_mbr03)(__STAT_T *stat, char *area,
                                          __INT_T *firsttime,
                                          DCHAR(errmsg) DCLEN(errmsg))
{
  ENTF90(DEALLOC_MBR03A, dealloc_mbr03a)(stat, area, firsttime,
                                          CADR(errmsg), (__CLEN_T)CLEN(errmsg));
}

void
ENTF90(DEALLOCX, deallocx)(__STAT_T *stat, char **area)
{
  (void)I8(__fort_dealloc)(*area, stat, LOCAL_MODE ? __fort_free : __fort_gfree);
}

/** \brief
 * deallocate global array
 */
void
I8(__fort_deallocate)(char *area)
{
  (void)I8(__fort_dealloc)(area, NULL, __fort_gfree);
}

/** \brief
 * deallocate local array
 */

void
I8(__fort_local_deallocate)(char *area)
{
  (void)I8(__fort_dealloc)(area, NULL, __fort_free);
}

static size_t AUTO_ALN_MINSZ = 128000;
static size_t AUTO_ALN_UNIT = 64;
static size_t AUTO_ALN_MAXADJ = 4096;

static void *
I8(__auto_alloc)(__NELEM_T nelem, __INT_T sz,
                 void *(*mallocroutine)(size_t))
{
  char *p, *area;
  size_t size, need;
  char msg[80];

#define AUTO_ALN_THRESH (AUTO_ALN_MAXADJ / AUTO_ALN_UNIT)
  static size_t aln_n = 0;
  size_t myaln;

  if (nelem > 0)
    need = nelem * sz;
  else
    need = 0;

  size = ((need + (ASZ - 1)) & ~(ASZ - 1)) + AUTOASZ; /* quad-alignment */

  if (size > AUTO_ALN_MINSZ) {
    myaln = aln_n;
    size += AUTO_ALN_UNIT * myaln;
    if (aln_n < AUTO_ALN_THRESH)
      aln_n++;
    else
      aln_n = 0;
  }

  p = (char *)(mallocroutine)(size);
  if (p == NULL) {
    MP_P_STDIO;
    sprintf(msg, "ALLOCATE: %lu bytes requested; not enough memory", need);
    MP_V_STDIO;
    __fort_abort(msg);
  }

  area = (char *)p + AUTOASZ; /* quad-alignment */

  if (size > AUTO_ALN_MINSZ)
    area += AUTO_ALN_UNIT * myaln;

  XYZZYP(area, p);

  return area;
}

#ifndef DESC_I8
/*
 * Simple globally visible by value auto_alloc;
 */
void *
ENTF90(AUTO_ALLOCV, auto_allocv)(__NELEM_T nelem, int sz)
{
  void *p;
  p = I8(__auto_alloc)(nelem, sz, malloc);
  return p;
}
#endif

void *
ENTF90(AUTO_ALLOC, auto_alloc)(__INT_T *nelem, __INT_T *sz)
{
  void *p;

  p = I8(__auto_alloc)(*nelem, *sz, malloc);
  return p;
}

void *
ENTF90(AUTO_ALLOC04, auto_alloc04)(__NELEM_T *nelem, __INT_T *sz)
{
  void *p;

  p = I8(__auto_alloc)(*nelem, *sz, malloc);
  return p;
}

void *
ENTF90(AUTO_CALLOC, auto_calloc)(__INT_T *nelem, __INT_T *sz)
{
  size_t size;
  void *p;

  p = I8(__auto_alloc)(*nelem, *sz, malloc);
  if (p && *nelem > 0) {
    size = *nelem * *sz;
    memset(p, 0, size);
  }

  return p;
}

void *
ENTF90(AUTO_CALLOC04, auto_calloc04)(__NELEM_T *nelem, __INT_T *sz)
{
  size_t size;
  void *p;

  p = I8(__auto_alloc)(*nelem, *sz, malloc);
  if (p && *nelem > 0) {
    size = *nelem * *sz;
    memset(p, 0, size);
  }

  return p;
}

void
ENTF90(AUTO_DEALLOC, auto_dealloc)(void *area) { free(XYZZY(area)); }

#if defined(DEBUG)
void
ENTRY(__FTN_ALLOC_DUMP, __ftn_alloc_dump)()
{
  ALLO_HDR *p;
  int lcpu, n;

  ALLHDR();

  lcpu = GET_DIST_LCPU;
  printf("%d list of allocated blocks:\n", lcpu);
  for (n = 0; n < num_hdrs; n++) {
    for (p = allo_list[n].next; p != NULL; p = p->next) {
      printf("%d    block: %p, area: %p\n", lcpu, p, p->area);
    }
  }
}
#endif
