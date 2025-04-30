/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * Fortran character support routines
 */

#include <stdint.h>
#include <stdio.h>
#include "global.h"
#include "stdarg.h"
#include "enames.h"
#include <string.h>
#include "llcrit.h"
#include "mpalloc.h"

#ifndef NULL
#define NULL (void *)0
#endif


/* ***********************************************************************/
/** \brief
 * Copies a series of character strings into another.
 * 
 * This function is used
 * to implement character assignments and concatenations. It pads with
 * blanks if the destination string is longer than the sum of the lengths
 * of the source strings and truncates if the sum of the lengths of the
 * source strings is longer than the destination string.
 *
 * Allow the target of the concatenation to appear in its right-hand side
 * which is standard f90 and is a common extension to f77.
 *
 * \param    n              number of source strings
 * \param    to             pointer to destination string
 * \param    to_len         length of destination string
 * <pre>
 *  Vargs:
 *    {from           pointer to kth source string
 *     from_len}*     length of kth source string
 * </pre>
 */
/* ***********************************************************************/
void
Ftn_str_copy(int n, char *to, int to_len, ...)
{
  va_list ap;
  char *from;
  int from_len;
  int idx2;
  int cnt;
  typedef struct {
    char *str;
    int len;
    int dyn;
  } SRC_STR;
  SRC_STR *src_p, *qq;
  SRC_STR aa[4]; /* statically allocate for 4 source strings */
  int src_p_allocd;
  char *to_p;
  char *to_end;
  char *from_end;
  int any_allocd;

  if (to_len <= 0) {
    return;
  }
  if (n <= (sizeof(aa) / sizeof(SRC_STR))) {
    qq = src_p = aa;
    src_p_allocd = 0;
  } else {
    qq = src_p = (SRC_STR *)_mp_malloc(sizeof(SRC_STR) * n);
    src_p_allocd = 1;
  }
  va_start(ap, to_len);
#ifdef DEBUG
  printf("to_len = %d\n", to_len);
#endif
  to_end = to - 1;
  any_allocd = idx2 = 0;
  for (cnt = n; cnt > 0; cnt--, qq++) {
    from = va_arg(ap, char *);
    from_len = va_arg(ap, int);
#ifdef DEBUG
    printf("from_len = %d\n", from_len);
#endif
    if (from_len < 0)
      from_len = 0;
    qq->str = from;
    qq->len = from_len;
    qq->dyn = 0;
    to_end += from_len;
    from_end = from + (from_len - 1);
    if ((from >= to && from <= to_end) ||
        (from_end >= to && from_end <= to_end))
      if (from_len) {
        qq->str = _mp_malloc(from_len);
        memcpy(qq->str, from, from_len);
        qq->dyn = 1;
#ifdef DEBUG
        printf("string %d overlaps\n", n - cnt);
        printf("mallocd %08x\n", qq->str);
#endif
        any_allocd = 1;
      }
    idx2 += from_len;
    if (idx2 >= to_len)
      break;
  }
  va_end(ap);

  qq = src_p;
  to_p = to;
  to_end = to + to_len; /* position after the end of the destination */
  for (cnt = n; cnt > 0; cnt--, qq++) {
    from = qq->str;
    for (from_len = qq->len; from_len > 0; from_len--, from++) {
      *to_p++ = *from;
      if (to_p == to_end)
        goto exit_return;
#ifdef DEBUG
      printf("from_char = %c\n", *from);
#endif
    }
  }

exit_return:
  while (to_p < to_end) { /* remember, to_end is 1 after end */
    /* blank fill to right */
    *to_p++ = ' ';
  }

  if (any_allocd) {
    idx2 = 0;
    qq = src_p;
    for (cnt = n; cnt > 0; cnt--, qq++) {
      if (qq->dyn) {
        _mp_free(qq->str);
#ifdef DEBUG
        printf("freed   %08x\n", qq->str);
#endif
      }
      idx2 += qq->len;
      if (idx2 >= to_len)
        break;
    }
  }

  if (src_p_allocd)
    _mp_free(src_p);
}

/** \brief single source, no overlap */
void
Ftn_str_cpy1(char *to, int to_len, char *from, int from_len)
{
  char *to_p, *to_end;

  if (to_len <= 0) {
    return;
  }
  if (from_len < 0)
    from_len = 0;
  if (to_len <= from_len) {
    memcpy(to, from, to_len);
    return;
  }
  memcpy(to, from, from_len);
  /*memset(to+from_len, ' ', to_len - from_len);*/
  to_p = to + from_len;
  to_end = to + to_len;
  while (to_p < to_end) { /* remember, to_end is 1 after end */
    /* blank fill to right */
    *to_p++ = ' ';
  }
  return;
}


/* ***********************************************************************/
/** \brief
 * Utility routines to allocatespace for character expressions
 * whose lengths are known only at run-time.
 * The compiler creates a variable to locate the list of blocks allocated
 * during a subprogram. This variable is initialized to NULL upon entry to
 * the subprogram.  When the subprogram exits, all of the blocks are freed.
 * Each block of space consists of n 'words':
 * - ++  first word        - pointer to the next allocated block,
 * - ++  remaining word(s) - space for the character data.
 *
 * \param     size - number of bytes needed,
 * \param     hdr  - pointer to the compiler-created variable locating the
 *            list of allocated blocks. Ftn_str_malloc updates this  variable.
 * \returns  returns a pointer to the space after the 'next pointer'.
 *
 * Note that KANJI versions are unneeded since the compiler just calls
 * Ftn_str_malloc() with an adjusted length.
 */
/* ***********************************************************************/
char **
Ftn_str_malloc(int size, char ***hdr)
{
  int nbytes;
  char **p, **q;

/*
 * round request to the size of a pointer & also accommodate a 'next'
 * pointer
 */
#define PTRSZ sizeof(char *)
  nbytes = ((size + PTRSZ - 1) / PTRSZ) * PTRSZ + PTRSZ;
  p = (char **)_mp_malloc(nbytes);
  if (p == NULL) {
    MP_P_STDIO;
    fprintf(__io_stderr(),
            "FTN-F-STR_MALLOC  unable to allocate area of %d bytes\n", size);
    MP_V_STDIO;
    Ftn_exit(1);
  }
  q = *hdr;
  *p = (char *)q; /* link this block to the blocks already allocated */
  *hdr = p;       /* update the list pointer */
  return p + 1;
}

/* ***********************************************************************/
/** \brief
 * Utility routine to deallocate space for character expressions
 * whose lengths are known only at run-time.
 *
 * The compiler creates a variable to locate the list of blocks allocated
 * during a subprogram. This variable is initialized to NULL upon entry to
 * the subprogram.  When the subprogram exits, all of the blocks are freed.
 * Each block of space consists of n 'words':
 * - ++  first word        - pointer to the next allocated block,
 * - ++  remaining word(s) - space for the character data.
 *
 *  \param first - pointer to the compiler-created variable locating the list of
 *                 allocated blocks. Ftn_str_free traverses the list of 
 *                 allocated blocks and frees each block.
 */
/* ***********************************************************************/
void
Ftn_str_free(char **first)
{
  char **p, **next;
  /* traverse the list */
  for (p = first; p != NULL;) {
    next = (char **)(*p);
    _mp_free(p);
    p = next;
  }
}


/* ***********************************************************************/
/** \brief
 * Copies a series of character strings into another.
 * 
 * This function is used
 * to implement character assignments and concatenations. It pads with
 * blanks if the destination string is longer than the sum of the lengths
 * of the source strings and truncates if the sum of the lengths of the
 * source strings is longer than the destination string.
 *
 * Allow the target of the concatenation to appear in its right-hand side
 * which is standard f90 and is a common extension to f77.
 *
 * \param   n              number of source strings
 * \param   to             pointer to destination string
 * \param   to_len         length of destination string
 * <pre>
 * Varargs:
 *   {from           pointer to kth source string
 *   from_len}*      length of kth source string
 * </pre>
 */
/* ***********************************************************************/
void
Ftn_str_copy_klen(int n, char *to, int64_t to_len, ...)
{
  va_list ap;
  char *from;
  int64_t from_len;
  int idx2;
  int cnt;
  typedef struct {
    char *str;
    int64_t len;
    int dyn;
  } SRC_STR;
  SRC_STR *src_p, *qq;
  SRC_STR aa[4]; /* statically allocate for 4 source strings */
  int src_p_allocd;
  char *to_p;
  char *to_end;
  char *from_end;
  int any_allocd;

  if (to_len <= 0) {
    return;
  }
  if (n <= (sizeof(aa) / sizeof(SRC_STR))) {
    qq = src_p = aa;
    src_p_allocd = 0;
  } else {
    qq = src_p = (SRC_STR *)_mp_malloc((size_t)sizeof(SRC_STR) * n);
    src_p_allocd = 1;
  }
  va_start(ap, to_len);
#ifdef DEBUG
  printf("to_len = %d\n", to_len);
#endif
  to_end = to - 1;
  any_allocd = idx2 = 0;
  for (cnt = n; cnt > 0; cnt--, qq++) {
    from = va_arg(ap, char *);
    from_len = va_arg(ap, int64_t);
#ifdef DEBUG
    printf("from_len = %ld\n", from_len);
#endif
    if (from_len < 0)
      from_len = 0;
    qq->str = from;
    qq->len = from_len;
    qq->dyn = 0;
    to_end += from_len;
    from_end = from + (from_len - 1);
    if ((from >= to && from <= to_end) ||
        (from_end >= to && from_end <= to_end))
      if (from_len) {
        qq->str = _mp_malloc((size_t)from_len);
        memcpy(qq->str, from, (size_t)from_len);
        qq->dyn = 1;
#ifdef DEBUG
        printf("string %d overlaps\n", n - cnt);
        printf("mallocd %08x\n", qq->str);
#endif
        any_allocd = 1;
      }
    idx2 += from_len;
    if (idx2 >= to_len)
      break;
  }
  va_end(ap);

  qq = src_p;
  to_p = to;
  to_end = to + to_len; /* position after the end of the destination */
  for (cnt = n; cnt > 0; cnt--, qq++) {
    from = qq->str;
    for (from_len = qq->len; from_len > 0; from_len--, from++) {
      *to_p++ = *from;
      if (to_p == to_end)
        goto exit_return;
#ifdef DEBUG
      printf("from_char = %c\n", *from);
#endif
    }
  }

exit_return:
  while (to_p < to_end) { /* remember, to_end is 1 after end */
    /* blank fill to right */
    *to_p++ = ' ';
  }

  if (any_allocd) {
    idx2 = 0;
    qq = src_p;
    for (cnt = n; cnt > 0; cnt--, qq++) {
      if (qq->dyn) {
        _mp_free(qq->str);
#ifdef DEBUG
        printf("freed   %08x\n", qq->str);
#endif
      }
      idx2 += qq->len;
      if (idx2 >= to_len)
        break;
    }
  }

  if (src_p_allocd)
    _mp_free(src_p);
}

/** \brief single source, no overlap */
void
Ftn_str_cpy1_klen(char *to, int64_t to_len, char *from, int64_t from_len)
{
  char *to_p, *to_end;

  if (to_len <= 0) {
    return;
  }
  if (from_len < 0)
    from_len = 0;
  if (to_len <= from_len) {
    memcpy(to, from, (size_t)to_len);
    return;
  }
  memcpy(to, from, (size_t)from_len);
  /*memset(to+from_len, ' ', to_len - from_len);*/
  to_p = to + from_len;
  to_end = to + to_len;
  while (to_p < to_end) { /* remember, to_end is 1 after end */
    /* blank fill to right */
    *to_p++ = ' ';
  }
  return;
}

/* ***********************************************************************/
/** \brief
 * Implements the INDEX intrinsic; is an integer function which returns the
 * value according to the INDEX intrinsic.
 *
 * \param a1        pointer to string being searched
 * \param a2        pointer to string being searched for
 * \param a1_len    length of a1
 * \param a2_len    length of a2
 */
/* ***********************************************************************/
int64_t
Ftn_str_index_klen( const unsigned char * const a1,
                    const unsigned char * const a2,
                    int64_t a1_len,
                    int64_t a2_len)
{
  int64_t idx1, idx2;
  int match;
  if (a1_len < 0)
    a1_len = 0;
  if (a2_len < 0)
    a2_len = 0;
  for (idx1 = 0; idx1 < a1_len; idx1++) {
    if (a2_len > (a1_len - idx1))
      return (0);
    match = TRUE;
    for (idx2 = 0; idx2 < a2_len; idx2++) {
      if (a1[idx1 + idx2] != a2[idx2]) {
        match = FALSE;
        break;
      }
    }
    if (match)
      return (idx1 + 1);
  }
  return (0);
}


/* ***********************************************************************/
/** \brief
 * Implements the INDEX intrinsic; is an integer function which returns the
 * value according to the INDEX intrinsic.
 *
 * \param a1        pointer to string being searched
 * \param a2        pointer to string being searched for
 * \param a1_len    length of a1
 * \param a2_len    length of a2
 *
 */
/* ***********************************************************************/
int Ftn_str_index(  const unsigned char * const a1,
                    const unsigned char * const a2,
                    int64_t a1_len,
                    int64_t a2_len)
{
    return Ftn_str_index_klen(a1, a2, a1_len, a2_len);
}

/* ***********************************************************************/
/** \brief
 * Implements realational operators with string operands and the lexical
 * intrinsics. Returns integer value:
 * -  0 => strings are the same
 * - -1 => a1 lexically less than a2
 * -  1 => a1 lexically greater than a2
 * If the strings are of unequal lengths, treats shorter string as if it were
 * padded with blanks.
 *
 * \param   a1      pointer to left hand string
 * \param   a2      pointer to right hand string
 * \param   a1_len  length of left hand string
 * \param   a2_len  length of right hand string
 */
/* ***********************************************************************/

int
Ftn_strcmp_klen(    const unsigned char * const a1,
                    const unsigned char * const a2,
                    int64_t a1_len,
                    int64_t a2_len)
{
  int ret_val, one;
  int64_t llong, lshort;
  int64_t idx1;
  const unsigned char * plong;
  const unsigned char * pshort;

  if (a1_len < 0)
    a1_len = 0;
  if (a2_len < 0)
    a2_len = 0;

  if ((a1_len | a2_len) == 0) return 0;

  if (a1_len == a2_len) {

    /*
     * For newer processors (circa 2018), the cutoff where using the
     * optimized C library memcmp is better than a scalar loop is with
     * input string length greater than 4-5 elements.
     *
     * Mileage might vary based on processor architecture (X86-64,
     * POWER, ARM64, ...).
     *
     * Most likely the cutoff length should be a parameter.
     */

    if (a1_len > 4) {
      ret_val = memcmp(a1, a2, a1_len);
      return ret_val == 0 ? 0 : (ret_val < 0 ? -1 : +1);
    }

    /*
     * Strings equal in length, but short.
     */

    idx1 = 0;
    do {
      if (a1[idx1] != a2[idx1]) {
        return a1[idx1] < a2[idx1] ? -1 : +1;
      }
      idx1++;
    } while (idx1 != a1_len);

    return 0;
  }

  /*
   * Find longer of the two string and setup pointers,lengths, and
   * return status accordingly.
   */

  if (a1_len > a2_len) {
    plong = a1;
    pshort = a2;
    llong = a1_len;
    lshort = a2_len;
    one = +1;
  } else {
    plong = a2;
    pshort = a1;
    llong = a2_len;
    lshort = a1_len;
    one = -1;
  }
/*
 * Alternate version 1 - generated code could be better.
  plong = a1_len > a2_len ? a1 : a2;
  pshort = a1_len > a2_len ? a2 : a1;
  llong = a1_len > a2_len ? a1_len : a2_len;
  lshort = a1_len > a2_len ? a2_len : a1_len;
  one = a1_len > a2_len ? +1 : -1;
 */

/*
 * Alternate version 2 - generated code better, but still tests ret_val too often.
  ret_val = a1_len > a2_len;
  plong = ret_val ? a1 : a2;
  pshort = ret_val ? a2 : a1;
  llong = ret_val ? a1_len : a2_len;
  lshort = ret_val ? a2_len : a1_len;
  one = ret_val ? +1 : -1;
 */

  /*
   * Step 1 - compare the first lshort characters of the two string.
   */

  ret_val = memcmp(plong, pshort, lshort);

  /*
   * Step 2 - possibly a quick exit if comparing just the shorter parts
   * of the strings don't match.
   */

  if (ret_val != 0) {
    return ret_val < 0 ? -one : one;
  }

  /*
   * Step 3 - if the last (llong - lshort) characters of plong are blank ' ',
   * then the strings are equal.  Otherwise compare the first non-blank
   * character to blank.
   */

  idx1 = lshort;
  do {
    if (plong[idx1] != ' ') {
      return plong[idx1] < ' ' ? -one : one;
    }
    ++idx1;
  } while (idx1 < llong);
  return 0;
}

/* ***********************************************************************/
/** \brief
 * Implements realational operators with string operands and the lexical
 * intrinsics. Returns integer value:
 * -  0 => strings are the same
 * - -1 => a1 lexically less than a2
 * -  1 => a1 lexically greater than a2
 * If the strings are of unequal lengths, treats shorter string as if it were
 * padded with blanks.
 *
 * \param   a1      pointer to left hand string
 * \param   a2      pointer to right hand string
 * \param   a1_len  length of left hand string
 * \param   a2_len  length of right hand string
 */
/* ***********************************************************************/

int
Ftn_strcmp( const unsigned char * const a1,
            const unsigned char * const a2,
            int a1_len,
            int a2_len)
{
  return Ftn_strcmp_klen(a1, a2, a1_len, a2_len);
}

/* ***********************************************************************/
/** \brief
 * Utility routine to allocate space for character expressions
 * whose lengths are known only at run-time.
 *
 * The compiler creates a variable to locate the list of blocks allocated
 * during a subprogram. This variable is initialized to NULL upon entry to
 * the subprogram.  When the subprogram exits, all of the blocks are freed.
 * Each block of space consists of n 'words':
 * - ++  first word        - pointer to the next allocated block,
 * - ++  remaining word(s) - space for the character data.
 *
 * \param     size - number of bytes needed,
 * \param     hdr  - pointer to the compiler-created variable locating the
 *             list of allocated blocks. Ftn_str_malloc updates this
 *             variable.
 * \return  returns a pointer to the space after the 'next pointer'.
 *
 * void  Ftn_str_free(char ***hdr)
 *      hdr  - pointer to the compiler-created variable locating the list of
 *             allocated blocks. Ftn_str_free traverses the list of allocated
 *             blocks and frees each block.
 *
 * Note that KANJI versions are unneeded since the compiler just calls
 * Ftn_str_malloc() with an adjusted length.
 */
/* ***********************************************************************/
char **
Ftn_str_malloc_klen(int64_t size, char ***hdr)
{
  int64_t nbytes;
  char **p, **q;

/*
 * round request to the size of a pointer & also accommodate a 'next'
 * pointer
 */
#define PTRSZ sizeof(char *)
  nbytes = ((size + PTRSZ - 1) / PTRSZ) * PTRSZ + PTRSZ;
  p = (char **)_mp_malloc((size_t)nbytes);
  if (p == NULL) {
    MP_P_STDIO;
    fprintf(__io_stderr(),
            "FTN-F-STR_MALLOC  unable to allocate area of %ld bytes\n", size);
    MP_V_STDIO;
    Ftn_exit(1);
  }
  q = *hdr;
  *p = (char *)q; /* link this block to the blocks already allocated */
  *hdr = p;       /* update the list pointer */
  return p + 1;
}
