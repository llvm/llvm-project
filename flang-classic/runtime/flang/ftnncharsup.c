/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* fortran support routines for Kanji (NCHARACTER) strings */

#include "stdarg.h"
#include "enames.h"
#define TRUE 1
#define FALSE 0

#ifndef NULL
#define NULL (void *)0
#endif

typedef unsigned short WCHAR;
#define BLANK 0xA1A1 /* Kanji blank */

/* ***********************************************************************/
/** \brief
 * Copies a series of character strings into another. This function is used
 * to implement character assignments and concatenations. It pads with
 * blanks if the destination string is longer than the sum of the lengths
 * of the source strings and truncates if the sum of the lengths of the
 * source strings is longer than the destination string
 *
 * \param n  number of source strings
 * \param to  pointer to destination string
 * \param to_len  length of destination string
 * <pre>
 * Varargs:
 *   {from          pointer to kth source string
 *    from_leno}*   length of kth source string
 * </pre>
 */
/* ***********************************************************************/
void
Ftn_nstr_copy(int n, WCHAR *to, int to_len, ...)
{
  va_list ap;
  WCHAR *from;
  int from_len;
  int i, j, k;

  va_start(ap, to_len);
  j = 0;
  for (i = 0; i < n; i++) {
    from = va_arg(ap, WCHAR *);
    from_len = va_arg(ap, int);
    for (k = 0; k < from_len; k++) {
      if (j == to_len)
        goto exit_return;
      to[j++] = from[k];
    }
  }
exit_return:
  va_end(ap);

  while (j < to_len)
    /* blank fill to right */
    to[j++] = BLANK;
}

/* ***********************************************************************/
/** \brief
 * Implements the INDEX intrinsic; is an integer function which returns the
 * value according to the INDEX intrinsic.
 */
/* ***********************************************************************/
int Ftn_nstr_index(WCHAR *a1,  /* pointer to string being searched */
                   WCHAR *a2,  /* pointer to string being searched for */
                   int a1_len, /* length of a1 */
                   int a2_len) /* length of a2 */
{
  int i, j, match;

  for (i = 0; i < a1_len; i++) {
    if (a2_len > (a1_len - i))
      return 0;
    match = TRUE;
    for (j = 0; j < a2_len; j++) {
      if (a1[i + j] != a2[j]) {
        match = FALSE;
        break;
      }
    }
    if (match)
      return (i + 1);
  }
  return 0;
}

/* ***********************************************************************/
/** \brief
 * Implements relational operators with string operands and the lexical
 * intrinsics. Returns integer value:
 * -  0 => strings are the same
 * - -1 => a1 lexically less than a2
 * -  1 => a1 lexically greater than a2
 * If the strings are of unequal lengths, treats shorter string as if it were
 * padded with blanks.
 */
/* ***********************************************************************/
int Ftn_nstrcmp(WCHAR *a1,  /* first string to be compared */
                WCHAR *a2,  /* second string to be compared */
                int a1_len, /* length of a1 */
                int a2_len) /* length of a2 */
{
  int i, k;

  /*  set k to minimum length of a1 and a2: */
  (a1_len > a2_len ? (k = a2_len) : (k = a1_len));

  /*  check first k characters: */
  for (i = 0; i < k; i++)
    if (a1[i] != a2[i]) {
      if (a1[i] < a2[i])
        return -1;
      return 1;
    }

  if (a1_len == a2_len)
    return 0; /* strings identical */

  /*  implicitly pad a2 with blanks if shorter than a1: */

  while (a1_len > a2_len)
    if (a1[a2_len++] != BLANK) {
      if (a1[a2_len - 1] < BLANK)
        return -1;
      return 1;
    }

  /*  implicitly pad a1 with blanks if shorter than a2: */

  while (a2_len > a1_len)
    if (a2[a1_len++] != BLANK) {
      if (a2[a1_len - 1] < BLANK)
        return 1;
      return -1;
    }

  return 0;
}

#define __HAVE_LONGLONG_T

#if defined(OSX8664) || defined(TARGET_LLVM_ARM64)
typedef long _LONGLONG_T;
typedef unsigned long _ULONGLONG_T;
#else
typedef long long _LONGLONG_T;
typedef unsigned long long _ULONGLONG_T;
#endif


/* ***********************************************************************/
/** \brief
 * Copies a series of character strings into another. This function is used
 * to implement character assignments and concatenations. It pads with
 * blanks if the destination string is longer than the sum of the lengths
 * of the source strings and truncates if the sum of the lengths of the
 * source strings is longer than the destination string
 *
 * \param n number of source strings
 * \param to pointer to destination string
 * \param to_len length of destination string
 * \param WCHAR *from pointer to kth source string
 * \param from_len length of kth source string
 */
/* ***********************************************************************/
void
Ftn_nstr_copy_klen(int n, WCHAR *to, _LONGLONG_T to_len, ...)
{
  va_list ap;
  WCHAR *from;
  int i;
  _LONGLONG_T from_len, j, k;

  va_start(ap, to_len);
  j = 0;
  for (i = 0; i < n; i++) {
    from = va_arg(ap, WCHAR *);
    from_len = va_arg(ap, _LONGLONG_T);
    for (k = 0; k < from_len; k++) {
      if (j == to_len)
        goto exit_return;
      to[j++] = from[k];
    }
  }
exit_return:
  va_end(ap);

  while (j < to_len)
    /* blank fill to right */
    to[j++] = BLANK;
}

/* ***********************************************************************/
/** \brief
 * Implements the INDEX intrinsic; is an integer function which returns the
 * value according to the INDEX intrinsic.
 */
/* **********************************************************************/
_LONGLONG_T
Ftn_nstr_index_klen(WCHAR *a1, /* pointer to string being searched */
                    WCHAR *a2, /* pointer to string being searched for */
                    _LONGLONG_T a1_len, /* length of a1 */
                    _LONGLONG_T a2_len) /* length of a2 */
{
  _LONGLONG_T i, j;
  int match;

  for (i = 0; i < a1_len; i++) {
    if (a2_len > (a1_len - i))
      return 0;
    match = TRUE;
    for (j = 0; j < a2_len; j++) {
      if (a1[i + j] != a2[j]) {
        match = FALSE;
        break;
      }
    }
    if (match)
      return (i + 1);
  }
  return 0;
}

/* ***********************************************************************/
/** \brief
 * Implements relational operators with string operands and the lexical
 * intrinsics. Returns integer value:
 * -  0 => strings are the same
 * - -1 => a1 lexically less than a2
 * -  1 => a1 lexically greater than a2
 * If the strings are of unequal lengths, treats shorter string as if it were
 * padded with blanks.
 */
/* ***********************************************************************/
int Ftn_nstrcmp_klen(WCHAR *a1,          /* first string to be compared */
                     WCHAR *a2,          /* second string to be compared */
                     _LONGLONG_T a1_len, /* length of a1 */
                     _LONGLONG_T a2_len) /* length of a2 */
{
  _LONGLONG_T i, k;

  /*  set k to minimum length of a1 and a2: */
  (a1_len > a2_len ? (k = a2_len) : (k = a1_len));

  /*  check first k characters: */
  for (i = 0; i < k; i++)
    if (a1[i] != a2[i]) {
      if (a1[i] < a2[i])
        return -1;
      return 1;
    }

  if (a1_len == a2_len)
    return 0; /* strings identical */

  /*  implicitly pad a2 with blanks if shorter than a1: */

  while (a1_len > a2_len)
    if (a1[a2_len++] != BLANK) {
      if (a1[a2_len - 1] < BLANK)
        return -1;
      return 1;
    }

  /*  implicitly pad a1 with blanks if shorter than a2: */

  while (a2_len > a1_len)
    if (a2[a1_len++] != BLANK) {
      if (a2[a1_len - 1] < BLANK)
        return 1;
      return -1;
    }

  return 0;
}
