/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* fortran miscelleneous character support routines */

#include <string.h>
#include "stdarg.h"
#include "enames.h"
#include "ftni64.h"
#define TRUE 1
#define FALSE 0

#ifndef NULL
#define NULL (void *)0
#endif

/*************************************************************************/
/* function: Ftn_str_kindex:
 *
 * Implements the INDEX intrinsic; is an integer*8 function which returns the
 * value according to the INDEX intrinsic.
 */
/*************************************************************************/
_LONGLONG_T
ftn_str_kindex(char *a1,   /* pointer to string being searched */
               char *a2,   /* pointer to string being searched for */
               int a1_len, /* length of a1 */
               int a2_len) /* length of a2 */
{
  int idx1, idx2, match;
  for (idx1 = 0; idx1 < a1_len; idx1++) {
    if (a2_len > (a1_len - idx1))
      return 0;
    match = TRUE;
    for (idx2 = 0; idx2 < a2_len; idx2++) {
      if (a1[idx1 + idx2] != a2[idx2]) {
        match = FALSE;
        break;
      }
    }
    if (match) {
      return (idx1 + 1);
    }
  }
  return 0;
}

/*************************************************************************/
/* function: Ftn_strcmp
 *
 * Implements realational operators with string operands and the lexical
 * intrinsics. Returns integer value:
 *    0 => strings are the same
 *   -1 => a1 lexically less than a2
 *    1 => a1 lexically greater than a2
 * If the strings are of unequal lengths, treats shorter string as if it were
 * padded with blanks.
 */
/*************************************************************************/
int Ftn_kstrcmp(char *a1,   /* first string to be compared */
                char *a2,   /* second string to be compared */
                int a1_len, /* length of a1 */
                int a2_len) /* length of a2 */
{
  int ret_val, idx1;
  if (a1_len == a2_len) {
    ret_val = (memcmp(a1, a2, a1_len));
    if (ret_val == 0)
      return (0);
    if (ret_val < 0)
      return (-1);
    if (ret_val > 0)
      return (1);
  }
  if (a1_len > a2_len) {
    /* first compare the first a2_len characters of the strings */
    ret_val = memcmp(a1, a2, a2_len);
    if (ret_val != 0) {
      if (ret_val < 0)
        return (-1);
      if (ret_val > 0)
        return (1);
    }
    /*
     * if the last (a1_len - a2_len) characters of a1 are blank, then the
     * strings are equal; otherwise, compare the first non-blank char. to
     * blank
     */

    for (idx1 = 0; idx1 < (a1_len - a2_len); idx1++) {
      if (a1[a2_len + idx1] != ' ') {
        if (a1[a2_len + idx1] > ' ')
          return (1);
        return (-1);
      }
    }
    return (0);
  } else {
    /* a2_len > a1_len */
    /* first compare the first a1_len characters of the strings */
    ret_val = memcmp(a1, a2, a1_len);
    if (ret_val != 0) {
      if (ret_val < 0)
        return (-1);
      if (ret_val > 0)
        return (1);
    }
    /*
     * if the last (a2_len - a1_len) characters of a2 are blank, then the
     * strings are equal; otherwise, compare the first non-blank char. to
     * blank
     */

    for (idx1 = 0; idx1 < (a2_len - a1_len); idx1++) {
      if (a2[a1_len + idx1] != ' ') {
        if (a2[a1_len + idx1] > ' ')
          return (-1);
        return (1);
      }
    }
    return (0);
  }
}

/*************************************************************************/
/* function: Ftn_str_kindex_klen:
 *
 * Implements the INDEX intrinsic; is an integer*8 function which returns the
 * value according to the INDEX intrinsic.
 */
/*************************************************************************/
_LONGLONG_T
ftn_str_kindex_klen(char *a1, /* pointer to string being searched */
                    char *a2, /* pointer to string being searched for */
                    _LONGLONG_T a1_len, /* length of a1 */
                    _LONGLONG_T a2_len) /* length of a2 */
{
  _LONGLONG_T idx1, idx2;
  int match;
  for (idx1 = 0; idx1 < a1_len; idx1++) {
    if (a2_len > (a1_len - idx1))
      return 0;
    match = TRUE;
    for (idx2 = 0; idx2 < a2_len; idx2++) {
      if (a1[idx1 + idx2] != a2[idx2]) {
        match = FALSE;
        break;
      }
    }
    if (match) {
      return (idx1 + 1);
    }
  }
  return 0;
}

/*************************************************************************/
/* function: Ftn_strcmp_klen
 *
 * Implements realational operators with string operands and the lexical
 * intrinsics. Returns integer value:
 *    0 => strings are the same
 *   -1 => a1 lexically less than a2
 *    1 => a1 lexically greater than a2
 * If the strings are of unequal lengths, treats shorter string as if it were
 * padded with blanks.
 */
/*************************************************************************/
int Ftn_kstrcmp_klen(char *a1,           /* first string to be compared */
                     char *a2,           /* second string to be compared */
                     _LONGLONG_T a1_len, /* length of a1 */
                     _LONGLONG_T a2_len) /* length of a2 */
{
  _LONGLONG_T idx1;
  int ret_val;
  if (a1_len == a2_len) {
    ret_val = (memcmp(a1, a2, (size_t)a1_len));
    if (ret_val == 0)
      return (0);
    if (ret_val < 0)
      return (-1);
    if (ret_val > 0)
      return (1);
  }
  if (a1_len > a2_len) {
    /* first compare the first a2_len characters of the strings */
    ret_val = memcmp(a1, a2, (size_t)a2_len);
    if (ret_val != 0) {
      if (ret_val < 0)
        return (-1);
      if (ret_val > 0)
        return (1);
    }
    /*
     * if the last (a1_len - a2_len) characters of a1 are blank, then the
     * strings are equal; otherwise, compare the first non-blank char. to
     * blank
     */

    for (idx1 = 0; idx1 < (a1_len - a2_len); idx1++) {
      if (a1[a2_len + idx1] != ' ') {
        if (a1[a2_len + idx1] > ' ')
          return (1);
        return (-1);
      }
    }
    return (0);
  } else {
    /* a2_len > a1_len */
    /* first compare the first a1_len characters of the strings */
    ret_val = memcmp(a1, a2, (size_t)a1_len);
    if (ret_val != 0) {
      if (ret_val < 0)
        return (-1);
      if (ret_val > 0)
        return (1);
    }
    /*
     * if the last (a2_len - a1_len) characters of a2 are blank, then the
     * strings are equal; otherwise, compare the first non-blank char. to
     * blank
     */

    for (idx1 = 0; idx1 < (a2_len - a1_len); idx1++) {
      if (a2[a1_len + idx1] != ' ') {
        if (a2[a1_len + idx1] > ' ')
          return (-1);
        return (1);
      }
    }
    return (0);
  }
}
