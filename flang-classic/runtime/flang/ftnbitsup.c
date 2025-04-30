/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 *  \brief fortran bit manipulation support routines
 */

#include "enames.h"
#include "dattype.h"

/* ***********************************************************************/
/* function: Ftn_ishftc
 *
 *  performs circular bit shift.
 *  sc > 0 => circular left shift.
 *  sc < 0 => circular right shift.
 */
/* ***********************************************************************/

int
Ftn_ishftc(int val, /* value containing field to be shifted */
           int sc,  /* shift count and direction */
           int rc)  /* # of rightmost val bits to be shifted */
{
  unsigned int mask, field, tmp1, tmp2;
  int norm;

/* define a remainder operation that doesn't use %; is this worth it? */
#define REMLOOP(a, b, c) for (a = b; a >= c; a -= c)

  if (rc > 32 || rc <= 1)
    return val;
  if (sc == 0)
    return val;

  mask = ((unsigned)0xffffffff) >> (32 - rc); /* mask to extract field */
  field = val & mask;

  if (sc >= 0) {/**  CIRCULAR LEFT SHIFT  **/
                /*
                 * normalize the shift with respect to the field size.
                 */
    REMLOOP(norm, sc, rc);
    if (norm == 0)
      return val;
    /*
     * perform left shift discarding the bits which are shifted out of
     * the field.
     */
    tmp1 = (field << norm) & mask;
    /*
     * for those bits shifted out, justify them by right shifting
     */
    tmp2 = field >> (rc - norm);
  } else /* sc < 0 */ {/**  CIRCULAR RIGHT SHIFT  **/
    sc = -sc;
    /*
     * normalize the shift with respect to the field size.
     */
    REMLOOP(norm, sc, rc);
    if (norm == 0)
      return val;
    /*
     * perform right shift discarding the bits which are shifted out of
     * the field.
     */
    tmp1 = field >> norm;
    /*
     * for those bits shifted out, justify them by left shifting
     */
    tmp2 = (field << (rc - norm)) & mask;
  }
  /*
   * tmp1 | tmp2 represents the field after it has been shifted.
   * this value replaces the old field value.
   */

  return ((val ^ field) | tmp1 | tmp2);
}

/* ***********************************************************************/
/* function: Ftn_i_iishftc, 16-bit integer
 *
 *  performs circular bit shift.
 *  sc > 0 => circular left shift.
 *  sc < 0 => circular right shift.
 */
/* ***********************************************************************/

int
Ftn_i_iishftc(int val, /* value containing field to be shifted */
              int sc,  /* shift count and direction */
              int rc)  /* # of rightmost val bits to be shifted */
{
  unsigned int mask, field, tmp1, tmp2;
  int norm, res;

  if (rc > 32 || rc <= 1)
    return val;
  if (sc == 0)
    return val;

  mask = ((unsigned)0xffffffff) >> (32 - rc); /* mask to extract field */
  field = val & mask;

  if (sc >= 0) {/*   CIRCULAR LEFT SHIFT  **/
                /*
                 * normalize the shift with respect to the field size.
                 */
    REMLOOP(norm, sc, rc);
    if (norm == 0)
      return val;
    /*
     * perform left shift discarding the bits which are shifted out of
     * the field.
     */
    tmp1 = (field << norm) & mask;
    /*
     * for those bits shifted out, justify them by right shifting
     */
    tmp2 = field >> (rc - norm);
  } else /* sc < 0 */ {/*   CIRCULAR RIGHT SHIFT  **/
    sc = -sc;
    /*
     * normalize the shift with respect to the field size.
     */
    REMLOOP(norm, sc, rc);
    if (norm == 0)
      return val;
    /*
     * perform right shift discarding the bits which are shifted out of
     * the field.
     */
    tmp1 = field >> norm;
    /*
     * for those bits shifted out, justify them by left shifting
     */
    tmp2 = (field << (rc - norm)) & mask;
  }
  /*
   * tmp1 | tmp2 represents the field after it has been shifted.
   * this value replaces the old field value.
   */

  res = ((val ^ field) | tmp1 | tmp2);
  return (res << 16) >> 16;
}

/* ***********************************************************************/
/* function: Ftn_i_i1shftc, 8-bit integer
 *
 *  performs circular bit shift.
 *  sc > 0 => circular left shift.
 *  sc < 0 => circular right shift.
 */
/* ***********************************************************************/

int
Ftn_i_i1shftc(int val, /* value containing field to be shifted */
              int sc,  /* shift count and direction */
              int rc)  /* # of rightmost val bits to be shifted */
{
  unsigned int mask, field, tmp1, tmp2;
  int norm, res;

  if (rc > 32 || rc <= 1)
    return val;
  if (sc == 0)
    return val;

  mask = ((unsigned)0xffffffff) >> (32 - rc); /* mask to extract field */
  field = val & mask;

  if (sc >= 0) {/*   CIRCULAR LEFT SHIFT  **/
                /*
                 * normalize the shift with respect to the field size.
                 */
    REMLOOP(norm, sc, rc);
    if (norm == 0)
      return val;
    /*
     * perform left shift discarding the bits which are shifted out of
     * the field.
     */
    tmp1 = (field << norm) & mask;
    /*
     * for those bits shifted out, justify them by right shifting
     */
    tmp2 = field >> (rc - norm);
  } else /* sc < 0 */ {/*   CIRCULAR RIGHT SHIFT  **/
    sc = -sc;
    /*
     * normalize the shift with respect to the field size.
     */
    REMLOOP(norm, sc, rc);
    if (norm == 0)
      return val;
    /*
     * perform right shift discarding the bits which are shifted out of
     * the field.
     */
    tmp1 = field >> norm;
    /*
     * for those bits shifted out, justify them by left shifting
     */
    tmp2 = (field << (rc - norm)) & mask;
  }
  /*
   * tmp1 | tmp2 represents the field after it has been shifted.
   * this value replaces the old field value.
   */

  res = ((val ^ field) | tmp1 | tmp2);
  return (res << 24) >> 24;
}

/* ***********************************************************************/
/* function: Ftn_jmvbits
 *
 * moves len bits from pos in src to posd in dest
 */
/* ***********************************************************************/
void
Ftn_jmvbits(int src,   /* source field */
            int pos,   /* start position in source field */
            int len,   /* number of bits to move */
            int *dest, /* destination field */
            int posd)  /* start position in dest field */
{
  int mask;
  int tmp;

  /* procedure */

  if (pos < 0 || posd < 0 || len <= 0)
    return;
  if ((pos + len) > 32)
    return; /* ERROR MSG ??? */
  if ((posd + len) > 32)
    return; /* ERROR MSG ??? */
  if (len == 32) {
    *dest = src;
    return;
  }

  /*  create mask of len bits in proper position for dest */

  mask = (((unsigned)0xffffffff) >> (32 - len)) << posd;

  /*  extract field from src, position it for dest, and mask it  */

  tmp = (((src >> pos)) << posd) & mask;

  /*  mask out field in dest and or in value */

  *dest = (*dest & (~mask)) | tmp;

  return;
}

/* ***********************************************************************/
/* function: Ftn_imvbits
 *
 * moves len bits from pos in src to posd in dest-- dest is 16-bit integer
 */
/* ***********************************************************************/
void
Ftn_imvbits(int src,         /* source field */
            int pos,         /* start position in source field */
            int len,         /* number of bits to move */
            short int *dest, /* destination field */
            int posd)        /* start position in dest field */
{
  int mask;
  int tmp;

  /* procedure */

  if (pos < 0 || posd < 0 || len <= 0)
    return;
  if ((pos + len) > 32)
    return; /* ERROR MSG ??? */
  if ((posd + len) > 16)
    return; /* ERROR MSG ??? */
  if (len == 16 && pos == 0) {
    *dest = src;
    return;
  }

  /*  create mask of len bits in proper position for dest */

  mask = (((unsigned)0xffffffff) >> (32 - len)) << posd;

  /*  extract field from src, position it for dest, and mask it  */

  tmp = (((src >> pos)) << posd) & mask;

  /*  mask out field in dest and or in value */

  *dest = (*dest & (~mask)) | tmp;

  return;
}

/* ***********************************************************************/
/* function: Ftn_jzext
 *
 * zero extends value to 32 bits
 */
/* ***********************************************************************/
int
Ftn_jzext(int val, /* value to be zero extended */
          int dt)  /* data type of value */
{
  int result;
  result = 0;
  switch (dt) {

  case TY_BINT:
  case TY_BLOG:
    result = val & 0x000000ff;
    break;

  case TY_SINT:
  case TY_SLOG:
    result = val & 0x0000ffff;
    break;

  case TY_INT:
  case TY_LOG:
    result = val;
    break;
  }
  return (result);
}

/* ***********************************************************************/
/* function: Ftn_izext
 *
 *zero extends value to 16 bits
 */
/* ***********************************************************************/
int
Ftn_izext(int val, /* value to be zero extended */
          int dt)  /* data type of value */
{
  int result;
  result = 0;
  switch (dt) {

  case TY_BINT:
  case TY_BLOG:
    result = val & 0x000000ff;
    break;

  case TY_SINT:
  case TY_SLOG:
    result = val & 0x0000ffff;
    break;

  case TY_INT:
  case TY_LOG:
    result = val;
    break;
  }
  return (result);
}
