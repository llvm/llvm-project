/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* fortran bit manipulation support routines */

#include "enames.h"
#include "dattype.h"
#include "ftni64.h"

/* forward declarations */
void shf64(DBLINT64, int, DBLINT64);
void ushf64(DBLUINT64, int, DBLUINT64);

/*************************************************************************/
/* function: ftn_i_kishftc
 *
 *  performs circular bit shift.
 *  sc > 0 => circular left shift.
 *  sc < 0 => circular right shift.
 */
/*************************************************************************/

__I8RET_T
ftn_i_kishftc(_LONGLONG_T op, /* value containing field to be shifted */
              int sc,         /* shift count and direction */
              int rc)         /* # of rightmost val bits to be shifted */
{
  DBLUINT64 i8neg1, mask, field, tmp1, tmp2, val;
  int norm;

/* define a remainder operation that doesn't use %; is this worth it? */
#define REMLOOP(a, b, c) for (a = b; a >= c; a -= c)

  INT64D u;

  u.lv = op;
  val[0] = I64_MSH(u.i);
  val[1] = I64_LSH(u.i);

  tmp1[0] = tmp1[1] = 0;
  tmp2[0] = tmp2[1] = 0;

  if (rc > 64 || rc <= 1)
    UTL_I_I64RET(val[0], val[1]);
  if (sc == 0)
    UTL_I_I64RET(val[0], val[1]);

  /* create mask to extract field */
  if (__ftn_32in64_) {
    mask[1] = ((unsigned)0xffffffff) >> (32 - rc);
    field[0] = 0;
    field[1] = val[1] & mask[1];
  } else {
    i8neg1[0] = -1;
    i8neg1[1] = -1;
    ushf64(i8neg1, rc - 64, mask);
    field[0] = val[0] & mask[0];
    field[1] = val[1] & mask[1];
  }

  if (sc >= 0) {/**  CIRCULAR LEFT SHIFT  **/
                /*
                 * normalize the shift with respect to the field size.
                 */
    REMLOOP(norm, sc, rc);
    if (norm == 0)
      UTL_I_I64RET(val[0], val[1]);
    /*
     * perform left shift discarding the bits which are shifted out of
     * the field.  then, for those bits shifted out, justify them by
     * right shifting
     */
    if (__ftn_32in64_) {
      tmp1[1] = (field[1] << norm) & mask[1];
      tmp2[1] = field[1] >> (rc - norm);
    } else {
      ushf64(field, norm, tmp1);
      tmp1[0] &= mask[0];
      tmp1[1] &= mask[1];
      ushf64(field, norm - rc, tmp2);
    }
  } else /* sc < 0 */ {/**  CIRCULAR RIGHT SHIFT  **/
    sc = -sc;
    /*
     * normalize the shift with respect to the field size.
     */
    REMLOOP(norm, sc, rc);
    if (norm == 0)
      UTL_I_I64RET(val[0], val[1]);
    /*
     * perform right shift discarding the bits which are shifted out of
     * the field.  then, for those bits shifted out, justify them by
     * left shifting
     */
    if (__ftn_32in64_) {
      tmp1[1] = field[1] >> norm;
      tmp2[1] = (field[1] << (rc - norm)) & mask[1];
    } else {
      ushf64(field, -norm, tmp1);
      ushf64(field, rc - norm, tmp2);
      tmp2[0] &= mask[0];
      tmp2[1] &= mask[1];
    }
  }
  /*
   * tmp1 | tmp2 represents the field after it has been shifted.
   * this value replaces the old field value.
   */
  val[0] = (val[0] ^ field[0]) | tmp1[0] | tmp2[0];
  val[1] = (val[1] ^ field[1]) | tmp1[1] | tmp2[1];
  UTL_I_I64RET(val[0], val[1]);
}

/*************************************************************************/
/* function: Ftn_kmvbits
 *
 * moves len bits from pos in src to posd in dest
 */
/*************************************************************************/
void Ftn_kmvbits(int *src,  /* source field */
                 int pos,   /* start position in source field */
                 int len,   /* number of bits to move */
                 int *dest, /* destination field */
                 int posd)  /* start position in dest field */
{
  int mask;
  int tmp;
  int maxpos;
  int maxlen;
  DBLUINT64 maski8;
  DBLUINT64 i8neg1, tmpi8, u_arg;

  /* procedure */

  if (pos < 0 || posd < 0 || len <= 0)
    return;
  if ((pos + len) > 64)
    return; /* ERROR MSG ??? */
  if ((posd + len) > 64)
    return; /* ERROR MSG ??? */

  /* THIS ASSUMES PHASE 1 -- 32 bits in 64bits (affect least
   * significant portion only) -- needs to be fixed for phase 2
   * (also, argument 'src' needs to be declared to be an int *)
   */
  if (__ftn_32in64_) {
    maxpos = 31;
    maxlen = 32;
  } else {
    maxpos = 63;
    maxlen = 64;
  }
  if (posd > maxpos || pos > maxpos)
    return;

  if (pos + len > maxlen) {
    tmp = (pos + len) - maxlen;
    len -= tmp;
  }
  if (posd + len > maxlen) {
    tmp = (posd + len) - maxlen;
    len -= tmp;
  }

  if (len <= 0)
    return;

  if (len == maxlen) {
    *dest = *src;
    return;
  }

  /*  create mask of len bits in proper position for dest */

  if (__ftn_32in64_) {
    mask = (((unsigned)0xffffffff) >> (maxlen - len)) << posd;

    /*  extract field from src, position it for dest, and mask it  */

    tmp = ((*src >> pos) << posd) & mask;

    /*  mask out field in dest and or in value */

    *dest = (*dest & (~mask)) | tmp;
  } else {
    u_arg[0] = I64_MSH(src);
    u_arg[1] = I64_LSH(src);

    i8neg1[0] = -1;
    i8neg1[1] = -1;
    ushf64(i8neg1, -(maxlen - len), maski8);
    ushf64(maski8, posd, maski8);
    ushf64(u_arg, -pos, tmpi8);
    ushf64(tmpi8, posd, tmpi8);
    tmpi8[0] &= maski8[0];
    tmpi8[1] &= maski8[1];
    I64_MSH(dest) = (I64_MSH(dest) & (~maski8[0])) | tmpi8[0];
    I64_LSH(dest) = (I64_LSH(dest) & (~maski8[1])) | tmpi8[1];
  }

  return;
}

/*************************************************************************/
/* function: ftn_i_kibclr
 *
 * clear a bit in a 64 bit value:        r = arg & ~(1 << bit)
 */
/*************************************************************************/
__I8RET_T
ftn_i_kibclr(int arg1, int arg2, /* value to be cleared */
             int bit)            /* bit to clear        */
{
  DBLINT64 result;
  DBLUINT64 i81, tmp;
  result[0] = result[1] = 0;
  i81[0] = 0;
  i81[1] = 1;
  ushf64(i81, bit, tmp);
  result[0] = arg2 & ~tmp[0];
  result[1] = arg1 & ~tmp[1];

  UTL_I_I64RET(result[0], result[1]);
}

/*************************************************************************/
/* function: ftn_i_kibits
 *
 * extract bits from a 64 bit value:
 *    r = (arg1,arg2 >> bitpos) & (-1 >> (64 - numbits))
 */
/*************************************************************************/
__I8RET_T
ftn_i_kibits(int arg1, int arg2, /* value to be extracted from    */
             int bitpos,         /* position of bit to start from */
             int numbits)        /* number of bits to extract     */
{
  DBLINT64 result;
  DBLUINT64 i8neg1, tmp, maski8, u_arg;
  u_arg[0] = arg2;
  u_arg[1] = arg1;

  result[0] = result[1] = 0;
  ushf64(u_arg, -bitpos, tmp);

  i8neg1[0] = -1;
  i8neg1[1] = -1;
  ushf64(i8neg1, -(64 - numbits), maski8);

  tmp[0] &= maski8[0];
  tmp[1] &= maski8[1];
  result[0] = tmp[0];
  result[1] = tmp[1];

  UTL_I_I64RET(result[0], result[1]);
}

/*************************************************************************/
/* function: ftn_i_kibset
 *
 * set bit in a 64 bit value:        r = arg1,arg2 | (1 << bit)
 */
/*************************************************************************/
__I8RET_T
ftn_i_kibset(int arg1, int arg2, /* value to be set   */
             int bit)            /* bit to set        */
{
  DBLINT64 i8one, result;
  DBLINT64 tmp;
  result[0] = result[1] = 0;
  i8one[0] = 0;
  i8one[1] = 1;
  shf64(i8one, bit, tmp);
  tmp[0] |= arg2;
  tmp[1] |= arg1;
  result[0] = tmp[0];
  result[1] = tmp[1];

  UTL_I_I64RET(result[0], result[1]);
}

/*************************************************************************/
/* function: ftn_i_bktest
 *
 * test bit in a 64 bit value:        r = (arg1,arg2 & (1 << bit)) != 0
 */
/*************************************************************************/
__I8RET_T
ftn_i_bktest(int arg1, int arg2, /* value to be tested  */
             int bit)            /* bit to test         */
{
  DBLINT64 i8one, result;
  DBLINT64 tmp;
  result[0] = result[1] = 0;
  i8one[0] = 0;
  i8one[1] = 1;
  shf64(i8one, bit, tmp);
  tmp[0] = arg2 & tmp[0];
  tmp[1] = arg1 & tmp[1];
  if ((tmp[0] == 0) && (tmp[1] == 0)) {
    result[0] = 0;
    result[1] = 0;
  } else {
    result[0] = -1;
    result[1] = -1;
  }

  UTL_I_I64RET(result[0], result[1]);
}

/*************************************************************************/
/*
 *
 *  shift a 64-bit signed integer
 *
 *	Arguments:
 *	    arg     INT [2] operand to be shifted
 *	    count   int shift count  (- => right shift; o.w., left shift).
 *	            A count outside of the range -63 to 64 results in a result
 *                  of zero; the caller takes into consideration machine
 *		    dependicies (such as the shift count is modulo 64).
 *	    result
 *
 *	Return value:
 *	    none.
 */
void shf64(DBLINT64 arg, int count, DBLINT64 result)
{
  DBLUINT64 u_arg; /* 'copy-in' unsigned value of arg */

  if (count >= 64 || count <= -64) {
    result[0] = 0;
    result[1] = 0;
    return;
  }
  if (count == 0) {
    result[0] = arg[0];
    result[1] = arg[1];
    return;
  }
  u_arg[0] = arg[0];
  u_arg[1] = arg[1];
  if (count > 0) {
    if (count < 32) {
      result[0] = (u_arg[0] << count) | (u_arg[1] >> (32 - count));
      result[1] = u_arg[1] << count;
    } else {
      result[0] = u_arg[1] << (count - 32);
      result[1] = 0;
    }
  } else if (count > -32) {
    result[0] = arg[0] >> -count; /* sign extend */
    result[1] = (u_arg[1] >> -count) | (u_arg[0] << (count + 32));
  } else {
    result[0] = arg[0] >> 31; /* sign extend */
    result[1] = arg[0] >> (-count - 32);
  }
}

/*
 *  shift an 64-bit unsigned integer
 *
 *	Arguments:
 *	    arg     UINT[2] operand to be shifted
 *	    count   int shift count  (- => left shift; o.w., right shift).
 *	            A count outside of the range -63 to 64 results in a result
 *                  of zero; the caller takes into consideration machine
 *		    dependicies (such as the shift count is modulo 64).
 *	    result
 *
 *	Return value:
 *	    none.
 */
void ushf64(DBLUINT64 arg, int count, DBLUINT64 result)
{
  DBLUINT64 u_arg; /* 'copy-in' value of arg */

  if (count >= 64 || count <= -64) {
    result[0] = 0;
    result[1] = 0;
    return;
  }
  if (count == 0) {
    result[0] = arg[0];
    result[1] = arg[1];
    return;
  }
  u_arg[0] = arg[0];
  u_arg[1] = arg[1];
  if (count >= 0) {
    if (count < 32) {
      result[0] = (u_arg[0] << count) | (u_arg[1] >> (32 - count));
      result[1] = (u_arg[1] << count);
    } else {
      result[0] = u_arg[1] << (count - 32);
      result[1] = 0;
    }
  } else if (count > -32) {
    result[0] = u_arg[0] >> -count;
    result[1] = (u_arg[1] >> -count) | (u_arg[0] << (count + 32));
  } else {
    result[0] = 0;
    result[1] = u_arg[0] >> (-count - 32);
  }
}
