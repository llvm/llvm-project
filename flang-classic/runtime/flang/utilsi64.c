/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "global.h"

/* This file provides the necessary routines for 64-bit integer formatted
 * I/O support.  This file is essentially a copy of the utils.c with
 * the exception that TM_I8 => integer*4 is the natural integer and
 * integer*8 is an extension.  All of these support routines could be
 * rewritten to use the appropriate C type which represents a 64-bit
 * integer rather than DBLINT64/DBLUINT64.
 */
int __ftn_32in64_;

static void neg64(DBLINT64, DBLINT64);
static void uneg64(DBLUINT64, DBLUINT64);
static int toi64(char *, DBLINT64, char *, int);
static int ucmp64(DBLUINT64, DBLUINT64);

/* defined in ftni64bitsup.c */
void shf64(DBLINT64, int, DBLINT64);
void ushf64(DBLUINT64, int, DBLUINT64);

/* has native support for 8-byte integers*/
#if !defined(_WIN64)
typedef long I8_T;
typedef unsigned long UI8_T;
#else
typedef __int64 I8_T;
typedef unsigned __int64 UI8_T;
#endif

typedef union {
  UI8_T u8;
  I8_T i8;
  struct {
    int lsw;
    int msw;
  } ovl8;
} OVL8;
#define NATIVEI8

/* ***************************************************************
 *
 * char string to 32-bit integer.
 *
 *      Arguments:
 *
 *          s       Input string containing number to be converted
 *			(string is NOT null terminated.)
 *          i       Pointer to returned integer value
 *          n       Number of chars from str to convert
 *          base    Radix of conversion -- 2, 8, 10, 16.  If
 *                  base is 16, then the digits a-f or A-F are
 *                  allowed.
 *
 *      Return Value:
 *
 *          -1      str does not represent a valid number in radix base
 *          -2      overflow occurred on conversion
 *           0      No error -- *ival contains the converted number.
 *
 *      Description:
 *
 *          This routine accepts char strings representing integers
 *          in any base of 16 or less.  The numbers must fit into
 *	    a 32-bit signed or unsigned integer. Only one preceding
 *	    sign char is allowed for all bases of numbers.
 *
 *      NOTE:
 *          This routine only works on 2's complement machines.
 *
 ****************************************************************/

int
__fort_atoxi32(char *s, INT *i, int n, int base)
{
  register char *end;
  register INT value;
  int sign;
  UINT xval, yval;

  /* Skip any leading blanks. */
  end = s + n;
  *i = 0;
  for (; s < end && *s == ' '; s++)
    ;

  /* Are there any chars left? */
  if (s >= end)
    return (-1);

  /* Look for a sign char. */
  sign = 1;
  if (*s == '-') {
    sign = -1;
    s++;
  } else if (*s == '+')
    s++;

  /* Are there any chars left? */
  if (s >= end)
    return (-1);

  switch (base) {
  case 2:
    for (value = 0; s < end; s++) {
      if ((value & 0x80000000L) != 0)
        goto ovflo;

      value <<= 1;

      if (*s < '0' || *s > '1')
        return (-1);

      if (*s == '1')
        value |= 1L;
    }
    break;
  case 8:
    for (value = 0; s < end; s++) {
      if ((value & 0xE0000000L) != 0)
        goto ovflo;

      value <<= 3;

      if (*s < '0' || *s > '7')
        return (-1);

      value |= (*s - '0');
    }
    break;
  case 16:
    for (value = 0; s < end; s++) {
      if ((value & 0xF0000000L) != 0)
        goto ovflo;

      value <<= 4;

      if (*s < '0')
        return (-1);
      else if (*s <= '9')
        value |= (*s - '0');
      else if (*s < 'A')
        return (-1);
      else if (*s <= 'F')
        value |= (*s - 'A' + 10);
      else if (*s < 'a')
        return (-1);
      else if (*s <= 'f')
        value |= (*s - 'a' + 10);
      else
        return (-1);
    }
    break;
  case 10:
    /* use unsigned */
    xval = yval = 0;
    for (; s < end; s++) {
      if (*s < '0' || *s > '9')
        return -1;
      xval *= 10;
      xval += (*s - '0');
      if ((xval & 0x80000000u) || (xval < yval)) {
        value = 0xffffffffu; /* 4294967295u */
        goto ovflo;
      }
      if (yval >= 0x0ccccccc && (xval - (*s - '0')) / 10 != yval) {
        /* Limit this check to when yval >= max_int/10 */
        value = 0xffffffffu; /* 4294967295u */
        goto ovflo;
      }
      yval = xval;
    }
    value = xval;
    break;
  default:
    return (-1);
  }

  if (sign == -1) {
    if (((UINT)value & 0x80000000u) != 0 && (UINT)value != 0x80000000u)
      goto ovflo;
    *i = (~value) + 1;
  } else
    *i = value;
  return (0);
ovflo:
  *i = value;
  return -2;
}

/* ***************************************************************
 *
 * char string to 64-bit integer.
 *
 *      Arguments:
 *
 *          s       Input string containing number to be converted
 *			(string is NOT null terminated.)
 *          ir      DBLUINT64 output value
 *          n       Number of chars from str to convert
 *          radix   Radix of conversion -- 2, 8, 10, 16.  If
 *                  base is 16, then the digits a-f or A-F are
 *                  allowed.
 *
 *      Return Value:
 *
 *          -1      str does not represent a valid number in radix base
 *          -2      overflow occurred on conversion
 *           0      No error -- *ir contains the converted number.
 *
 *      Description:
 *
 *          This routine accepts char strings representing integers
 *          in any base of 16 or less.  The numbers must fit into
 *	    a 64-bit signed. Only one preceding sign char is allowed
 *          for all bases of numbers.
 *
 *      NOTE:
 *          This routine only works on 2's complement machines.
 *
 ****************************************************************/

int
__fort_atoxi64(char *s, DBLINT64 ir, int n, int radix)
{
  int err;
  char *sp;
  char *end;

  /* eat whitespace */
  end = s + n;
  sp = s;
  for (; sp < end && *sp == ' '; sp++)
    n--;

  if (n <= 0)
    return (-1);

  err = toi64(sp, ir, end, radix);

  return (err);
}
/* **** __fort_i64toax
 *
 *   Converts [un]signed 64 integer into a char string of
 *   the selected radix.
 *
 */

#define ASCII_OFFSET 48
#define ASTERISK '*'
#define BLANK ' '
#define HEX_OFFSET 7
#define MINUS '-'
#define ZERO '0'

void
__fort_i64toax(DBLINT64 from, char *to, int count, int sign, int radix)  
{
  int bit_width;     /* width of the bit field for a particular
                      * radix */
  int bits_to_shift; /* number of bits to shift */
  int idx;           /* for loop control variable */
  INT mask;          /* mask for bit_width of a particular radix */
  int max_shift;     /* maximum number of bits from will will need
                      * to be shifted */
  int msd;           /* index of the most-signingicant digit in to */
  int num_bits;      /* number of bits to be shifted */
  DBLINT64 temp_from;   /* temp from (=(abs(from)) */
  DBLINT64 temp64;      /* temporary 64 bit integer */

  if ((from[0] == 0) && (from[1] == 0)) {
    msd = count - 1;
    to[msd] = ASCII_OFFSET;
  }
  else if (radix == 10) {
    OVL8 temp;
    I8_T rem, quot;
    if (sign == 0 && I64_MSH(from) == (INT)0x80000000 && I64_LSH(from) == 0) {
      if (count <= (int)strlen("-9223372036854775808")) {
        to[0] = ASTERISK;
        to[1] = '\0';
      } else
        strcpy(to, "-9223372036854775808");
      return;
    }
    if (sign == 1 && from[0] == -1 && from[1] == -1) {
      strcpy(to, "-1");
      return;
    }
    temp.ovl8.msw = I64_MSH(from);
    temp.ovl8.lsw = I64_LSH(from);
    if ((sign == 1) && (temp.ovl8.msw < 0)) {
      temp.ovl8.msw = temp.ovl8.msw & 0x7fffffff; /* negate sign bit */
      quot = temp.i8 / 10;
      rem = temp.i8 - quot * 10;
      rem = rem + 8;                         /* 8 = 2^63 % 10 */
      temp.i8 = quot + 922337203685477580LL; /* add msbdiv10 */

      if (rem >= 10) {
        rem = rem - 10;
        temp.i8 += 1;
      }
      msd = count - 2;
      to[msd + 1] = ASCII_OFFSET + rem;
    } else {
      temp.ovl8.msw = I64_MSH(from);
      temp.ovl8.lsw = I64_LSH(from);
      if ((sign == 0) && (temp.ovl8.msw < 0))
        temp.i8 = -temp.i8;
      msd = count - 1;
    }

    while ((msd >= 0) && temp.i8 != 0) {
      quot = temp.i8 / 10;
      rem = temp.i8 - quot * 10;
      to[msd] = ASCII_OFFSET + rem;
      temp.i8 = quot;
      msd = msd - 1;
    }

    if (msd == -1) {
      if (temp.i8 == 0)
        msd = 0;
    } else
      msd = msd + 1;
  }
  else {
    temp_from[0] = I64_MSH(from);
    temp_from[1] = I64_LSH(from);
    if ((sign == 0) && (I64_MSH(from) < 0))
      neg64(temp_from, temp_from);

    switch (radix) {
    case 2:
      max_shift = 63;
      bit_width = 1;
      mask = 1;
      break;
    case 8:
      max_shift = 63;
      bit_width = 3;
      mask = 7;
      break;
    case 16:
      max_shift = 60;
      bit_width = 4;
      mask = 15;
      break;
    }

    idx = count - 1;
    for (num_bits = 0; num_bits <= max_shift; num_bits = num_bits + bit_width) {
      if ((radix == 8) && (num_bits == 63))
        mask = 1;

      bits_to_shift = -num_bits;
      shf64(temp_from, bits_to_shift, temp64);

      to[idx] = ASCII_OFFSET + (temp64[1] & mask);

      if (to[idx] != ASCII_OFFSET)
        msd = idx;

      if (to[idx] > '9')
        to[idx] = to[idx] + HEX_OFFSET;

      if (idx == 0) {
        bits_to_shift = -(num_bits + bit_width);
        shf64(temp_from, bits_to_shift, temp64);
        if ((temp64[0] != 0) || (temp64[1] != 0))
          msd = -1;
        break; /* out of for loop */
      } else
        idx = idx - 1;
    }
  }

  if ((msd == -1) || ((sign == 0) && (msd == 0) && (I64_MSH(from) < 0)))
    to[0] = ASTERISK;
  else if (msd == 0) {
    to[0] = '0';
    to[1] = '\0';
  } else {
    if ((sign == 0) && (I64_MSH(from) < 0)) {
      msd = msd - 1;
      to[msd] = MINUS;
    }
    for (idx = msd; idx < count; ++idx)
      to[idx - msd] = to[idx];

    idx = count - msd;
    to[idx] = '\0';
  }
}

/*
 * error return value:
 * -1 = bad format
 * -2 = overflow / underflow
 *  0 = no error.
 */
static int toi64(char *s, DBLINT64 toi, char *end, int radix)
{
  DBLINT64 base; /* 64 bit integer equal to radix */
  DBLINT64 num;  /* numerical value of a particular digit */
  DBLUINT64 to;
  int negate;
  int ch;

  /* 64-bit integer with only its sign bit on */
  static DBLUINT64 sign_bit = {0x80000000, 0};

  /* maximum 64-bit signed integer */
  static DBLINT64 max_int = {0x7fffffff, 0xffffffff};

  OVL8 pto;

  negate = 0;
  if (*s == '+')
    s++;
  else if (*s == '-') {
    negate = 1;
    s++;
  }

  if (s >= end)
    return -1;

  to[0] = 0;
  to[1] = 0;
  base[0] = 0;
  base[1] = radix;
  num[0] = 0;
  toi[0] = 0;
  toi[1] = 0;

  switch (radix) {
  case 2:
    for (; s < end; s++) {
      if (to[0] & 0x80000000L)
        goto ovflo;

      ushf64(to, 1, to);

      if (*s < '0' || *s > '1')
        return -1;

      if (*s == '1')
        to[1] |= 1;
    }
    break;
  case 8:
    for (; s < end; s++) {
      if (to[0] & 0xE0000000L)
        goto ovflo;

      ushf64(to, 3, to);

      if (*s < '0' || *s > '7')
        return -1;

      to[1] |= (*s - '0');
    }
    break;
  case 16:
    for (; s < end; s++) {
      if (to[0] & 0xF0000000L)
        goto ovflo;
      ushf64(to, 4, to);
      ch = *s & 0xff;
      if (ch < '0')
        return (-1);
      else if (ch <= '9')
        to[1] |= (ch - '0');
      else if (ch < 'A')
        return (-1);
      else if (ch <= 'F')
        to[1] |= (ch - 'A' + 10);
      else if (ch < 'a')
        return (-1);
      else if (ch <= 'f')
        to[1] |= (ch - 'a' + 10);
      else
        return (-1);
    }
    break;
  case 10:
    pto.u8 = 0;
    for (; s < end; s++) {
      UI8_T prev;
      prev = pto.u8;
      ch = *s & 0xff;
      if (ch < '0' || ch > '9')
        return -1;
      pto.u8 *= 10;
      pto.u8 += ch - '0';
      if (pto.u8 < prev) {
        to[0] = max_int[0];
        to[1] = max_int[1];
        goto ovflo;
      }
    }
    to[0] = pto.ovl8.msw;
    to[1] = pto.ovl8.lsw;
    break;
  default:
    return -1;
  }

  if (negate) {
    if (ucmp64(to, sign_bit) == 1)
      return -2;
    uneg64(to, to);
  }

  I64_MSH(toi) = to[0];
  I64_LSH(toi) = to[1];
  return 0;
ovflo:
  return -2;
}

static void neg64(DBLINT64 arg, DBLINT64 result)
{
  int sign; /* sign of the low-order word of arg prior to
             * being complemented */
  sign = (unsigned)arg[1] >> 31;
  result[0] = ~arg[0];
  result[1] = (~arg[1]) + 1;
  if (sign == 0 && result[1] >= 0)
    result[0]++;
}

static void uneg64(DBLUINT64 arg, DBLUINT64 result)
{
  int sign; /* sign of the low-order word of arg prior to
             * being complemented */

  sign = (unsigned)arg[1] >> 31;
  result[0] = ~arg[0];
  result[1] = (~arg[1]) + 1;
  if (sign == 0 && (int)result[1] >= 0)
    result[0]++;
}


static int ucmp64(DBLUINT64 arg1, DBLUINT64 arg2)
{
  if (arg1[0] == arg2[0]) {
    if (arg1[1] == arg2[1])
      return 0;
    if (arg1[1] < arg2[1])
      return -1;
    return 1;
  }
  if (arg1[0] < arg2[0])
    return -1;
  return 1;
}

#ifdef MTHI64
static void __utl_i_mul128(DBLINT64, DBLINT64, INT[4]);
static void neg128(INT[4], INT[4]);
static void uneg64(DBLUINT64, DBLUINT64);
static void ushf64(DBLUINT64, int, DBLINT64);
static void shf128(INT[4], int, INT[4]);

/*
 *  Add 2 64-bit integers
 *
 *	Arguments:
 *	    arg1, arg2  64-bit addends
 *	    result      64-bit result of arg1 + arg2
 *
 *	Return value:
 *	    none
 */
void __utl_i_add64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  int carry;    /* value to be carried from adding the lower
                 * 32 bits */
  int sign_sum; /* sum of the sign bits of the low-order
                 * words of arg1 and arg2 */
  sign_sum = ((unsigned)arg1[1] >> 31) + ((unsigned)arg2[1] >> 31);
  result[1] = arg1[1] + arg2[1];
  /*
   * sign_sum = 2 -> carry guaranteed
   *          = 1 -> if sum sign bit off then carry
   *          = 0 -> carry not possible
   */
  carry = sign_sum > (int)((unsigned)result[1] >> 31);
  result[0] = arg1[0] + arg2[0] + carry;
}

/*
 *  Subtract two 64-bit integers
 *
 *	Arguments:
 *	    arg1    minuend
 *	    arg2    subtrahend
 *	    result  arg1 - arg2
 *
 *	Return value:
 *	    none
 */
void __utl_i_sub64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result) 

{
  int borrow;    /* value to be borrowed from adding the lower
                  * 32 bits */
  int sign_diff; /* difference of the sign bits of the
                  * low-order words of arg1 and arg2 */

  sign_diff = ((unsigned)arg1[1] >> 31) - ((unsigned)arg2[1] >> 31);
  result[1] = arg1[1] - arg2[1];
  /*
   * sign_diff = -1 -> borrow guaranteed
   *           = 0  -> if diff sign bit on (arg2 > arg1) then borrow
   *             1  -> borrow not possible
   */
  borrow = sign_diff < (int)((unsigned)result[1] >> 31);
  result[0] = (arg1[0] - borrow) - arg2[0];
}

/*
 *
 *       Multiply two 64-bit integers to produce a 64-bit
 *       integer product.
 */

void __utl_i_mul64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  INT temp_result[4]; /* the product returned by MUL128 */

  __utl_i_mul128(arg1, arg2, temp_result);
  result[0] = temp_result[2];
  result[1] = temp_result[3];
}

static void __utl_i_mul128(DBLINT64 arg1, DBLINT64 arg2, INT result[4])
{
  int i;              /* for loop control variable */
  DBLINT64 temp_arg;     /* temporary argument used in calculating the
                          * product */
  INT temp_result[4]; /* temporary result */
  int negate;         /* flag which indicated the result needs to
                       * be negated */

  if ((arg1[0] == 0 && arg1[1] == 0) || (arg2[0] == 0 && arg2[1] == 0)) {
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
    return;
  }
  temp_result[0] = 0;
  temp_result[1] = 0;

  if (arg1[0] < 0) {
    neg64(arg1, &temp_result[2]);
    negate = 1;
  } else {
    temp_result[2] = arg1[0];
    temp_result[3] = arg1[1];
    negate = 0;
  }

  if (arg2[0] < 0) {
    neg64(arg2, temp_arg);
    negate = !negate;
  } else {
    temp_arg[0] = arg2[0];
    temp_arg[1] = arg2[1];
  }

  for (i = 0; i < 64; ++i) {
    if ((temp_result[3] & 1) == 1)
      __utl_i_add64(temp_result, temp_arg, temp_result);

    shf128(temp_result, -1, temp_result);
  }

  if (negate)
    neg128(temp_result, result);
  else
    for (i = 0; i < 4; ++i)
      result[i] = temp_result[i];
}

void __utl_i_div64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
{
  DBLINT64 den;          /* denominator used in calculating the
                       * quotient */
  int i;              /* for loop control variable */
  int temp_result[4]; /* temporary result used in
                       * calculating the quotient */
  int negate;         /* flag which indicates the result needs to
                       * be negated */
  int one;            /* one passed to shf128 */

  if ((arg1[0] == 0 && arg1[1] == 0) || (arg2[0] == 0 && arg2[1] == 0)) {
    result[0] = 0;
    result[1] = 0;
    return;
  }
  temp_result[0] = 0;
  temp_result[1] = 0;

  if (arg1[0] < 0) {
    neg64(arg1, &temp_result[2]);
    negate = 1;
  } else {
    temp_result[2] = arg1[0];
    temp_result[3] = arg1[1];
    negate = 0;
  }

  if (arg2[0] < 0) {
    neg64(arg2, den);
    negate = !negate;
  } else {
    den[0] = arg2[0];
    den[1] = arg2[1];
  }

  one = 1;

  for (i = 0; i < 64; ++i) {
    shf128(temp_result, one, temp_result);
    if (ucmp64(temp_result, den) >= 0) {
      __utl_i_sub64(temp_result, den, temp_result);
      temp_result[3] = temp_result[3] + 1;
    }
  }

  if (negate)
    neg64(&temp_result[2], result);
  else {
    result[0] = temp_result[2];
    result[1] = temp_result[3];
  }
}

static void neg128(INT arg[4], INT result[4]) 

{
  int i;    /* loop control variable */
  int sign; /* sign of the word of arg prior to being */

  for (i = 0; i < 4; ++i)
    result[i] = ~arg[i];

  i = 3;
  sign = (unsigned)result[i] >> 31;
  result[i] = result[i] + 1;
  while (i > 0 && sign == 1 && result[i] >= 0) {
    i = i - 1;
    sign = (unsigned)result[i] >> 31;
    result[i] = result[i] + 1;
  }
}

void __utl_i_udiv64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result)
{
  DBLUINT64 den;         /* denominator used in calculating the
                       * quotient */
  int i;              /* for loop control variable */
  int temp_result[4]; /* temporary result used in
                       * calculating the quotient */
  int negate;         /* flag which indicates the result needs to
                       * be negated */
  int one;            /* one passed to shf128 */

  if ((arg1[0] == 0 && arg1[1] == 0) || (arg2[0] == 0 && arg2[1] == 0)) {
    result[0] = 0;
    result[1] = 0;
    return;
  }
  temp_result[0] = 0;
  temp_result[1] = 0;

  if ((int)arg1[0] < 0) {
    uneg64(arg1, &temp_result[2]);
    negate = 1;
  } else {
    temp_result[2] = arg1[0];
    temp_result[3] = arg1[1];
    negate = 0;
  }

  if ((int)arg2[0] < 0) {
    uneg64(arg2, den);
    negate = !negate;
  } else {
    den[0] = arg2[0];
    den[1] = arg2[1];
  }

  one = 1;

  for (i = 0; i < 64; ++i) {
    shf128(temp_result, one, temp_result);
    if (ucmp64(temp_result, den) >= 0) {
      __utl_i_sub64(temp_result, den, temp_result);
      temp_result[3] = temp_result[3] + 1;
    }
  }

  if (negate)
    uneg64(&temp_result[2], result);
  else {
    result[0] = temp_result[2];
    result[1] = temp_result[3];
  }
}

static void uneg64(DBLUINT64 arg, DBLUINT64 result)
{
  int sign; /* sign of the low-order word of arg prior to
             * being complemented */

  sign = (unsigned)arg[1] >> 31;
  result[0] = ~arg[0];
  result[1] = (~arg[1]) + 1;
  if (sign == 0 && (int)result[1] >= 0)
    result[0]++;
}

static void ushf64(DBLUINT64 arg, int count, DBLINT64 result)
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
  if (count >= 0) {
    if (count < 32) {
      result[0] = (u_arg[0] << count) | (u_arg[1] >> 32 - count);
      result[1] = u_arg[1] << count;
    } else {
      result[0] = u_arg[1] << count - 32;
      result[1] = 0;
    }
  } else if (count > -32) {
    result[0] = arg[0] >> -count;
    result[1] = (u_arg[1] >> -count) | (u_arg[0] << count + 32);
  } else {
    result[0] = 0;
    result[1] = arg[0] >> -count - 32;
  }
}

static void shf128(INT arg[4], int count, INT result[4])
{

  /* Local variables */

  int i;         /* for loop control variable */
  int idx;       /* index into result */
  int num_bits;  /* number of bits to be shifted */
  UINT u_arg[4]; /* unsigned arg used in shift */

/* Function declarations */

#define _ABS(x) ((x) < 0 ? -(x) : (x))

  if (_ABS(count) >= 128) {
    for (i = 0; i < 4; ++i)
      result[i] = 0;
    return;
  } /* end_if */
  if (count == 0) {
    for (i = 0; i < 4; ++i)
      result[i] = arg[i];
    return;
  }
  for (i = 0; i < 4; ++i)
    u_arg[i] = arg[i];

  if (count > 0) {
    num_bits = count % 32;
    idx = 0;

    for (i = (count / 32); i < 3; ++i) {
      result[idx] = (u_arg[i] << num_bits) | (u_arg[i + 1] >> (32 - num_bits));
      idx = idx + 1;
    } /* end_for */

    result[idx] = (u_arg[3] << num_bits);

    if (idx < 3)
      for (i = idx + 1; i < 4; ++i)
        result[i] = 0;
  } else {
    num_bits = (-(count)) % 32;
    idx = 3;

    for (i = 3 - ((-(count)) / 32); i > 0; --i) {
      result[idx] = (u_arg[i] >> num_bits) | (u_arg[i - 1] << (32 - num_bits));
      idx--;
    }

    result[idx] = (u_arg[0] >> num_bits);

    if (idx > 0)
      for (i = 0; i < idx; ++i)
        result[i] = 0;
  }
}
#endif /* #ifdef MTHI64 */

