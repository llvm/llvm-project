/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "dblint64.h"

typedef union {
  DBLINT64 wd;        /* canonical msw & lsw view of long long values */
  int hf[2];       /* native msw & lsw signed view of long long values */
  unsigned uhf[2]; /* native msw & lsw unsigned view of long long values */
  long long value;
  unsigned long long uvalue;
} LL_SHAPE;

#undef FL
#undef UFL
#undef LSW
#undef MSW
#undef ULSW
#undef UMSW
#define FL(s) s.value
#define UFL(s) s.uvalue
/* Little endian view of a long long as two ints */
#define LSW(s) s.hf[0]
#define MSW(s) s.hf[1]
#define ULSW(s) s.uhf[0]
#define UMSW(s) s.uhf[1]

#define VOID void

#define UTL_I_I64RET(m, l) return (__utl_i_i64ret(m, l))
/*extern  VOID UTL_I_I64RET();*/

long long
__utl_i_i64ret(int msw, int lsw)
{
  union {
    int i[2];
    long long z;
  } uu;
  uu.i[0] = lsw;
  uu.i[1] = msw;
  return uu.z;
}

typedef int INT;
typedef unsigned int UINT;

#define __mth_i_kmul(a, b) (a) * (b)

INT __mth_i_kcmp(), __mth_i_kucmp(), __mth_i_kcmpz(), __mth_i_kucmpz();
VOID __mth_i_krshift();
VOID __mth_i_klshift();
VOID __mth_i_kurshift();
VOID __utl_i_add64(), __utl_i_div64();
VOID __utl_i_sub64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result);
VOID __utl_i_mul64(), __utl_i_udiv64();
static VOID neg64(), shf64(), shf128by1();

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
VOID __utl_i_add64(arg1, arg2, result) DBLINT64 arg1, arg2, result;
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
  return;
}

/** \brief Subtract two 64-bit integers
 *
 *  \param arg1    minuend
 *  \param arg2    subtrahend
 *  \param result  arg1 - arg2
 */
VOID
__utl_i_sub64(DBLINT64 arg1, DBLINT64 arg2, DBLINT64 result)
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
  return;
}

/*
 *
 *       Multiply two 64-bit integers to produce a 64-bit
 *       integer product.
 */
VOID __utl_i_mul64(arg1, arg2, result) DBLINT64 arg1, arg2, result;
{
  LL_SHAPE v1, v2, r;

  /* Canonical 64-bit (big endian) form -> little endian */
  v1.wd[0] = arg1[1];
  v1.wd[1] = arg1[0];

  v2.wd[0] = arg2[1];
  v2.wd[1] = arg2[0];

  r.value = __mth_i_kmul(v1.value, v2.value);

  /* Little endian form -> canonical form */
  result[0] = r.wd[1];
  result[1] = r.wd[0];
}

/*
 *  Divide two long long ints to produce a long long quotient.
 */
long long
__mth_i_kdiv(long long x, long long y)
{
  int negate; /* flag indicating the result needs to be negated */
  LL_SHAPE a, b, r;

  FL(a) = x;
  if (MSW(a) >= 0) {
    negate = 0;
  } else {
    FL(a) = -FL(a);
    negate = 1;
  }

  FL(b) = y;
  if (MSW(b) < 0) {
    FL(b) = -FL(b);
    negate = !negate;
  }

  if (MSW(a) == 0 && MSW(b) == 0) {
    MSW(r) = 0;
    *(unsigned *)&LSW(r) = (unsigned)LSW(a) / (unsigned)LSW(b);
  } else {
    DBLINT64 arg1, arg2; /* DBLINT64 is big endian!! */
    DBLINT64 result;
    arg1[1] = LSW(a);
    arg1[0] = MSW(a);
    arg2[1] = LSW(b);
    arg2[0] = MSW(b);
    __utl_i_div64(arg1, arg2, result);
    LSW(r) = result[1];
    MSW(r) = result[0];
  }

  if (negate)
    return -FL(r);
  else
    return FL(r);
}

/*
 *  Divide two unsigned long long ints to produce a unsigned long long quotient.
 */
unsigned long long
__mth_i_ukdiv(unsigned long long x, unsigned long long y)
{
  LL_SHAPE a, b, r;

  UFL(a) = x;
  UFL(b) = y;

  if (UMSW(a) == 0 && UMSW(b) == 0) {
    UMSW(r) = 0;
    ULSW(r) = ULSW(a) / ULSW(b);
  } else {
    DBLUINT64 arg1, arg2; /* DBLUINT64 is big endian!! */
    DBLUINT64 result;
    arg1[1] = ULSW(a);
    arg1[0] = UMSW(a);
    arg2[1] = ULSW(b);
    arg2[0] = UMSW(b);
    __utl_i_udiv64(arg1, arg2, result);
    ULSW(r) = result[1];
    UMSW(r) = result[0];
  }
  return UFL(r);
}

/*
 *       Divide two 64-bit integers to produce a 64-bit
 *       integer quotient.
 */
VOID __utl_i_div64(arg1, arg2, result) DBLINT64 arg1, arg2, result;

{
  DBLINT64 den;          /* denominator used in calculating the
                       * quotient */
  int i;              /* for loop control variable */
  int temp_result[4]; /* temporary result used in
                       * calculating the quotient */
  int negate;         /* flag which indicates the result needs to
                       * be negated */

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

  for (i = 0; i < 64; ++i) {
    shf128by1(temp_result, temp_result);
    if (__mth_i_kucmp(temp_result[1], temp_result[0], den[1], den[0]) >= 0) {
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

/*
 *  negate a 64-bit integer
 *
 *	Arguments:
 *	    arg     64-bit value to be negated
 *	    result  - arg
 *
 *	Return value:
 *	    none.
 */

static VOID neg64(arg, result) DBLINT64 arg, result;

{
  int sign; /* sign of the low-order word of arg prior to
             * being complemented */

  sign = (unsigned)arg[1] >> 31;
  result[0] = ~arg[0];
  result[1] = (~arg[1]) + 1;
  if (sign == 0 && result[1] >= 0)
    result[0]++;
}

/** \brief Divide two 64-bit unsigned integers to produce a 64-bit unsigned
 * integer quotient.
 */
VOID
__utl_i_udiv64(DBLUINT64 arg1, DBLUINT64 arg2, DBLUINT64 result)
{
  DBLINT64 den;          /* denominator used in calculating the
                       * quotient */
  int i;              /* for loop control variable */
  int temp_result[4]; /* temporary result used in
                       * calculating the quotient */

  if ((arg1[0] == 0 && arg1[1] == 0) || (arg2[0] == 0 && arg2[1] == 0)) {
    result[0] = 0;
    result[1] = 0;
    return;
  }
  temp_result[0] = 0;
  temp_result[1] = 0;
  temp_result[2] = arg1[0];
  temp_result[3] = arg1[1];

  den[0] = arg2[0];
  den[1] = arg2[1];

  for (i = 0; i < 64; ++i) {
    shf128by1(temp_result, temp_result);
    if (__mth_i_kucmp(temp_result[1], temp_result[0], den[1], den[0]) >= 0) {
      __utl_i_sub64(temp_result, den, temp_result);
      temp_result[3] = temp_result[3] + 1;
    }
  }

  result[0] = temp_result[2];
  result[1] = temp_result[3];
}

/* this is in assembly now. New routines are krshift,kurshift,klshift */
/* leave these in here for now.  So old objects/libs get resolved. */
/*
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
/* new version - inline shf64 */
long long __mth_i_kicshft(op1, op2, count, direct) UINT op1,
    op2; /* really INT */
INT count, direct;
{
  DBLINT64 result;
  if (count < 0 || count >= 64) {
    UTL_I_I64RET(0, 0);
  }
  if (count == 0) {
    result[0] = op2;
    result[1] = op1;
    UTL_I_I64RET(result[0], result[1]);
  }
  if (direct) {/* right shift */
    if (count < 32) {
      result[0] = (INT)op2 >> count; /* sign extend */
      result[1] = (op1 >> count) | (op2 << (32 - count));
    } else {
      result[0] = (INT)op2 >> 31; /* sign extend */
      result[1] = (INT)op2 >> (count - 32);
    }
  } else {/* left ushift */
    if (count < 32) {
      result[0] = (op2 << count) | (op1 >> (32 - count));
      result[1] = op1 << count;
    } else {
      result[0] = op1 << (count - 32);
      result[1] = 0;
    }
  }
  UTL_I_I64RET(result[0], result[1]);
}

/* new version - inline ushf64 */
long long __mth_i_ukicshft(op1, op2, count, direct) UINT op1, op2;
INT count, direct;
{
  DBLINT64 result;
  if (count < 0 || count >= 64) {
    UTL_I_I64RET(0, 0);
  }
  if (count == 0) {
    result[0] = op2;
    result[1] = op1;
    UTL_I_I64RET(result[0], result[1]);
  }
  if (direct) {/* right ushift */
    if (count < 32) {
      result[0] = op2 >> count;
      result[1] = (op1 >> count) | (op2 << (32 - count));
    } else {
      result[0] = 0;
      result[1] = op2 >> (count - 32);
    }
  } else {/* left ushift */
    if (count < 32) {
      result[0] = (op2 << count) | (op1 >> (32 - count));
      result[1] = op1 << count;
    } else {
      result[0] = op1 << (count - 32);
      result[1] = 0;
    }
  }
  UTL_I_I64RET(result[0], result[1]);
}

long long __mth_i_kishft(op1, op2, arg2) INT op1, op2, arg2;
{
  DBLINT64 arg1;
  DBLINT64 result;
  arg1[1] = op1;
  arg1[0] = op2;
  shf64(arg1, arg2, result);
  UTL_I_I64RET(result[0], result[1]);
}

static VOID shf64(arg, count, result) DBLINT64 arg;
int count;
DBLINT64 result;
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

static VOID shf128by1(arg, result) INT arg[4];
INT result[4];
{
  UINT u_arg[4];
  int i; /* for loop control variable */

  /* to get unsigned, and because arg and result may be same loc */
  for (i = 0; i < 4; ++i)
    u_arg[i] = arg[i];

  for (i = 0; i < 3; ++i) {
    result[i] = (u_arg[i] << 1) | (u_arg[i + 1] >> (32 - 1));
  } /* end_for */
  result[3] = (u_arg[3] << 1);
}

/* this is all in assembly now. */
/*
 *  compare 64-bit unsigned integer against zero
 *
 *	Arguments:
 *	    arg1   operand 1 of compare
 *
 *	Return value:
 *           0 = operand equal to zero
 *           1 = operand greater than zero (ie. nonzero)
 */
INT __mth_i_kucmpz(op1, op2) register UINT op1, op2;
{
  if ((op1 | op2) == 0)
    return 0;
  return 1;
}

/*
 *  compare two 64-bit unsigned integers
 *
 *	Arguments:
 *	    arg1   operand 1 of compare
 *	    arg2   operand 2 of compare
 *
 *	Return value:
 *          -1 = first operand less than second
 *           0 = first operand equal to second
 *           1 = first operand greater than second
 */
INT __mth_i_kucmp(op1, op2, op3, op4) UINT op1, op2, op3, op4;
{
  if (op2 == op4) {
    if (op1 == op3)
      return 0;
    if (op1 < op3)
      return -1;
    return 1;
  }
  if (op2 < op4)
    return -1;
  return 1;
}

/*
 *  compare 64-bit signed integer against zero
 */
INT __mth_i_kcmpz(op1, op2) INT op1, op2;
{
  if (op2 == 0) {
    if (op1 == 0)
      return 0; /* is zero */
    return 1;   /* > zero  */
  }
  if (op2 < 0)
    return -1; /* < zero */
  return 1;    /* > zero */
}

INT __mth_i_kcmp(op1, op2, op3, op4) INT op1, op2, op3, op4;
{
  if (op2 == op4) {
    if (op1 == op3)
      return 0;
    if ((unsigned)op1 < (unsigned)op3)
      return -1;
    return 1;
  }
  if (op2 < op4)
    return -1;
  return 1;
}

/**********************************************************
 *
 *	Unpacked floating point routines.
 *	These routines will convert IEEE
 *	floating point to an unpacked internal
 *	format and back again.
 *
 *********************************************************/

/****************************************************************
 *
 * The IEEE floating point format is as follows:
 *
 *	single:
 *		  31  30       23       22                   0
 *              +----+-----------+     +----------------------+
 *         W0   |sign| exp + 127 |  1. |       mantissa       |
 *              +----+-----------+     +----------------------+
 *
 *	double:
 *                31  30         20       19                 0
 *              +----+-------------+     +--------------------+
 *         W0   |sign| exp + 1023  |  1. |      mantissa      |
 *              +----+-------------+     +--------------------+
 *               31                                          0
 *              +---------------------------------------------+
 *         W1   |                  mantissa                   |
 *              +---------------------------------------------+
 *
 ***************************************************************/

/*  define bit manipulation macros:  */

#define bitmask(w) ((UINT)(1UL << (w)) - 1)
#define extract(i, a, b) (((INT)(i) >> (b)) & bitmask((a) - (b) + 1))
#define bit_fld(i, a, b) (((INT)(i)&bitmask((a) - (b) + 1)) << (b))
#define bit_insrt(x, i, a, b)	x = (((INT)(x) & ~bit_fld(~0L,(a),(b)))

typedef int IEEE32;    /* IEEE single precision float number */
typedef int IEEE64[2]; /* IEEE double precision float number */

#define IEEE32_sign(f) extract(f, 31, 31)
#define IEEE32_exp(f) (extract(f, 30, 23) - 127)
#define IEEE32_man(f) (bit_fld(1, 23, 23) | extract(f, 22, 0))
#define IEEE32_zero(f) (extract(f, 30, 0) == 0)
#define IEEE32_infin(f) (extract(f, 30, 0) == bit_fld(255, 30, 23))
#define IEEE32_nan(f) (!IEEE32_infin(f) && IEEE32_exp(f) == 128)
#define IEEE32_pack(f, sign, exp, man)                                         \
  f = bit_fld(sign, 31, 31) | bit_fld(exp + 127, 30, 23) |                     \
      bit_fld(man[0], 22, 0)

#define IEEE64_sign(d) extract(d[1], 31, 31)
#define IEEE64_exp(d) (extract(d[1], 30, 20) - 1023)
#define IEEE64_manhi(d) (bit_fld(1, 20, 20) | extract(d[1], 19, 0))
#define IEEE64_manlo(d) (d[0])
#define IEEE64_zero(d) (extract(d[1], 30, 0) == 0L && IEEE64_manlo(d) == 0L)
#define IEEE64_infin(d)                                                        \
  (extract(d[1], 30, 0) == bit_fld(255, 30, 23) && d[0] == 0)
#define IEEE64_nan(d) (!IEEE64_infin(d) && IEEE64_exp(d) == 1024)
#define IEEE64_pack(d, sign, exp, man)                                         \
  d[1] = bit_fld(sign, 31, 31) | bit_fld(exp + 1023, 30, 20) |                 \
         bit_fld(man[0], 19, 0);                                               \
  d[0] = man[1]

/****************************************************************
 *
 * The unpacked floating point format is as follows:
 *
 *              +-----------+   +-----------+   +-------------+
 *              |    val    |   |   sign    |   |     exp     |
 *              +-----------+   +-----------+   +-------------+
 *               31              20       19                 0
 *              +------------------+     +--------------------+
 *         W0   |     mantissa     |  .  |      mantissa      |
 *              +------------------+     +--------------------+
 *               31                                          0
 *              +---------------------------------------------+
 *         W1   |                  mantissa                   |
 *              +---------------------------------------------+
 *               31                                          0
 *              +---------------------------------------------+
 *         W2   |                  mantissa                   |
 *              +---------------------------------------------+
 *               31                                          0
 *              +---------------------------------------------+
 *         W3   |                  mantissa                   |
 *              +---------------------------------------------+
 *
 ***************************************************************/

typedef enum { ZERO, NIL, NORMAL, BIG, INFIN, NAN, DIVZ } VAL;

#define POS 0
#define NEG 1

typedef struct {
  VAL fval;    /* Value */
  int fsgn;    /* Sign */
  int fexp;    /* Exponent */
  INT fman[4]; /* Mantissa */
} UFP;

static VOID manadd(m1, m2)
    /* add mantissas of two unpacked floating point numbers. */
    INT m1[4]; /* first and result */
INT m2[4];     /* second */
{
  INT t1, t2, carry;
  INT lo, hi;
  int i;

  carry = 0;
  for (i = 3; i >= 0; i--) {
    /* add low halves + carry */
    t1 = m1[i] & 0x0000FFFFL;
    t2 = m2[i] & 0x0000FFFFL;
    lo = t1 + t2 + carry;
    carry = (lo >> 16) & 0x0000FFFFL;
    lo &= 0x0000FFFFL;

    /* add high halves + carry */
    t1 = (m1[i] >> 16) & 0x0000FFFFL;
    t2 = (m2[i] >> 16) & 0x0000FFFFL;
    hi = t1 + t2 + carry;
    carry = (hi >> 16) & 0x0000FFFFL;
    hi <<= 16;

    /* merge halves */
    m1[i] = hi | lo;
  }
  /* we assume no carry is
   * generated from the last
   * add */
}

static VOID manshftl(m, n)
    /* shift mantissa left */
    INT m[4]; /* number */
int n;        /* amount to shift */
{
  register int i;
  register int j;
  int mask;

  /* do whole word shifts first */
  for (i = n; i >= 32; i -= 32) {
    m[0] = m[1];
    m[1] = m[2];
    m[2] = m[3];
    m[3] = 0L;
  }
  if (i > 0) {
    j = 32 - i;
    mask = bitmask(i);
    m[0] = (m[0] << i) | ((m[1] >> j) & mask);
    m[1] = (m[1] << i) | ((m[2] >> j) & mask);
    m[2] = (m[2] << i) | ((m[3] >> j) & mask);
    m[3] = (m[3] << i);
  }
}

static VOID manshftr(m, n)
    /* shift mantissa right */
    INT m[4]; /* number */
int n;        /* amount to shift */
{
  register int i;
  register int j;
  int mask;

  /* do whole word shifts first */
  for (i = n; i >= 32; i -= 32) {
    m[3] = m[2];
    m[2] = m[1];
    m[1] = m[0];
    m[0] = 0L;
  }
  if (i > 0) {
    j = 32 - i;
    mask = bitmask(j);
    m[3] = ((m[3] >> i) & mask) | (m[2] << j);
    m[2] = ((m[2] >> i) & mask) | (m[1] << j);
    m[1] = ((m[1] >> i) & mask) | (m[0] << j);
    m[0] = (m[0] >> i) & mask;
  }
}

static VOID manrnd(m, bits)
    /* Round the mantissa to the number of bits specified. */
    INT m[4]; /* Mantissa to round */
int bits;     /* Number of bits of precision */
{
  int rndwrd, rndbit;
  int oddwrd, oddbit;
  INT round[4];
  static INT one[4] = {0L, 0L, 0L, 1L};

  rndwrd = bits / 32;
  rndbit = 31 - (bits % 32);
  oddwrd = (bits - 1) / 32;
  oddbit = 31 - ((bits - 1) % 32);

  /* check round bit */
  if (extract(m[rndwrd], rndbit, rndbit) == 1) {
    round[0] = 0xFFFFFFFFL;
    round[1] = 0xFFFFFFFFL;
    round[2] = 0xFFFFFFFFL;
    round[3] = 0xFFFFFFFFL;
    manshftr(round, bits + 1);
    /* Round up by just under
     * 1/2.
     */
    manadd(m, round);

    /* If exactly 1/2 then round
     * only if it's an odd number.
     */
    if (extract(m[rndwrd], rndbit, rndbit) == 1 &&
        extract(m[oddwrd], oddbit, oddbit) == 1) {
      manadd(m, one);
    }
  }
  /* Get rid of rounded off bits. */
  manshftr(m, 128 - bits);
  manshftl(m, 128 - bits);
}

static VOID ufpnorm(u)
    /* normalize the unpacked number. */
    register UFP *u; /* unpacked number */
{
  /* If it's zero then there's no
   * need to normalize. */
  if (u->fman[0] == 0 && u->fman[1] == 0 && u->fman[2] == 0 && u->fman[3] == 0)
    return;

  /* We may have to right shift */
  while (extract(u->fman[0], 31, 21) != 0) {
    manshftr(u->fman, 1);
    u->fexp++;
  }

  /* repeatedly left shift and decrement
   * exp until normalized */
  while (extract(u->fman[0], 20, 20) == 0) {
    manshftl(u->fman, 1);
    u->fexp--;
  }
}

static VOID ufprnd(u, bits)
    /* Round the unpacked floating point to the number of bits of binary
     * fraction specified.
     */
    UFP *u; /* Number to round */
int bits;   /* Number of bits of precision */
{
  VOID ufpnorm();

  ufpnorm(u);
  manrnd(u->fman, bits + 12); /* Binary point is 12 bits over */
  ufpnorm(u);                 /* Might have denormalized it.  */
}

/** \brief Floating point double unpack
 *  \param d double precision number
 *  \param u pointer to area for unpack
*/
static VOID
dtoufp(IEEE64 d, register UFP *u)
{
  u->fval = NORMAL;
  u->fexp = IEEE64_exp(d);
  u->fsgn = IEEE64_sign(d);
  u->fman[0] = IEEE64_manhi(d);
  u->fman[1] = IEEE64_manlo(d);
  u->fman[2] = 0L;
  u->fman[3] = 0L;

  /* special case checks */
  if (IEEE64_nan(d))
    u->fval = NAN;

  if (IEEE64_infin(d))
    u->fval = INFIN;

  if (IEEE64_zero(d)) {
    u->fval = ZERO;
    u->fexp = 0;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }
}

static VOID ftoufp(f, u)
    /* -- floating point single unpack */
    IEEE32 *f;   /* single precision number */
register UFP *u; /* unpacked result */
{
  u->fval = NORMAL;
  u->fexp = IEEE32_exp(*f);
  u->fsgn = IEEE32_sign(*f);
  u->fman[0] = IEEE32_man(*f);
  u->fman[1] = 0L;
  u->fman[2] = 0L;
  u->fman[3] = 0L;
  manshftr(u->fman, 3);

  /* special case checks */
  if (IEEE32_nan(*f))
    u->fval = NAN;

  if (IEEE32_infin(*f))
    u->fval = INFIN;

  if (IEEE32_zero(*f)) {
    u->fval = ZERO;
    u->fexp = 0;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }
}

static VOID i64toufp(i, u)
    /* 64-bit integer to unpacked float */
    DBLINT64 i;
UFP *u;
{
  DBLINT64 tmp;

  if (i[0] == 0L && i[1] == 0L) {
    u->fsgn = POS;
    u->fval = ZERO;
    u->fexp = 0;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
    return;
  }
  tmp[0] = i[0];
  tmp[1] = i[1];
  u->fval = NORMAL;
  u->fexp = 52;
  if ((*i & 0x80000000L) != 0) {
    u->fsgn = NEG;
    neg64(i, tmp);
  } else
    u->fsgn = POS;
  u->fman[0] = tmp[0];
  u->fman[1] = tmp[1];
  u->fman[2] = 0L;
  u->fman[3] = 0L;
}

static VOID ufptod(u, r)
    /* floating point double pack */
    register UFP *u; /* unpacked number */
IEEE64 r;            /* packed result */
{
  /* Round and normalize the unpacked
   * number first. */
  ufprnd(u, 52);

  if (u->fval == ZERO || u->fval == NIL) {/* zero */
    u->fexp = -1023;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  if (u->fval == NAN) {/* Not a number */
    u->fexp = 1024;
    u->fman[0] = ~0L;
    u->fman[1] = ~0L;
  }

  if (u->fval == INFIN || u->fval == BIG || u->fval == DIVZ) {/* infinity */
    u->fexp = 1024;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  if (u->fval == NORMAL && u->fexp <= -1023) {/* underflow */
    u->fval = NIL;
    u->fexp = -1023;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  if (u->fval == NORMAL && u->fexp >= 1024) {/* overflow */
    u->fval = BIG;
    u->fexp = 1024;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  IEEE64_pack(r, u->fsgn, u->fexp, u->fman);
}

static VOID ufptof(u, r)
    /* unpacked floating point single float */
    register UFP *u; /* unpacked number */
IEEE32 *r;           /* packed result */
{
  /* Round and normalize the unpacked
   * number first.  */
  ufprnd(u, 23);
  manshftl(u->fman, 3);

  if (u->fval == ZERO || u->fval == NIL) {/* zero */
    u->fexp = -127;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  if (u->fval == INFIN || u->fval == BIG || u->fval == DIVZ) {/* infinity */
    u->fexp = 128;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  if (u->fval == NAN) {/* Not a number */
    u->fexp = 128;
    u->fman[0] = ~0L;
    u->fman[1] = ~0L;
  }

  if (u->fval == NORMAL && u->fexp <= -127) {/* underflow */
    u->fval = NIL;
    u->fexp = -127;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  if (u->fval == NORMAL && u->fexp >= 128) {/* overflow */
    u->fval = BIG;
    u->fexp = 128;
    u->fman[0] = 0L;
    u->fman[1] = 0L;
  }

  IEEE32_pack(*r, u->fsgn, u->fexp, u->fman);
}

static VOID ufptoi64(u, i)
    /* unpacked float to 64-bit integer */
    UFP *u;
DBLINT64 i;
{
  /* Normalize the unpacked
   * number first.  */
  ufpnorm(u);
  if (u->fexp - 52 > 0)
    manshftl(u->fman, u->fexp - 52);
  else
    manshftr(u->fman, 52 - u->fexp);

  if (u->fval == ZERO || u->fval == NIL) {
    i[0] = 0;
    i[1] = 0;
    return;
  }

  if (u->fval == NAN) {/* Not a number */
    i[0] = 0;
    i[1] = 0;
    return;
  }

  if (u->fval == INFIN || u->fval == BIG || u->fexp > 62 ||
      ((u->fman[0] & 0x80000000L) != 0L && u->fman[1] == 0L)) {/* overflow */
    u->fval = BIG;
    if (u->fsgn == NEG) {
      i[0] = 0x80000000L;
      i[1] = 0x00000000L;
    } else {
      i[0] = 0x7FFFFFFFL;
      i[1] = 0xFFFFFFFFL;
    }
    return;
  }

  i[0] = u->fman[0];
  i[1] = u->fman[1];
  if (u->fsgn == NEG)
    neg64(i, i);
}

VOID __utl_i_dfix64(d, i)
    /* double precision to 64-bit integer */
    double d; /*IEEE64 format and double are LITTLE_ENDIAN */
DBLINT64 i;
{
  UFP u;

  dtoufp((int*)&d, &u);
  ufptoi64(&u, i);
}

double __utl_i_dflt64(i)
    /* 64 -- 64-bit integer to double */
    DBLINT64 i;
{
  UFP u;
  IEEE64 d;

  i64toufp(i, &u);
  ufptod(&u, d);
  return *((double *)d); /*IEEE64 format and double are LITTLE_ENDIAN */
}

VOID __utl_i_fix64(float ff, DBLINT64 i) /* use prototype to pass as float */
/* single float to 64-bit */
{
  IEEE32 f;
  UFP u;

  f = *(IEEE32 *)&ff; /*IEEE32 format and float are LITTLE_ENDIAN*/
  ftoufp(&f, &u);
  ufptoi64(&u, i);
}

float __utl_i_flt64(DBLINT64 i) /* use prototype to return as float */
/* 64-bit integer to single precision */
{
  UFP u;
  IEEE32 f;

  i64toufp(i, &u);
  ufptof(&u, &f);
  return *((float *)&f); /*IEEE32 format and float are LITTLE_ENDIAN */
}
