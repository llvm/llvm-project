/* Test floating-point arithmetic operations.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fenv.h>
#include <assert.h>

#ifndef ESIZE
typedef double tocheck_t;
#define ESIZE 11
#define MSIZE 52
#define FUNC(x) x
#endif

#define R_NEAREST 1
#define R_ZERO 2
#define R_UP 4
#define R_DOWN 8
#define R_ALL (R_NEAREST|R_ZERO|R_UP|R_DOWN)
static fenv_t rmodes[4];
static const char * const rmnames[4] =
{ "nearest","zero","+Inf","-Inf" };

typedef union {
  tocheck_t tc;
  unsigned char c[sizeof (tocheck_t)];
} union_t;

/* Don't try reading these in a font that doesn't distinguish
   O and zero.  */
typedef enum {
  P_Z    = 0x0,  /* 00000...0 */
  P_000O = 0x1,  /* 00011...1 */
  P_001Z = 0x2,  /* 00100...0 */
  P_00O  = 0x3,  /* 00111...1 */
  P_01Z  = 0x4,  /* 01000...0 */
  P_010O = 0x5,  /* 01011...1 */
  P_011Z = 0x6,  /* 01100...0 */
  P_0O   = 0x7,  /* 01111...1 */
  P_1Z   = 0x8,  /* 10000...0 */
  P_100O = 0x9,  /* 10011...1 */
  P_101Z = 0xa,  /* 10100...0 */
  P_10O  = 0xb,  /* 10111...1 */
  P_11Z  = 0xc,  /* 11000...0 */
  P_110O = 0xd,  /* 11011...1 */
  P_111Z = 0xe,  /* 11100...0 */
  P_O    = 0xf,  /* 11111...1 */
  P_Z1   = 0x11, /* 000...001 */
  P_Z10  = 0x12, /* 000...010 */
  P_Z11  = 0x13, /* 000...011 */
  P_0O00 = 0x14, /* 011...100 */
  P_0O01 = 0x15, /* 011...101 */
  P_0O0  = 0x16, /* 011...110 */
  P_1Z1  = 0x19, /* 100...001 */
  P_1Z10 = 0x1a, /* 100...010 */
  P_1Z11 = 0x1b, /* 100...011 */
  P_O00  = 0x1c, /* 111...100 */
  P_O01  = 0x1d, /* 111...101 */
  P_O0   = 0x1e, /* 111...110 */
  P_R    = 0x20, /* rrr...rrr */ /* ('r' means random. ) */
  P_Ro   = 0x21, /* rrr...rrr, with odd parity.  */
  P_0R   = 0x22, /* 0rr...rrr */
  P_1R   = 0x23, /* 1rr...rrr */
  P_Rno  = 0x24, /* rrr...rrr, but not all ones.  */
} pattern_t;

static void
pattern_fill(pattern_t ptn, unsigned char *start, int bitoffset, int count)
{
#define bitset(count, value) \
      start[(count)/8] = (start[(count)/8] & ~(1 << 7-(count)%8)  \
                          |  (value) << 7-(count)%8)
  int i;

  if (ptn >= 0 && ptn <= 0xf)
    {
      /* Patterns between 0 and 0xF have the following format:
	 The LSBit is used to fill the last n-3 bits of the pattern;
	 The next 3 bits are the first 3 bits of the pattern. */
      for (i = 0; i < count; i++)
	if (i < 3)
	  bitset((bitoffset+i), ptn >> (3-i) & 1);
	else
	  bitset((bitoffset+i), ptn >> 0 & 1);
    }
  else if (ptn <= 0x1f)
    {
      /* Patterns between 0x10 and 0x1F have the following format:
	 The two LSBits are the last two bits of the pattern;
	 The 0x8 bit is the first bit of the pattern;
	 The 0x4 bit is used to fill the remainder. */
      for (i = 0; i < count; i++)
	if (i == 0)
	  bitset((bitoffset+i), ptn >> 3 & 1);
	else if (i >= count-2)
	  bitset((bitoffset+i), ptn >> (count-1-i) & 1);
	else
	  bitset((bitoffset+i), ptn >> 2 & 1);
    }
  else switch (ptn)
    {
    case P_0R: case P_1R:
      assert(count > 0);
      bitset(bitoffset, ptn & 1);
      count--;
      bitoffset++;
    case P_R:
      for (; count > 0; count--, bitoffset++)
	bitset(bitoffset, rand() & 1);
      break;
    case P_Ro:
      {
	int op = 1;
	assert(count > 0);
	for (; count > 1; count--, bitoffset++)
	  bitset(bitoffset, op ^= (rand() & 1));
	bitset(bitoffset, op);
	break;
      }
    case P_Rno:
      {
	int op = 1;
	assert(count > 0);
	for (; count > 1; count--, bitoffset++)
	{
	  int r = rand() & 1;
	  op &= r;
	  bitset(bitoffset, r);
	}
	bitset(bitoffset, rand() & (op ^ 1));
	break;
      }

    default:
      assert(0);
    }
#undef bitset
}

static tocheck_t
pattern(int negative, pattern_t exp, pattern_t mant)
{
  union_t result;
#if 0
  int i;
#endif

  pattern_fill(negative ? P_O : P_Z, result.c, 0, 1);
  pattern_fill(exp, result.c, 1, ESIZE);
  pattern_fill(mant, result.c, ESIZE+1, MSIZE);
#if 0
  printf("neg=%d exp=%02x mant=%02x: ", negative, exp, mant);
  for (i = 0; i < sizeof (tocheck_t); i++)
    printf("%02x", result.c[i]);
  printf("\n");
#endif
  return result.tc;
}

/* Return the closest different tocheck_t to 'x' in the direction of
   'direction', or 'x' if there is no such value.  Assumes 'x' is not
   a NaN.  */
static tocheck_t
delta(tocheck_t x, int direction)
{
  union_t xx;
  int i;

  xx.tc = x;
  if (xx.c[0] & 0x80)
    direction = -direction;
  if (direction == +1)
    {
      union_t tx;
      tx.tc = pattern(xx.c[0] >> 7, P_O, P_Z);
      if (memcmp (tx.c, xx.c, sizeof (tocheck_t)) == 0)
	return x;
    }
  for (i = sizeof (tocheck_t)-1; i > 0; i--)
    {
      xx.c[i] += direction;
      if (xx.c[i] != (direction > 0 ? 0 : 0xff))
	return xx.tc;
    }
  if (direction < 0 && (xx.c[0] & 0x7f) == 0)
    return pattern(~(xx.c[0] >> 7) & 1, P_Z, P_Z1);
  else
    {
      xx.c[0] += direction;
      return xx.tc;
    }
}

static int nerrors = 0;

#ifdef FE_ALL_INVALID
static const int all_exceptions = FE_ALL_INVALID | FE_ALL_EXCEPT;
#else
static const int all_exceptions = FE_ALL_EXCEPT;
#endif

static void
check_result(int line, const char *rm, tocheck_t expected, tocheck_t actual)
{
  if (memcmp (&expected, &actual, sizeof (tocheck_t)) != 0)
    {
      unsigned char *ex, *ac;
      size_t i;

      printf("%s:%d:round %s:result failed\n"
	     " expected result 0x", __FILE__, line, rm);
      ex = (unsigned char *)&expected;
      ac = (unsigned char *)&actual;
      for (i = 0; i < sizeof (tocheck_t); i++)
	printf("%02x", ex[i]);
      printf(" got 0x");
      for (i = 0; i < sizeof (tocheck_t); i++)
	printf("%02x", ac[i]);
      printf("\n");
      nerrors++;
    }
}

static const struct {
  int except;
  const char *name;
} excepts[] = {
#define except_entry(ex) { ex, #ex } ,
#ifdef FE_INEXACT
  except_entry(FE_INEXACT)
#else
# define FE_INEXACT 0
#endif
#ifdef FE_DIVBYZERO
  except_entry(FE_DIVBYZERO)
#else
# define FE_DIVBYZERO 0
#endif
#ifdef FE_UNDERFLOW
  except_entry(FE_UNDERFLOW)
#else
# define FE_UNDERFLOW 0
#endif
#ifdef FE_OVERFLOW
  except_entry(FE_OVERFLOW)
#else
# define FE_OVERFLOW 0
#endif
#ifdef FE_INVALID
  except_entry(FE_INVALID)
#else
# define FE_INVALID 0
#endif
#ifdef FE_INVALID_SNAN
  except_entry(FE_INVALID_SNAN)
#else
# define FE_INVALID_SNAN FE_INVALID
#endif
#ifdef FE_INVALID_ISI
  except_entry(FE_INVALID_ISI)
#else
# define FE_INVALID_ISI FE_INVALID
#endif
#ifdef FE_INVALID_IDI
  except_entry(FE_INVALID_IDI)
#else
# define FE_INVALID_IDI FE_INVALID
#endif
#ifdef FE_INVALID_ZDZ
  except_entry(FE_INVALID_ZDZ)
#else
# define FE_INVALID_ZDZ FE_INVALID
#endif
#ifdef FE_INVALID_COMPARE
  except_entry(FE_INVALID_COMPARE)
#else
# define FE_INVALID_COMPARE FE_INVALID
#endif
#ifdef FE_INVALID_SOFTWARE
  except_entry(FE_INVALID_SOFTWARE)
#else
# define FE_INVALID_SOFTWARE FE_INVALID
#endif
#ifdef FE_INVALID_SQRT
  except_entry(FE_INVALID_SQRT)
#else
# define FE_INVALID_SQRT FE_INVALID
#endif
#ifdef FE_INVALID_INTEGER_CONVERSION
  except_entry(FE_INVALID_INTEGER_CONVERSION)
#else
# define FE_INVALID_INTEGER_CONVERSION FE_INVALID
#endif
};

static int excepts_missing = 0;

static void
check_excepts(int line, const char *rm, int expected, int actual)
{
  if (expected & excepts_missing)
    expected = expected & ~excepts_missing | FE_INVALID_SNAN;
  if ((expected & all_exceptions) != actual)
    {
      size_t i;
      printf("%s:%d:round %s:exceptions failed\n"
	     " expected exceptions ", __FILE__, line,rm);
      for (i = 0; i < sizeof (excepts) / sizeof (excepts[0]); i++)
	if (expected & excepts[i].except)
	  printf("%s ",excepts[i].name);
      if ((expected & all_exceptions) == 0)
	printf("- ");
      printf("got");
      for (i = 0; i < sizeof (excepts) / sizeof (excepts[0]); i++)
	if (actual & excepts[i].except)
	  printf(" %s",excepts[i].name);
      if ((actual & all_exceptions) == 0)
	printf("- ");
      printf(".\n");
      nerrors++;
    }
}

typedef enum {
  B_ADD, B_SUB, B_MUL, B_DIV, B_NEG, B_ABS, B_SQRT
} op_t;
typedef struct {
  int line;
  op_t op;
  int a_sgn;
  pattern_t a_exp, a_mant;
  int b_sgn;
  pattern_t b_exp, b_mant;
  int rmode;
  int excepts;
  int x_sgn;
  pattern_t x_exp, x_mant;
} optest_t;
static const optest_t optests[] = {
  /* Additions of zero.  */
  {__LINE__,B_ADD, 0,P_Z,P_Z, 0,P_Z,P_Z, R_ALL,0, 0,P_Z,P_Z },
  {__LINE__,B_ADD, 1,P_Z,P_Z, 0,P_Z,P_Z, R_ALL & ~R_DOWN,0, 0,P_Z,P_Z },
  {__LINE__,B_ADD, 1,P_Z,P_Z, 0,P_Z,P_Z, R_DOWN,0, 1,P_Z,P_Z },
  {__LINE__,B_ADD, 1,P_Z,P_Z, 1,P_Z,P_Z, R_ALL,0, 1,P_Z,P_Z },

  /* Additions with NaN.  */
  {__LINE__,B_ADD, 0,P_O,P_101Z, 0,P_Z,P_Z, R_ALL,0, 0,P_O,P_101Z },
  {__LINE__,B_ADD, 0,P_O,P_01Z, 0,P_Z,P_Z, R_ALL,
   FE_INVALID | FE_INVALID_SNAN, 0,P_O,P_11Z },
  {__LINE__,B_ADD, 0,P_O,P_Z, 0,P_O,P_0O, R_ALL,
   FE_INVALID | FE_INVALID_SNAN, 0,P_O,P_O },
  {__LINE__,B_ADD, 0,P_Z,P_Z, 0,P_O,P_11Z, R_ALL,0, 0,P_O,P_11Z },
  {__LINE__,B_ADD, 0,P_O,P_001Z, 0,P_O,P_001Z, R_ALL,
   FE_INVALID | FE_INVALID_SNAN, 0,P_O,P_101Z },
  {__LINE__,B_ADD, 0,P_O,P_1Z, 0,P_Z,P_Z, R_ALL,0, 0,P_O,P_1Z },
  {__LINE__,B_ADD, 0,P_0O,P_Z, 0,P_O,P_10O, R_ALL,0, 0,P_O,P_10O },

  /* Additions with infinity.  */
  {__LINE__,B_ADD, 0,P_O,P_Z, 0,P_Z,P_Z, R_ALL,0, 0,P_O,P_Z },
  {__LINE__,B_ADD, 0,P_O,P_Z, 1,P_Z,P_Z, R_ALL,0, 0,P_O,P_Z },
  {__LINE__,B_ADD, 1,P_O,P_Z, 0,P_Z,P_Z, R_ALL,0, 1,P_O,P_Z },
  {__LINE__,B_ADD, 1,P_O,P_Z, 1,P_Z,P_Z, R_ALL,0, 1,P_O,P_Z },
  {__LINE__,B_ADD, 0,P_O,P_Z, 0,P_O,P_Z, R_ALL,0, 0,P_O,P_Z },
  {__LINE__,B_ADD, 1,P_O,P_Z, 1,P_O,P_Z, R_ALL,0, 1,P_O,P_Z },
  {__LINE__,B_ADD, 0,P_O,P_Z, 1,P_O,P_Z, R_ALL,
   FE_INVALID | FE_INVALID_ISI, 0,P_O,P_1Z },
  {__LINE__,B_ADD, 1,P_O,P_Z, 0,P_O,P_Z, R_ALL,
   FE_INVALID | FE_INVALID_ISI, 0,P_O,P_1Z },
  {__LINE__,B_ADD, 0,P_O,P_Z, 0,P_0O,P_Z, R_ALL,0, 0,P_O,P_Z },
  {__LINE__,B_ADD, 1,P_O,P_Z, 0,P_0O,P_Z, R_ALL,0, 1,P_O,P_Z },
  {__LINE__,B_ADD, 0,P_O,P_Z, 1,P_0O,P_Z, R_ALL,0, 0,P_O,P_Z },
  {__LINE__,B_ADD, 1,P_O,P_Z, 1,P_0O,P_Z, R_ALL,0, 1,P_O,P_Z },

  /* Overflow (and zero).  */
  {__LINE__,B_ADD, 0,P_O0,P_Z, 0,P_O0,P_Z, R_NEAREST | R_UP,
   FE_INEXACT | FE_OVERFLOW, 0,P_O,P_Z },
  {__LINE__,B_ADD, 0,P_O0,P_Z, 0,P_O0,P_Z, R_ZERO | R_DOWN,
   FE_INEXACT | FE_OVERFLOW, 0,P_O0,P_O },
  {__LINE__,B_ADD, 1,P_O0,P_Z, 1,P_O0,P_Z, R_NEAREST | R_DOWN,
   FE_INEXACT | FE_OVERFLOW, 1,P_O,P_Z },
  {__LINE__,B_ADD, 1,P_O0,P_Z, 1,P_O0,P_Z, R_ZERO | R_UP,
   FE_INEXACT | FE_OVERFLOW, 1,P_O0,P_O },
  {__LINE__,B_ADD, 0,P_O0,P_Z, 1,P_O0,P_Z, R_ALL & ~R_DOWN,
   0, 0,P_Z,P_Z },
  {__LINE__,B_ADD, 0,P_O0,P_Z, 1,P_O0,P_Z, R_DOWN,
   0, 1,P_Z,P_Z },

  /* Negation.  */
  {__LINE__,B_NEG, 0,P_Z,P_Z,   0,0,0, R_ALL, 0, 1,P_Z,P_Z },
  {__LINE__,B_NEG, 1,P_Z,P_Z,   0,0,0, R_ALL, 0, 0,P_Z,P_Z },
  {__LINE__,B_NEG, 0,P_O,P_Z,   0,0,0, R_ALL, 0, 1,P_O,P_Z },
  {__LINE__,B_NEG, 1,P_O,P_Z,   0,0,0, R_ALL, 0, 0,P_O,P_Z },
  {__LINE__,B_NEG, 0,P_O,P_1Z,  0,0,0, R_ALL, 0, 1,P_O,P_1Z },
  {__LINE__,B_NEG, 1,P_O,P_1Z,  0,0,0, R_ALL, 0, 0,P_O,P_1Z },
  {__LINE__,B_NEG, 0,P_O,P_01Z, 0,0,0, R_ALL, 0, 1,P_O,P_01Z },
  {__LINE__,B_NEG, 1,P_O,P_01Z, 0,0,0, R_ALL, 0, 0,P_O,P_01Z },
  {__LINE__,B_NEG, 0,P_1Z,P_1Z1, 0,0,0, R_ALL, 0, 1,P_1Z,P_1Z1 },
  {__LINE__,B_NEG, 1,P_1Z,P_1Z1, 0,0,0, R_ALL, 0, 0,P_1Z,P_1Z1 },
  {__LINE__,B_NEG, 0,P_Z,P_Z1,  0,0,0, R_ALL, 0, 1,P_Z,P_Z1 },
  {__LINE__,B_NEG, 1,P_Z,P_Z1,  0,0,0, R_ALL, 0, 0,P_Z,P_Z1 },

  /* Absolute value.  */
  {__LINE__,B_ABS, 0,P_Z,P_Z,   0,0,0, R_ALL, 0, 0,P_Z,P_Z },
  {__LINE__,B_ABS, 1,P_Z,P_Z,   0,0,0, R_ALL, 0, 0,P_Z,P_Z },
  {__LINE__,B_ABS, 0,P_O,P_Z,   0,0,0, R_ALL, 0, 0,P_O,P_Z },
  {__LINE__,B_ABS, 1,P_O,P_Z,   0,0,0, R_ALL, 0, 0,P_O,P_Z },
  {__LINE__,B_ABS, 0,P_O,P_1Z,  0,0,0, R_ALL, 0, 0,P_O,P_1Z },
  {__LINE__,B_ABS, 1,P_O,P_1Z,  0,0,0, R_ALL, 0, 0,P_O,P_1Z },
  {__LINE__,B_ABS, 0,P_O,P_01Z, 0,0,0, R_ALL, 0, 0,P_O,P_01Z },
  {__LINE__,B_ABS, 1,P_O,P_01Z, 0,0,0, R_ALL, 0, 0,P_O,P_01Z },
  {__LINE__,B_ABS, 0,P_1Z,P_1Z1, 0,0,0, R_ALL, 0, 0,P_1Z,P_1Z1 },
  {__LINE__,B_ABS, 1,P_1Z,P_1Z1, 0,0,0, R_ALL, 0, 0,P_1Z,P_1Z1 },
  {__LINE__,B_ABS, 0,P_Z,P_Z1,  0,0,0, R_ALL, 0, 0,P_Z,P_Z1 },
  {__LINE__,B_ABS, 1,P_Z,P_Z1,  0,0,0, R_ALL, 0, 0,P_Z,P_Z1 },

  /* Square root.  */
  {__LINE__,B_SQRT, 0,P_Z,P_Z,   0,0,0, R_ALL, 0, 0,P_Z,P_Z },
  {__LINE__,B_SQRT, 1,P_Z,P_Z,   0,0,0, R_ALL, 0, 1,P_Z,P_Z },
  {__LINE__,B_SQRT, 0,P_O,P_1Z,  0,0,0, R_ALL, 0, 0,P_O,P_1Z },
  {__LINE__,B_SQRT, 1,P_O,P_1Z,  0,0,0, R_ALL, 0, 1,P_O,P_1Z },
  {__LINE__,B_SQRT, 0,P_O,P_01Z, 0,0,0, R_ALL,
   FE_INVALID | FE_INVALID_SNAN, 0,P_O,P_11Z },
  {__LINE__,B_SQRT, 1,P_O,P_01Z, 0,0,0, R_ALL,
   FE_INVALID | FE_INVALID_SNAN, 1,P_O,P_11Z },

  {__LINE__,B_SQRT, 0,P_O,P_Z,   0,0,0, R_ALL, 0, 0,P_O,P_Z },
  {__LINE__,B_SQRT, 0,P_0O,P_Z,  0,0,0, R_ALL, 0, 0,P_0O,P_Z },

  {__LINE__,B_SQRT, 1,P_O,P_Z,   0,0,0, R_ALL,
   FE_INVALID | FE_INVALID_SQRT, 0,P_O,P_1Z },
  {__LINE__,B_SQRT, 1,P_1Z,P_1Z1, 0,0,0, R_ALL,
   FE_INVALID | FE_INVALID_SQRT, 0,P_O,P_1Z },
  {__LINE__,B_SQRT, 1,P_Z,P_Z1,  0,0,0, R_ALL,
   FE_INVALID | FE_INVALID_SQRT, 0,P_O,P_1Z },

};

static void
check_op(void)
{
  size_t i;
  int j;
  tocheck_t r, a, b, x;
  int raised;

  for (i = 0; i < sizeof (optests) / sizeof (optests[0]); i++)
    {
      a = pattern(optests[i].a_sgn, optests[i].a_exp,
		  optests[i].a_mant);
      b = pattern(optests[i].b_sgn, optests[i].b_exp,
		  optests[i].b_mant);
      x = pattern(optests[i].x_sgn, optests[i].x_exp,
		  optests[i].x_mant);
      for (j = 0; j < 4; j++)
	if (optests[i].rmode & 1<<j)
	  {
	    fesetenv(rmodes+j);
	    switch (optests[i].op)
	      {
	      case B_ADD: r = a + b; break;
	      case B_SUB: r = a - b; break;
	      case B_MUL: r = a * b; break;
	      case B_DIV: r = a / b; break;
	      case B_NEG: r = -a; break;
	      case B_ABS: r = FUNC(fabs)(a); break;
	      case B_SQRT: r = FUNC(sqrt)(a); break;
	      }
	    raised = fetestexcept(all_exceptions);
	    check_result(optests[i].line,rmnames[j],x,r);
	    check_excepts(optests[i].line,rmnames[j],
			  optests[i].excepts,raised);
	  }
    }
}

static void
fail_xr(int line, const char *rm, tocheck_t x, tocheck_t r, tocheck_t xx,
	int xflag)
{
  size_t i;
  unsigned char *cx, *cr, *cxx;

  printf("%s:%d:round %s:fail\n with x=0x", __FILE__, line,rm);
  cx = (unsigned char *)&x;
  cr = (unsigned char *)&r;
  cxx = (unsigned char *)&xx;
  for (i = 0; i < sizeof (tocheck_t); i++)
    printf("%02x", cx[i]);
  printf(" r=0x");
  for (i = 0; i < sizeof (tocheck_t); i++)
    printf("%02x", cr[i]);
  printf(" xx=0x");
  for (i = 0; i < sizeof (tocheck_t); i++)
    printf("%02x", cxx[i]);
  printf(" inexact=%d\n", xflag != 0);
  nerrors++;
}

static void
check_sqrt(tocheck_t a)
{
  int j;
  tocheck_t r0, r1, r2, x0, x1, x2;
  int raised = 0;
  int ok;

  for (j = 0; j < 4; j++)
    {
      int excepts;

      fesetenv(rmodes+j);
      r1 = FUNC(sqrt)(a);
      excepts = fetestexcept(all_exceptions);
      fesetenv(FE_DFL_ENV);
      raised |= excepts & ~FE_INEXACT;
      x1 = r1 * r1 - a;
      if (excepts & FE_INEXACT)
	{
	  r0 = delta(r1,-1); r2 = delta(r1,1);
	  switch (1 << j)
	    {
	    case R_NEAREST:
	      x0 = r0 * r0 - a; x2 = r2 * r2 - a;
	      ok = fabs(x0) >= fabs(x1) && fabs(x1) <= fabs(x2);
	      break;
	    case R_ZERO:  case R_DOWN:
	      x2 = r2 * r2 - a;
	      ok = x1 <= 0 && x2 >= 0;
	      break;
	    case R_UP:
	      x0 = r0 * r0 - a;
	      ok = x1 >= 0 && x0 <= 0;
	      break;
	    default:
	      assert(0);
	    }
	}
      else
	ok = x1 == 0;
      if (!ok)
	fail_xr(__LINE__,rmnames[j],a,r1,x1,excepts&FE_INEXACT);
    }
  check_excepts(__LINE__,"all",0,raised);
}

int main(int argc, char **argv)
{
  int i;

  _LIB_VERSION = _IEEE_;

  /* Set up environments for rounding modes.  */
  fesetenv(FE_DFL_ENV);
  fesetround(FE_TONEAREST);
  fegetenv(rmodes+0);
  fesetround(FE_TOWARDZERO);
  fegetenv(rmodes+1);
  fesetround(FE_UPWARD);
  fegetenv(rmodes+2);
  fesetround(FE_DOWNWARD);
  fegetenv(rmodes+3);

#if defined(FE_INVALID_SOFTWARE) || defined(FE_INVALID_SQRT)
  /* There's this really stupid feature of the 601... */
  fesetenv(FE_DFL_ENV);
  feraiseexcept(FE_INVALID_SOFTWARE);
  if (!fetestexcept(FE_INVALID_SOFTWARE))
    excepts_missing |= FE_INVALID_SOFTWARE;
  fesetenv(FE_DFL_ENV);
  feraiseexcept(FE_INVALID_SQRT);
  if (!fetestexcept(FE_INVALID_SQRT))
    excepts_missing |= FE_INVALID_SQRT;
#endif

  check_op();
  for (i = 0; i < 100000; i++)
    check_sqrt(pattern(0, P_Rno, P_R));
  for (i = 0; i < 100; i++)
    check_sqrt(pattern(0, P_Z, P_R));
  check_sqrt(pattern(0,P_Z,P_Z1));

  printf("%d errors.\n", nerrors);
  return nerrors == 0 ? 0 : 1;
}
