/* Convert string representing a number to float value, using given locale.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <bits/floatn.h>

#ifdef FLOAT
# define BUILD_DOUBLE 0
#else
# define BUILD_DOUBLE 1
#endif

#if BUILD_DOUBLE
# if __HAVE_FLOAT64 && !__HAVE_DISTINCT_FLOAT64
#  define strtof64_l __hide_strtof64_l
#  define wcstof64_l __hide_wcstof64_l
# endif
# if __HAVE_FLOAT32X && !__HAVE_DISTINCT_FLOAT32X
#  define strtof32x_l __hide_strtof32x_l
#  define wcstof32x_l __hide_wcstof32x_l
# endif
#endif

#include <locale.h>

extern double ____strtod_l_internal (const char *, char **, int, locale_t);

/* Configuration part.  These macros are defined by `strtold.c',
   `strtof.c', `wcstod.c', `wcstold.c', and `wcstof.c' to produce the
   `long double' and `float' versions of the reader.  */
#ifndef FLOAT
# include <math_ldbl_opt.h>
# define FLOAT		double
# define FLT		DBL
# ifdef USE_WIDE_CHAR
#  define STRTOF	wcstod_l
#  define __STRTOF	__wcstod_l
#  define STRTOF_NAN	__wcstod_nan
# else
#  define STRTOF	strtod_l
#  define __STRTOF	__strtod_l
#  define STRTOF_NAN	__strtod_nan
# endif
# define MPN2FLOAT	__mpn_construct_double
# define FLOAT_HUGE_VAL	HUGE_VAL
#endif
/* End of configuration part.  */

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include "../locale/localeinfo.h"
#include <math.h>
#include <math-barriers.h>
#include <math-narrow-eval.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <rounding-mode.h>
#include <tininess.h>

/* The gmp headers need some configuration frobs.  */
#define HAVE_ALLOCA 1

/* Include gmp-mparam.h first, such that definitions of _SHORT_LIMB
   and _LONG_LONG_LIMB in it can take effect into gmp.h.  */
#include <gmp-mparam.h>
#include <gmp.h>
#include "gmp-impl.h"
#include "longlong.h"
#include "fpioconst.h"

#include <assert.h>


/* We use this code for the extended locale handling where the
   function gets as an additional argument the locale which has to be
   used.  To access the values we have to redefine the _NL_CURRENT and
   _NL_CURRENT_WORD macros.  */
#undef _NL_CURRENT
#define _NL_CURRENT(category, item) \
  (current->values[_NL_ITEM_INDEX (item)].string)
#undef _NL_CURRENT_WORD
#define _NL_CURRENT_WORD(category, item) \
  ((uint32_t) current->values[_NL_ITEM_INDEX (item)].word)

#if defined _LIBC || defined HAVE_WCHAR_H
# include <wchar.h>
#endif

#ifdef USE_WIDE_CHAR
# include <wctype.h>
# define STRING_TYPE wchar_t
# define CHAR_TYPE wint_t
# define L_(Ch) L##Ch
# define ISSPACE(Ch) __iswspace_l ((Ch), loc)
# define ISDIGIT(Ch) __iswdigit_l ((Ch), loc)
# define ISXDIGIT(Ch) __iswxdigit_l ((Ch), loc)
# define TOLOWER(Ch) __towlower_l ((Ch), loc)
# define TOLOWER_C(Ch) __towlower_l ((Ch), _nl_C_locobj_ptr)
# define STRNCASECMP(S1, S2, N) \
  __wcsncasecmp_l ((S1), (S2), (N), _nl_C_locobj_ptr)
#else
# define STRING_TYPE char
# define CHAR_TYPE char
# define L_(Ch) Ch
# define ISSPACE(Ch) __isspace_l ((Ch), loc)
# define ISDIGIT(Ch) __isdigit_l ((Ch), loc)
# define ISXDIGIT(Ch) __isxdigit_l ((Ch), loc)
# define TOLOWER(Ch) __tolower_l ((Ch), loc)
# define TOLOWER_C(Ch) __tolower_l ((Ch), _nl_C_locobj_ptr)
# define STRNCASECMP(S1, S2, N) \
  __strncasecmp_l ((S1), (S2), (N), _nl_C_locobj_ptr)
#endif


/* Constants we need from float.h; select the set for the FLOAT precision.  */
#define MANT_DIG	PASTE(FLT,_MANT_DIG)
#define	DIG		PASTE(FLT,_DIG)
#define	MAX_EXP		PASTE(FLT,_MAX_EXP)
#define	MIN_EXP		PASTE(FLT,_MIN_EXP)
#define MAX_10_EXP	PASTE(FLT,_MAX_10_EXP)
#define MIN_10_EXP	PASTE(FLT,_MIN_10_EXP)
#define MAX_VALUE	PASTE(FLT,_MAX)
#define MIN_VALUE	PASTE(FLT,_MIN)

/* Extra macros required to get FLT expanded before the pasting.  */
#define PASTE(a,b)	PASTE1(a,b)
#define PASTE1(a,b)	a##b

/* Function to construct a floating point number from an MP integer
   containing the fraction bits, a base 2 exponent, and a sign flag.  */
extern FLOAT MPN2FLOAT (mp_srcptr mpn, int exponent, int negative);

/* Definitions according to limb size used.  */
#if	BITS_PER_MP_LIMB == 32
# define MAX_DIG_PER_LIMB	9
# define MAX_FAC_PER_LIMB	1000000000UL
#elif	BITS_PER_MP_LIMB == 64
# define MAX_DIG_PER_LIMB	19
# define MAX_FAC_PER_LIMB	10000000000000000000ULL
#else
# error "mp_limb_t size " BITS_PER_MP_LIMB "not accounted for"
#endif

extern const mp_limb_t _tens_in_limb[MAX_DIG_PER_LIMB + 1];

#ifndef	howmany
#define	howmany(x,y)		(((x)+((y)-1))/(y))
#endif
#define SWAP(x, y)		({ typeof(x) _tmp = x; x = y; y = _tmp; })

#define	RETURN_LIMB_SIZE		howmany (MANT_DIG, BITS_PER_MP_LIMB)

#define RETURN(val,end)							      \
    do { if (endptr != NULL) *endptr = (STRING_TYPE *) (end);		      \
	 return val; } while (0)

/* Maximum size necessary for mpn integers to hold floating point
   numbers.  The largest number we need to hold is 10^n where 2^-n is
   1/4 ulp of the smallest representable value (that is, n = MANT_DIG
   - MIN_EXP + 2).  Approximate using 10^3 < 2^10.  */
#define	MPNSIZE		(howmany (1 + ((MANT_DIG - MIN_EXP + 2) * 10) / 3, \
				  BITS_PER_MP_LIMB) + 2)
/* Declare an mpn integer variable that big.  */
#define	MPN_VAR(name)	mp_limb_t name[MPNSIZE]; mp_size_t name##size
/* Copy an mpn integer value.  */
#define MPN_ASSIGN(dst, src) \
	memcpy (dst, src, (dst##size = src##size) * sizeof (mp_limb_t))


/* Set errno and return an overflowing value with sign specified by
   NEGATIVE.  */
static FLOAT
overflow_value (int negative)
{
  __set_errno (ERANGE);
  FLOAT result = math_narrow_eval ((negative ? -MAX_VALUE : MAX_VALUE)
				   * MAX_VALUE);
  return result;
}


/* Set errno and return an underflowing value with sign specified by
   NEGATIVE.  */
static FLOAT
underflow_value (int negative)
{
  __set_errno (ERANGE);
  FLOAT result = math_narrow_eval ((negative ? -MIN_VALUE : MIN_VALUE)
				   * MIN_VALUE);
  return result;
}


/* Return a floating point number of the needed type according to the given
   multi-precision number after possible rounding.  */
static FLOAT
round_and_return (mp_limb_t *retval, intmax_t exponent, int negative,
		  mp_limb_t round_limb, mp_size_t round_bit, int more_bits)
{
  int mode = get_rounding_mode ();

  if (exponent < MIN_EXP - 1)
    {
      if (exponent < MIN_EXP - 1 - MANT_DIG)
	return underflow_value (negative);

      mp_size_t shift = MIN_EXP - 1 - exponent;
      bool is_tiny = true;

      more_bits |= (round_limb & ((((mp_limb_t) 1) << round_bit) - 1)) != 0;
      if (shift == MANT_DIG)
	/* This is a special case to handle the very seldom case where
	   the mantissa will be empty after the shift.  */
	{
	  int i;

	  round_limb = retval[RETURN_LIMB_SIZE - 1];
	  round_bit = (MANT_DIG - 1) % BITS_PER_MP_LIMB;
	  for (i = 0; i < RETURN_LIMB_SIZE - 1; ++i)
	    more_bits |= retval[i] != 0;
	  MPN_ZERO (retval, RETURN_LIMB_SIZE);
	}
      else if (shift >= BITS_PER_MP_LIMB)
	{
	  int i;

	  round_limb = retval[(shift - 1) / BITS_PER_MP_LIMB];
	  round_bit = (shift - 1) % BITS_PER_MP_LIMB;
	  for (i = 0; i < (shift - 1) / BITS_PER_MP_LIMB; ++i)
	    more_bits |= retval[i] != 0;
	  more_bits |= ((round_limb & ((((mp_limb_t) 1) << round_bit) - 1))
			!= 0);

	  /* __mpn_rshift requires 0 < shift < BITS_PER_MP_LIMB.  */
	  if ((shift % BITS_PER_MP_LIMB) != 0)
	    (void) __mpn_rshift (retval, &retval[shift / BITS_PER_MP_LIMB],
			         RETURN_LIMB_SIZE - (shift / BITS_PER_MP_LIMB),
			         shift % BITS_PER_MP_LIMB);
	  else
	    for (i = 0; i < RETURN_LIMB_SIZE - (shift / BITS_PER_MP_LIMB); i++)
	      retval[i] = retval[i + (shift / BITS_PER_MP_LIMB)];
	  MPN_ZERO (&retval[RETURN_LIMB_SIZE - (shift / BITS_PER_MP_LIMB)],
		    shift / BITS_PER_MP_LIMB);
	}
      else if (shift > 0)
	{
	  if (TININESS_AFTER_ROUNDING && shift == 1)
	    {
	      /* Whether the result counts as tiny depends on whether,
		 after rounding to the normal precision, it still has
		 a subnormal exponent.  */
	      mp_limb_t retval_normal[RETURN_LIMB_SIZE];
	      if (round_away (negative,
			      (retval[0] & 1) != 0,
			      (round_limb
			       & (((mp_limb_t) 1) << round_bit)) != 0,
			      (more_bits
			       || ((round_limb
				    & ((((mp_limb_t) 1) << round_bit) - 1))
				   != 0)),
			      mode))
		{
		  mp_limb_t cy = __mpn_add_1 (retval_normal, retval,
					      RETURN_LIMB_SIZE, 1);

		  if (((MANT_DIG % BITS_PER_MP_LIMB) == 0 && cy)
		      || ((MANT_DIG % BITS_PER_MP_LIMB) != 0
			  && ((retval_normal[RETURN_LIMB_SIZE - 1]
			       & (((mp_limb_t) 1)
				  << (MANT_DIG % BITS_PER_MP_LIMB)))
			      != 0)))
		    is_tiny = false;
		}
	    }
	  round_limb = retval[0];
	  round_bit = shift - 1;
	  (void) __mpn_rshift (retval, retval, RETURN_LIMB_SIZE, shift);
	}
      /* This is a hook for the m68k long double format, where the
	 exponent bias is the same for normalized and denormalized
	 numbers.  */
#ifndef DENORM_EXP
# define DENORM_EXP (MIN_EXP - 2)
#endif
      exponent = DENORM_EXP;
      if (is_tiny
	  && ((round_limb & (((mp_limb_t) 1) << round_bit)) != 0
	      || more_bits
	      || (round_limb & ((((mp_limb_t) 1) << round_bit) - 1)) != 0))
	{
	  __set_errno (ERANGE);
	  FLOAT force_underflow = MIN_VALUE * MIN_VALUE;
	  math_force_eval (force_underflow);
	}
    }

  if (exponent >= MAX_EXP)
    goto overflow;

  bool half_bit = (round_limb & (((mp_limb_t) 1) << round_bit)) != 0;
  bool more_bits_nonzero
    = (more_bits
       || (round_limb & ((((mp_limb_t) 1) << round_bit) - 1)) != 0);
  if (round_away (negative,
		  (retval[0] & 1) != 0,
		  half_bit,
		  more_bits_nonzero,
		  mode))
    {
      mp_limb_t cy = __mpn_add_1 (retval, retval, RETURN_LIMB_SIZE, 1);

      if (((MANT_DIG % BITS_PER_MP_LIMB) == 0 && cy)
	  || ((MANT_DIG % BITS_PER_MP_LIMB) != 0
	      && (retval[RETURN_LIMB_SIZE - 1]
		  & (((mp_limb_t) 1) << (MANT_DIG % BITS_PER_MP_LIMB))) != 0))
	{
	  ++exponent;
	  (void) __mpn_rshift (retval, retval, RETURN_LIMB_SIZE, 1);
	  retval[RETURN_LIMB_SIZE - 1]
	    |= ((mp_limb_t) 1) << ((MANT_DIG - 1) % BITS_PER_MP_LIMB);
	}
      else if (exponent == DENORM_EXP
	       && (retval[RETURN_LIMB_SIZE - 1]
		   & (((mp_limb_t) 1) << ((MANT_DIG - 1) % BITS_PER_MP_LIMB)))
	       != 0)
	  /* The number was denormalized but now normalized.  */
	exponent = MIN_EXP - 1;
    }

  if (exponent >= MAX_EXP)
  overflow:
    return overflow_value (negative);

  if (half_bit || more_bits_nonzero)
    {
      FLOAT force_inexact = (FLOAT) 1 + MIN_VALUE;
      math_force_eval (force_inexact);
    }
  return MPN2FLOAT (retval, exponent, negative);
}


/* Read a multi-precision integer starting at STR with exactly DIGCNT digits
   into N.  Return the size of the number limbs in NSIZE at the first
   character od the string that is not part of the integer as the function
   value.  If the EXPONENT is small enough to be taken as an additional
   factor for the resulting number (see code) multiply by it.  */
static const STRING_TYPE *
str_to_mpn (const STRING_TYPE *str, int digcnt, mp_limb_t *n, mp_size_t *nsize,
	    intmax_t *exponent
#ifndef USE_WIDE_CHAR
	    , const char *decimal, size_t decimal_len, const char *thousands
#endif

	    )
{
  /* Number of digits for actual limb.  */
  int cnt = 0;
  mp_limb_t low = 0;
  mp_limb_t start;

  *nsize = 0;
  assert (digcnt > 0);
  do
    {
      if (cnt == MAX_DIG_PER_LIMB)
	{
	  if (*nsize == 0)
	    {
	      n[0] = low;
	      *nsize = 1;
	    }
	  else
	    {
	      mp_limb_t cy;
	      cy = __mpn_mul_1 (n, n, *nsize, MAX_FAC_PER_LIMB);
	      cy += __mpn_add_1 (n, n, *nsize, low);
	      if (cy != 0)
		{
		  assert (*nsize < MPNSIZE);
		  n[*nsize] = cy;
		  ++(*nsize);
		}
	    }
	  cnt = 0;
	  low = 0;
	}

      /* There might be thousands separators or radix characters in
	 the string.  But these all can be ignored because we know the
	 format of the number is correct and we have an exact number
	 of characters to read.  */
#ifdef USE_WIDE_CHAR
      if (*str < L'0' || *str > L'9')
	++str;
#else
      if (*str < '0' || *str > '9')
	{
	  int inner = 0;
	  if (thousands != NULL && *str == *thousands
	      && ({ for (inner = 1; thousands[inner] != '\0'; ++inner)
		      if (thousands[inner] != str[inner])
			break;
		    thousands[inner] == '\0'; }))
	    str += inner;
	  else
	    str += decimal_len;
	}
#endif
      low = low * 10 + *str++ - L_('0');
      ++cnt;
    }
  while (--digcnt > 0);

  if (*exponent > 0 && *exponent <= MAX_DIG_PER_LIMB - cnt)
    {
      low *= _tens_in_limb[*exponent];
      start = _tens_in_limb[cnt + *exponent];
      *exponent = 0;
    }
  else
    start = _tens_in_limb[cnt];

  if (*nsize == 0)
    {
      n[0] = low;
      *nsize = 1;
    }
  else
    {
      mp_limb_t cy;
      cy = __mpn_mul_1 (n, n, *nsize, start);
      cy += __mpn_add_1 (n, n, *nsize, low);
      if (cy != 0)
	{
	  assert (*nsize < MPNSIZE);
	  n[(*nsize)++] = cy;
	}
    }

  return str;
}


/* Shift {PTR, SIZE} COUNT bits to the left, and fill the vacated bits
   with the COUNT most significant bits of LIMB.

   Implemented as a macro, so that __builtin_constant_p works even at -O0.

   Tege doesn't like this macro so I have to write it here myself. :)
   --drepper */
#define __mpn_lshift_1(ptr, size, count, limb) \
  do									\
    {									\
      mp_limb_t *__ptr = (ptr);						\
      if (__builtin_constant_p (count) && count == BITS_PER_MP_LIMB)	\
	{								\
	  mp_size_t i;							\
	  for (i = (size) - 1; i > 0; --i)				\
	    __ptr[i] = __ptr[i - 1];					\
	  __ptr[0] = (limb);						\
	}								\
      else								\
	{								\
	  /* We assume count > 0 && count < BITS_PER_MP_LIMB here.  */	\
	  unsigned int __count = (count);				\
	  (void) __mpn_lshift (__ptr, __ptr, size, __count);		\
	  __ptr[0] |= (limb) >> (BITS_PER_MP_LIMB - __count);		\
	}								\
    }									\
  while (0)


#define INTERNAL(x) INTERNAL1(x)
#define INTERNAL1(x) __##x##_internal
#ifndef ____STRTOF_INTERNAL
# define ____STRTOF_INTERNAL INTERNAL (__STRTOF)
#endif

/* This file defines a function to check for correct grouping.  */
#include "grouping.h"


/* Return a floating point number with the value of the given string NPTR.
   Set *ENDPTR to the character after the last used one.  If the number is
   smaller than the smallest representable number, set `errno' to ERANGE and
   return 0.0.  If the number is too big to be represented, set `errno' to
   ERANGE and return HUGE_VAL with the appropriate sign.  */
FLOAT
____STRTOF_INTERNAL (const STRING_TYPE *nptr, STRING_TYPE **endptr, int group,
		     locale_t loc)
{
  int negative;			/* The sign of the number.  */
  MPN_VAR (num);		/* MP representation of the number.  */
  intmax_t exponent;		/* Exponent of the number.  */

  /* Numbers starting `0X' or `0x' have to be processed with base 16.  */
  int base = 10;

  /* When we have to compute fractional digits we form a fraction with a
     second multi-precision number (and we sometimes need a second for
     temporary results).  */
  MPN_VAR (den);

  /* Representation for the return value.  */
  mp_limb_t retval[RETURN_LIMB_SIZE];
  /* Number of bits currently in result value.  */
  int bits;

  /* Running pointer after the last character processed in the string.  */
  const STRING_TYPE *cp, *tp;
  /* Start of significant part of the number.  */
  const STRING_TYPE *startp, *start_of_digits;
  /* Points at the character following the integer and fractional digits.  */
  const STRING_TYPE *expp;
  /* Total number of digit and number of digits in integer part.  */
  size_t dig_no, int_no, lead_zero;
  /* Contains the last character read.  */
  CHAR_TYPE c;

/* We should get wint_t from <stddef.h>, but not all GCC versions define it
   there.  So define it ourselves if it remains undefined.  */
#ifndef _WINT_T
  typedef unsigned int wint_t;
#endif
  /* The radix character of the current locale.  */
#ifdef USE_WIDE_CHAR
  wchar_t decimal;
#else
  const char *decimal;
  size_t decimal_len;
#endif
  /* The thousands character of the current locale.  */
#ifdef USE_WIDE_CHAR
  wchar_t thousands = L'\0';
#else
  const char *thousands = NULL;
#endif
  /* The numeric grouping specification of the current locale,
     in the format described in <locale.h>.  */
  const char *grouping;
  /* Used in several places.  */
  int cnt;

  struct __locale_data *current = loc->__locales[LC_NUMERIC];

  if (__glibc_unlikely (group))
    {
      grouping = _NL_CURRENT (LC_NUMERIC, GROUPING);
      if (*grouping <= 0 || *grouping == CHAR_MAX)
	grouping = NULL;
      else
	{
	  /* Figure out the thousands separator character.  */
#ifdef USE_WIDE_CHAR
	  thousands = _NL_CURRENT_WORD (LC_NUMERIC,
					_NL_NUMERIC_THOUSANDS_SEP_WC);
	  if (thousands == L'\0')
	    grouping = NULL;
#else
	  thousands = _NL_CURRENT (LC_NUMERIC, THOUSANDS_SEP);
	  if (*thousands == '\0')
	    {
	      thousands = NULL;
	      grouping = NULL;
	    }
#endif
	}
    }
  else
    grouping = NULL;

  /* Find the locale's decimal point character.  */
#ifdef USE_WIDE_CHAR
  decimal = _NL_CURRENT_WORD (LC_NUMERIC, _NL_NUMERIC_DECIMAL_POINT_WC);
  assert (decimal != L'\0');
# define decimal_len 1
#else
  decimal = _NL_CURRENT (LC_NUMERIC, DECIMAL_POINT);
  decimal_len = strlen (decimal);
  assert (decimal_len > 0);
#endif

  /* Prepare number representation.  */
  exponent = 0;
  negative = 0;
  bits = 0;

  /* Parse string to get maximal legal prefix.  We need the number of
     characters of the integer part, the fractional part and the exponent.  */
  cp = nptr - 1;
  /* Ignore leading white space.  */
  do
    c = *++cp;
  while (ISSPACE (c));

  /* Get sign of the result.  */
  if (c == L_('-'))
    {
      negative = 1;
      c = *++cp;
    }
  else if (c == L_('+'))
    c = *++cp;

  /* Return 0.0 if no legal string is found.
     No character is used even if a sign was found.  */
#ifdef USE_WIDE_CHAR
  if (c == (wint_t) decimal
      && (wint_t) cp[1] >= L'0' && (wint_t) cp[1] <= L'9')
    {
      /* We accept it.  This funny construct is here only to indent
	 the code correctly.  */
    }
#else
  for (cnt = 0; decimal[cnt] != '\0'; ++cnt)
    if (cp[cnt] != decimal[cnt])
      break;
  if (decimal[cnt] == '\0' && cp[cnt] >= '0' && cp[cnt] <= '9')
    {
      /* We accept it.  This funny construct is here only to indent
	 the code correctly.  */
    }
#endif
  else if (c < L_('0') || c > L_('9'))
    {
      /* Check for `INF' or `INFINITY'.  */
      CHAR_TYPE lowc = TOLOWER_C (c);

      if (lowc == L_('i') && STRNCASECMP (cp, L_("inf"), 3) == 0)
	{
	  /* Return +/- infinity.  */
	  if (endptr != NULL)
	    *endptr = (STRING_TYPE *)
		      (cp + (STRNCASECMP (cp + 3, L_("inity"), 5) == 0
			     ? 8 : 3));

	  return negative ? -FLOAT_HUGE_VAL : FLOAT_HUGE_VAL;
	}

      if (lowc == L_('n') && STRNCASECMP (cp, L_("nan"), 3) == 0)
	{
	  /* Return NaN.  */
	  FLOAT retval = NAN;

	  cp += 3;

	  /* Match `(n-char-sequence-digit)'.  */
	  if (*cp == L_('('))
	    {
	      const STRING_TYPE *startp = cp;
	      STRING_TYPE *endp;
	      retval = STRTOF_NAN (cp + 1, &endp, L_(')'));
	      if (*endp == L_(')'))
		/* Consume the closing parenthesis.  */
		cp = endp + 1;
	      else
		/* Only match the NAN part.  */
		cp = startp;
	    }

	  if (endptr != NULL)
	    *endptr = (STRING_TYPE *) cp;

	  return negative ? -retval : retval;
	}

      /* It is really a text we do not recognize.  */
      RETURN (0.0, nptr);
    }

  /* First look whether we are faced with a hexadecimal number.  */
  if (c == L_('0') && TOLOWER (cp[1]) == L_('x'))
    {
      /* Okay, it is a hexa-decimal number.  Remember this and skip
	 the characters.  BTW: hexadecimal numbers must not be
	 grouped.  */
      base = 16;
      cp += 2;
      c = *cp;
      grouping = NULL;
    }

  /* Record the start of the digits, in case we will check their grouping.  */
  start_of_digits = startp = cp;

  /* Ignore leading zeroes.  This helps us to avoid useless computations.  */
#ifdef USE_WIDE_CHAR
  while (c == L'0' || ((wint_t) thousands != L'\0' && c == (wint_t) thousands))
    c = *++cp;
#else
  if (__glibc_likely (thousands == NULL))
    while (c == '0')
      c = *++cp;
  else
    {
      /* We also have the multibyte thousands string.  */
      while (1)
	{
	  if (c != '0')
	    {
	      for (cnt = 0; thousands[cnt] != '\0'; ++cnt)
		if (thousands[cnt] != cp[cnt])
		  break;
	      if (thousands[cnt] != '\0')
		break;
	      cp += cnt - 1;
	    }
	  c = *++cp;
	}
    }
#endif

  /* If no other digit but a '0' is found the result is 0.0.
     Return current read pointer.  */
  CHAR_TYPE lowc = TOLOWER (c);
  if (!((c >= L_('0') && c <= L_('9'))
	|| (base == 16 && lowc >= L_('a') && lowc <= L_('f'))
	|| (
#ifdef USE_WIDE_CHAR
	    c == (wint_t) decimal
#else
	    ({ for (cnt = 0; decimal[cnt] != '\0'; ++cnt)
		 if (decimal[cnt] != cp[cnt])
		   break;
	       decimal[cnt] == '\0'; })
#endif
	    /* '0x.' alone is not a valid hexadecimal number.
	       '.' alone is not valid either, but that has been checked
	       already earlier.  */
	    && (base != 16
		|| cp != start_of_digits
		|| (cp[decimal_len] >= L_('0') && cp[decimal_len] <= L_('9'))
		|| ({ CHAR_TYPE lo = TOLOWER (cp[decimal_len]);
		      lo >= L_('a') && lo <= L_('f'); })))
	|| (base == 16 && (cp != start_of_digits
			   && lowc == L_('p')))
	|| (base != 16 && lowc == L_('e'))))
    {
#ifdef USE_WIDE_CHAR
      tp = __correctly_grouped_prefixwc (start_of_digits, cp, thousands,
					 grouping);
#else
      tp = __correctly_grouped_prefixmb (start_of_digits, cp, thousands,
					 grouping);
#endif
      /* If TP is at the start of the digits, there was no correctly
	 grouped prefix of the string; so no number found.  */
      RETURN (negative ? -0.0 : 0.0,
	      tp == start_of_digits ? (base == 16 ? cp - 1 : nptr) : tp);
    }

  /* Remember first significant digit and read following characters until the
     decimal point, exponent character or any non-FP number character.  */
  startp = cp;
  dig_no = 0;
  while (1)
    {
      if ((c >= L_('0') && c <= L_('9'))
	  || (base == 16
	      && ({ CHAR_TYPE lo = TOLOWER (c);
		    lo >= L_('a') && lo <= L_('f'); })))
	++dig_no;
      else
	{
#ifdef USE_WIDE_CHAR
	  if (__builtin_expect ((wint_t) thousands == L'\0', 1)
	      || c != (wint_t) thousands)
	    /* Not a digit or separator: end of the integer part.  */
	    break;
#else
	  if (__glibc_likely (thousands == NULL))
	    break;
	  else
	    {
	      for (cnt = 0; thousands[cnt] != '\0'; ++cnt)
		if (thousands[cnt] != cp[cnt])
		  break;
	      if (thousands[cnt] != '\0')
		break;
	      cp += cnt - 1;
	    }
#endif
	}
      c = *++cp;
    }

  if (__builtin_expect (grouping != NULL, 0) && cp > start_of_digits)
    {
      /* Check the grouping of the digits.  */
#ifdef USE_WIDE_CHAR
      tp = __correctly_grouped_prefixwc (start_of_digits, cp, thousands,
					 grouping);
#else
      tp = __correctly_grouped_prefixmb (start_of_digits, cp, thousands,
					 grouping);
#endif
      if (cp != tp)
	{
	  /* Less than the entire string was correctly grouped.  */

	  if (tp == start_of_digits)
	    /* No valid group of numbers at all: no valid number.  */
	    RETURN (0.0, nptr);

	  if (tp < startp)
	    /* The number is validly grouped, but consists
	       only of zeroes.  The whole value is zero.  */
	    RETURN (negative ? -0.0 : 0.0, tp);

	  /* Recompute DIG_NO so we won't read more digits than
	     are properly grouped.  */
	  cp = tp;
	  dig_no = 0;
	  for (tp = startp; tp < cp; ++tp)
	    if (*tp >= L_('0') && *tp <= L_('9'))
	      ++dig_no;

	  int_no = dig_no;
	  lead_zero = 0;

	  goto number_parsed;
	}
    }

  /* We have the number of digits in the integer part.  Whether these
     are all or any is really a fractional digit will be decided
     later.  */
  int_no = dig_no;
  lead_zero = int_no == 0 ? (size_t) -1 : 0;

  /* Read the fractional digits.  A special case are the 'american
     style' numbers like `16.' i.e. with decimal point but without
     trailing digits.  */
  if (
#ifdef USE_WIDE_CHAR
      c == (wint_t) decimal
#else
      ({ for (cnt = 0; decimal[cnt] != '\0'; ++cnt)
	   if (decimal[cnt] != cp[cnt])
	     break;
	 decimal[cnt] == '\0'; })
#endif
      )
    {
      cp += decimal_len;
      c = *cp;
      while ((c >= L_('0') && c <= L_('9'))
	     || (base == 16 && ({ CHAR_TYPE lo = TOLOWER (c);
				  lo >= L_('a') && lo <= L_('f'); })))
	{
	  if (c != L_('0') && lead_zero == (size_t) -1)
	    lead_zero = dig_no - int_no;
	  ++dig_no;
	  c = *++cp;
	}
    }
  assert (dig_no <= (uintmax_t) INTMAX_MAX);

  /* Remember start of exponent (if any).  */
  expp = cp;

  /* Read exponent.  */
  lowc = TOLOWER (c);
  if ((base == 16 && lowc == L_('p'))
      || (base != 16 && lowc == L_('e')))
    {
      int exp_negative = 0;

      c = *++cp;
      if (c == L_('-'))
	{
	  exp_negative = 1;
	  c = *++cp;
	}
      else if (c == L_('+'))
	c = *++cp;

      if (c >= L_('0') && c <= L_('9'))
	{
	  intmax_t exp_limit;

	  /* Get the exponent limit. */
	  if (base == 16)
	    {
	      if (exp_negative)
		{
		  assert (int_no <= (uintmax_t) (INTMAX_MAX
						 + MIN_EXP - MANT_DIG) / 4);
		  exp_limit = -MIN_EXP + MANT_DIG + 4 * (intmax_t) int_no;
		}
	      else
		{
		  if (int_no)
		    {
		      assert (lead_zero == 0
			      && int_no <= (uintmax_t) INTMAX_MAX / 4);
		      exp_limit = MAX_EXP - 4 * (intmax_t) int_no + 3;
		    }
		  else if (lead_zero == (size_t) -1)
		    {
		      /* The number is zero and this limit is
			 arbitrary.  */
		      exp_limit = MAX_EXP + 3;
		    }
		  else
		    {
		      assert (lead_zero
			      <= (uintmax_t) (INTMAX_MAX - MAX_EXP - 3) / 4);
		      exp_limit = (MAX_EXP
				   + 4 * (intmax_t) lead_zero
				   + 3);
		    }
		}
	    }
	  else
	    {
	      if (exp_negative)
		{
		  assert (int_no
			  <= (uintmax_t) (INTMAX_MAX + MIN_10_EXP - MANT_DIG));
		  exp_limit = -MIN_10_EXP + MANT_DIG + (intmax_t) int_no;
		}
	      else
		{
		  if (int_no)
		    {
		      assert (lead_zero == 0
			      && int_no <= (uintmax_t) INTMAX_MAX);
		      exp_limit = MAX_10_EXP - (intmax_t) int_no + 1;
		    }
		  else if (lead_zero == (size_t) -1)
		    {
		      /* The number is zero and this limit is
			 arbitrary.  */
		      exp_limit = MAX_10_EXP + 1;
		    }
		  else
		    {
		      assert (lead_zero
			      <= (uintmax_t) (INTMAX_MAX - MAX_10_EXP - 1));
		      exp_limit = MAX_10_EXP + (intmax_t) lead_zero + 1;
		    }
		}
	    }

	  if (exp_limit < 0)
	    exp_limit = 0;

	  do
	    {
	      if (__builtin_expect ((exponent > exp_limit / 10
				     || (exponent == exp_limit / 10
					 && c - L_('0') > exp_limit % 10)), 0))
		/* The exponent is too large/small to represent a valid
		   number.  */
		{
		  FLOAT result;

		  /* We have to take care for special situation: a joker
		     might have written "0.0e100000" which is in fact
		     zero.  */
		  if (lead_zero == (size_t) -1)
		    result = negative ? -0.0 : 0.0;
		  else
		    {
		      /* Overflow or underflow.  */
		      result = (exp_negative
				? underflow_value (negative)
				: overflow_value (negative));
		    }

		  /* Accept all following digits as part of the exponent.  */
		  do
		    ++cp;
		  while (*cp >= L_('0') && *cp <= L_('9'));

		  RETURN (result, cp);
		  /* NOTREACHED */
		}

	      exponent *= 10;
	      exponent += c - L_('0');

	      c = *++cp;
	    }
	  while (c >= L_('0') && c <= L_('9'));

	  if (exp_negative)
	    exponent = -exponent;
	}
      else
	cp = expp;
    }

  /* We don't want to have to work with trailing zeroes after the radix.  */
  if (dig_no > int_no)
    {
      while (expp[-1] == L_('0'))
	{
	  --expp;
	  --dig_no;
	}
      assert (dig_no >= int_no);
    }

  if (dig_no == int_no && dig_no > 0 && exponent < 0)
    do
      {
	while (! (base == 16 ? ISXDIGIT (expp[-1]) : ISDIGIT (expp[-1])))
	  --expp;

	if (expp[-1] != L_('0'))
	  break;

	--expp;
	--dig_no;
	--int_no;
	exponent += base == 16 ? 4 : 1;
      }
    while (dig_no > 0 && exponent < 0);

 number_parsed:

  /* The whole string is parsed.  Store the address of the next character.  */
  if (endptr)
    *endptr = (STRING_TYPE *) cp;

  if (dig_no == 0)
    return negative ? -0.0 : 0.0;

  if (lead_zero)
    {
      /* Find the decimal point */
#ifdef USE_WIDE_CHAR
      while (*startp != decimal)
	++startp;
#else
      while (1)
	{
	  if (*startp == decimal[0])
	    {
	      for (cnt = 1; decimal[cnt] != '\0'; ++cnt)
		if (decimal[cnt] != startp[cnt])
		  break;
	      if (decimal[cnt] == '\0')
		break;
	    }
	  ++startp;
	}
#endif
      startp += lead_zero + decimal_len;
      assert (lead_zero <= (base == 16
			    ? (uintmax_t) INTMAX_MAX / 4
			    : (uintmax_t) INTMAX_MAX));
      assert (lead_zero <= (base == 16
			    ? ((uintmax_t) exponent
			       - (uintmax_t) INTMAX_MIN) / 4
			    : ((uintmax_t) exponent - (uintmax_t) INTMAX_MIN)));
      exponent -= base == 16 ? 4 * (intmax_t) lead_zero : (intmax_t) lead_zero;
      dig_no -= lead_zero;
    }

  /* If the BASE is 16 we can use a simpler algorithm.  */
  if (base == 16)
    {
      static const int nbits[16] = { 0, 1, 2, 2, 3, 3, 3, 3,
				     4, 4, 4, 4, 4, 4, 4, 4 };
      int idx = (MANT_DIG - 1) / BITS_PER_MP_LIMB;
      int pos = (MANT_DIG - 1) % BITS_PER_MP_LIMB;
      mp_limb_t val;

      while (!ISXDIGIT (*startp))
	++startp;
      while (*startp == L_('0'))
	++startp;
      if (ISDIGIT (*startp))
	val = *startp++ - L_('0');
      else
	val = 10 + TOLOWER (*startp++) - L_('a');
      bits = nbits[val];
      /* We cannot have a leading zero.  */
      assert (bits != 0);

      if (pos + 1 >= 4 || pos + 1 >= bits)
	{
	  /* We don't have to care for wrapping.  This is the normal
	     case so we add the first clause in the `if' expression as
	     an optimization.  It is a compile-time constant and so does
	     not cost anything.  */
	  retval[idx] = val << (pos - bits + 1);
	  pos -= bits;
	}
      else
	{
	  retval[idx--] = val >> (bits - pos - 1);
	  retval[idx] = val << (BITS_PER_MP_LIMB - (bits - pos - 1));
	  pos = BITS_PER_MP_LIMB - 1 - (bits - pos - 1);
	}

      /* Adjust the exponent for the bits we are shifting in.  */
      assert (int_no <= (uintmax_t) (exponent < 0
				     ? (INTMAX_MAX - bits + 1) / 4
				     : (INTMAX_MAX - exponent - bits + 1) / 4));
      exponent += bits - 1 + ((intmax_t) int_no - 1) * 4;

      while (--dig_no > 0 && idx >= 0)
	{
	  if (!ISXDIGIT (*startp))
	    startp += decimal_len;
	  if (ISDIGIT (*startp))
	    val = *startp++ - L_('0');
	  else
	    val = 10 + TOLOWER (*startp++) - L_('a');

	  if (pos + 1 >= 4)
	    {
	      retval[idx] |= val << (pos - 4 + 1);
	      pos -= 4;
	    }
	  else
	    {
	      retval[idx--] |= val >> (4 - pos - 1);
	      val <<= BITS_PER_MP_LIMB - (4 - pos - 1);
	      if (idx < 0)
		{
		  int rest_nonzero = 0;
		  while (--dig_no > 0)
		    {
		      if (*startp != L_('0'))
			{
			  rest_nonzero = 1;
			  break;
			}
		      startp++;
		    }
		  return round_and_return (retval, exponent, negative, val,
					   BITS_PER_MP_LIMB - 1, rest_nonzero);
		}

	      retval[idx] = val;
	      pos = BITS_PER_MP_LIMB - 1 - (4 - pos - 1);
	    }
	}

      /* We ran out of digits.  */
      MPN_ZERO (retval, idx);

      return round_and_return (retval, exponent, negative, 0, 0, 0);
    }

  /* Now we have the number of digits in total and the integer digits as well
     as the exponent and its sign.  We can decide whether the read digits are
     really integer digits or belong to the fractional part; i.e. we normalize
     123e-2 to 1.23.  */
  {
    intmax_t incr = (exponent < 0
		     ? MAX (-(intmax_t) int_no, exponent)
		     : MIN ((intmax_t) dig_no - (intmax_t) int_no, exponent));
    int_no += incr;
    exponent -= incr;
  }

  if (__glibc_unlikely (exponent > MAX_10_EXP + 1 - (intmax_t) int_no))
    return overflow_value (negative);

  /* 10^(MIN_10_EXP-1) is not normal.  Thus, 10^(MIN_10_EXP-1) /
     2^MANT_DIG is below half the least subnormal, so anything with a
     base-10 exponent less than the base-10 exponent (which is
     MIN_10_EXP - 1 - ceil(MANT_DIG*log10(2))) of that value
     underflows.  DIG is floor((MANT_DIG-1)log10(2)), so an exponent
     below MIN_10_EXP - (DIG + 3) underflows.  But EXPONENT is
     actually an exponent multiplied only by a fractional part, not an
     integer part, so an exponent below MIN_10_EXP - (DIG + 2)
     underflows.  */
  if (__glibc_unlikely (exponent < MIN_10_EXP - (DIG + 2)))
    return underflow_value (negative);

  if (int_no > 0)
    {
      /* Read the integer part as a multi-precision number to NUM.  */
      startp = str_to_mpn (startp, int_no, num, &numsize, &exponent
#ifndef USE_WIDE_CHAR
			   , decimal, decimal_len, thousands
#endif
			   );

      if (exponent > 0)
	{
	  /* We now multiply the gained number by the given power of ten.  */
	  mp_limb_t *psrc = num;
	  mp_limb_t *pdest = den;
	  int expbit = 1;
	  const struct mp_power *ttab = &_fpioconst_pow10[0];

	  do
	    {
	      if ((exponent & expbit) != 0)
		{
		  size_t size = ttab->arraysize - _FPIO_CONST_OFFSET;
		  mp_limb_t cy;
		  exponent ^= expbit;

		  /* FIXME: not the whole multiplication has to be
		     done.  If we have the needed number of bits we
		     only need the information whether more non-zero
		     bits follow.  */
		  if (numsize >= ttab->arraysize - _FPIO_CONST_OFFSET)
		    cy = __mpn_mul (pdest, psrc, numsize,
				    &__tens[ttab->arrayoff
					   + _FPIO_CONST_OFFSET],
				    size);
		  else
		    cy = __mpn_mul (pdest, &__tens[ttab->arrayoff
						  + _FPIO_CONST_OFFSET],
				    size, psrc, numsize);
		  numsize += size;
		  if (cy == 0)
		    --numsize;
		  (void) SWAP (psrc, pdest);
		}
	      expbit <<= 1;
	      ++ttab;
	    }
	  while (exponent != 0);

	  if (psrc == den)
	    memcpy (num, den, numsize * sizeof (mp_limb_t));
	}

      /* Determine how many bits of the result we already have.  */
      count_leading_zeros (bits, num[numsize - 1]);
      bits = numsize * BITS_PER_MP_LIMB - bits;

      /* Now we know the exponent of the number in base two.
	 Check it against the maximum possible exponent.  */
      if (__glibc_unlikely (bits > MAX_EXP))
	return overflow_value (negative);

      /* We have already the first BITS bits of the result.  Together with
	 the information whether more non-zero bits follow this is enough
	 to determine the result.  */
      if (bits > MANT_DIG)
	{
	  int i;
	  const mp_size_t least_idx = (bits - MANT_DIG) / BITS_PER_MP_LIMB;
	  const mp_size_t least_bit = (bits - MANT_DIG) % BITS_PER_MP_LIMB;
	  const mp_size_t round_idx = least_bit == 0 ? least_idx - 1
						     : least_idx;
	  const mp_size_t round_bit = least_bit == 0 ? BITS_PER_MP_LIMB - 1
						     : least_bit - 1;

	  if (least_bit == 0)
	    memcpy (retval, &num[least_idx],
		    RETURN_LIMB_SIZE * sizeof (mp_limb_t));
	  else
	    {
	      for (i = least_idx; i < numsize - 1; ++i)
		retval[i - least_idx] = (num[i] >> least_bit)
					| (num[i + 1]
					   << (BITS_PER_MP_LIMB - least_bit));
	      if (i - least_idx < RETURN_LIMB_SIZE)
		retval[RETURN_LIMB_SIZE - 1] = num[i] >> least_bit;
	    }

	  /* Check whether any limb beside the ones in RETVAL are non-zero.  */
	  for (i = 0; num[i] == 0; ++i)
	    ;

	  return round_and_return (retval, bits - 1, negative,
				   num[round_idx], round_bit,
				   int_no < dig_no || i < round_idx);
	  /* NOTREACHED */
	}
      else if (dig_no == int_no)
	{
	  const mp_size_t target_bit = (MANT_DIG - 1) % BITS_PER_MP_LIMB;
	  const mp_size_t is_bit = (bits - 1) % BITS_PER_MP_LIMB;

	  if (target_bit == is_bit)
	    {
	      memcpy (&retval[RETURN_LIMB_SIZE - numsize], num,
		      numsize * sizeof (mp_limb_t));
	      /* FIXME: the following loop can be avoided if we assume a
		 maximal MANT_DIG value.  */
	      MPN_ZERO (retval, RETURN_LIMB_SIZE - numsize);
	    }
	  else if (target_bit > is_bit)
	    {
	      (void) __mpn_lshift (&retval[RETURN_LIMB_SIZE - numsize],
				   num, numsize, target_bit - is_bit);
	      /* FIXME: the following loop can be avoided if we assume a
		 maximal MANT_DIG value.  */
	      MPN_ZERO (retval, RETURN_LIMB_SIZE - numsize);
	    }
	  else
	    {
	      mp_limb_t cy;
	      assert (numsize < RETURN_LIMB_SIZE);

	      cy = __mpn_rshift (&retval[RETURN_LIMB_SIZE - numsize],
				 num, numsize, is_bit - target_bit);
	      retval[RETURN_LIMB_SIZE - numsize - 1] = cy;
	      /* FIXME: the following loop can be avoided if we assume a
		 maximal MANT_DIG value.  */
	      MPN_ZERO (retval, RETURN_LIMB_SIZE - numsize - 1);
	    }

	  return round_and_return (retval, bits - 1, negative, 0, 0, 0);
	  /* NOTREACHED */
	}

      /* Store the bits we already have.  */
      memcpy (retval, num, numsize * sizeof (mp_limb_t));
#if RETURN_LIMB_SIZE > 1
      if (numsize < RETURN_LIMB_SIZE)
# if RETURN_LIMB_SIZE == 2
	retval[numsize] = 0;
# else
	MPN_ZERO (retval + numsize, RETURN_LIMB_SIZE - numsize);
# endif
#endif
    }

  /* We have to compute at least some of the fractional digits.  */
  {
    /* We construct a fraction and the result of the division gives us
       the needed digits.  The denominator is 1.0 multiplied by the
       exponent of the lowest digit; i.e. 0.123 gives 123 / 1000 and
       123e-6 gives 123 / 1000000.  */

    int expbit;
    int neg_exp;
    int more_bits;
    int need_frac_digits;
    mp_limb_t cy;
    mp_limb_t *psrc = den;
    mp_limb_t *pdest = num;
    const struct mp_power *ttab = &_fpioconst_pow10[0];

    assert (dig_no > int_no
	    && exponent <= 0
	    && exponent >= MIN_10_EXP - (DIG + 2));

    /* We need to compute MANT_DIG - BITS fractional bits that lie
       within the mantissa of the result, the following bit for
       rounding, and to know whether any subsequent bit is 0.
       Computing a bit with value 2^-n means looking at n digits after
       the decimal point.  */
    if (bits > 0)
      {
	/* The bits required are those immediately after the point.  */
	assert (int_no > 0 && exponent == 0);
	need_frac_digits = 1 + MANT_DIG - bits;
      }
    else
      {
	/* The number is in the form .123eEXPONENT.  */
	assert (int_no == 0 && *startp != L_('0'));
	/* The number is at least 10^(EXPONENT-1), and 10^3 <
	   2^10.  */
	int neg_exp_2 = ((1 - exponent) * 10) / 3 + 1;
	/* The number is at least 2^-NEG_EXP_2.  We need up to
	   MANT_DIG bits following that bit.  */
	need_frac_digits = neg_exp_2 + MANT_DIG;
	/* However, we never need bits beyond 1/4 ulp of the smallest
	   representable value.  (That 1/4 ulp bit is only needed to
	   determine tinyness on machines where tinyness is determined
	   after rounding.)  */
	if (need_frac_digits > MANT_DIG - MIN_EXP + 2)
	  need_frac_digits = MANT_DIG - MIN_EXP + 2;
	/* At this point, NEED_FRAC_DIGITS is the total number of
	   digits needed after the point, but some of those may be
	   leading 0s.  */
	need_frac_digits += exponent;
	/* Any cases underflowing enough that none of the fractional
	   digits are needed should have been caught earlier (such
	   cases are on the order of 10^-n or smaller where 2^-n is
	   the least subnormal).  */
	assert (need_frac_digits > 0);
      }

    if (need_frac_digits > (intmax_t) dig_no - (intmax_t) int_no)
      need_frac_digits = (intmax_t) dig_no - (intmax_t) int_no;

    if ((intmax_t) dig_no > (intmax_t) int_no + need_frac_digits)
      {
	dig_no = int_no + need_frac_digits;
	more_bits = 1;
      }
    else
      more_bits = 0;

    neg_exp = (intmax_t) dig_no - (intmax_t) int_no - exponent;

    /* Construct the denominator.  */
    densize = 0;
    expbit = 1;
    do
      {
	if ((neg_exp & expbit) != 0)
	  {
	    mp_limb_t cy;
	    neg_exp ^= expbit;

	    if (densize == 0)
	      {
		densize = ttab->arraysize - _FPIO_CONST_OFFSET;
		memcpy (psrc, &__tens[ttab->arrayoff + _FPIO_CONST_OFFSET],
			densize * sizeof (mp_limb_t));
	      }
	    else
	      {
		cy = __mpn_mul (pdest, &__tens[ttab->arrayoff
					      + _FPIO_CONST_OFFSET],
				ttab->arraysize - _FPIO_CONST_OFFSET,
				psrc, densize);
		densize += ttab->arraysize - _FPIO_CONST_OFFSET;
		if (cy == 0)
		  --densize;
		(void) SWAP (psrc, pdest);
	      }
	  }
	expbit <<= 1;
	++ttab;
      }
    while (neg_exp != 0);

    if (psrc == num)
      memcpy (den, num, densize * sizeof (mp_limb_t));

    /* Read the fractional digits from the string.  */
    (void) str_to_mpn (startp, dig_no - int_no, num, &numsize, &exponent
#ifndef USE_WIDE_CHAR
		       , decimal, decimal_len, thousands
#endif
		       );

    /* We now have to shift both numbers so that the highest bit in the
       denominator is set.  In the same process we copy the numerator to
       a high place in the array so that the division constructs the wanted
       digits.  This is done by a "quasi fix point" number representation.

       num:   ddddddddddd . 0000000000000000000000
	      |--- m ---|
       den:                            ddddddddddd      n >= m
				       |--- n ---|
     */

    count_leading_zeros (cnt, den[densize - 1]);

    if (cnt > 0)
      {
	/* Don't call `mpn_shift' with a count of zero since the specification
	   does not allow this.  */
	(void) __mpn_lshift (den, den, densize, cnt);
	cy = __mpn_lshift (num, num, numsize, cnt);
	if (cy != 0)
	  num[numsize++] = cy;
      }

    /* Now we are ready for the division.  But it is not necessary to
       do a full multi-precision division because we only need a small
       number of bits for the result.  So we do not use __mpn_divmod
       here but instead do the division here by hand and stop whenever
       the needed number of bits is reached.  The code itself comes
       from the GNU MP Library by Torbj\"orn Granlund.  */

    exponent = bits;

    switch (densize)
      {
      case 1:
	{
	  mp_limb_t d, n, quot;
	  int used = 0;

	  n = num[0];
	  d = den[0];
	  assert (numsize == 1 && n < d);

	  do
	    {
	      udiv_qrnnd (quot, n, n, 0, d);

#define got_limb							      \
	      if (bits == 0)						      \
		{							      \
		  int cnt;						      \
		  if (quot == 0)					      \
		    cnt = BITS_PER_MP_LIMB;				      \
		  else							      \
		    count_leading_zeros (cnt, quot);			      \
		  exponent -= cnt;					      \
		  if (BITS_PER_MP_LIMB - cnt > MANT_DIG)		      \
		    {							      \
		      used = MANT_DIG + cnt;				      \
		      retval[0] = quot >> (BITS_PER_MP_LIMB - used);	      \
		      bits = MANT_DIG + 1;				      \
		    }							      \
		  else							      \
		    {							      \
		      /* Note that we only clear the second element.  */      \
		      /* The conditional is determined at compile time.  */   \
		      if (RETURN_LIMB_SIZE > 1)				      \
			retval[1] = 0;					      \
		      retval[0] = quot;					      \
		      bits = -cnt;					      \
		    }							      \
		}							      \
	      else if (bits + BITS_PER_MP_LIMB <= MANT_DIG)		      \
		__mpn_lshift_1 (retval, RETURN_LIMB_SIZE, BITS_PER_MP_LIMB,   \
				quot);					      \
	      else							      \
		{							      \
		  used = MANT_DIG - bits;				      \
		  if (used > 0)						      \
		    __mpn_lshift_1 (retval, RETURN_LIMB_SIZE, used, quot);    \
		}							      \
	      bits += BITS_PER_MP_LIMB

	      got_limb;
	    }
	  while (bits <= MANT_DIG);

	  return round_and_return (retval, exponent - 1, negative,
				   quot, BITS_PER_MP_LIMB - 1 - used,
				   more_bits || n != 0);
	}
      case 2:
	{
	  mp_limb_t d0, d1, n0, n1;
	  mp_limb_t quot = 0;
	  int used = 0;

	  d0 = den[0];
	  d1 = den[1];

	  if (numsize < densize)
	    {
	      if (num[0] >= d1)
		{
		  /* The numerator of the number occupies fewer bits than
		     the denominator but the one limb is bigger than the
		     high limb of the numerator.  */
		  n1 = 0;
		  n0 = num[0];
		}
	      else
		{
		  if (bits <= 0)
		    exponent -= BITS_PER_MP_LIMB;
		  else
		    {
		      if (bits + BITS_PER_MP_LIMB <= MANT_DIG)
			__mpn_lshift_1 (retval, RETURN_LIMB_SIZE,
					BITS_PER_MP_LIMB, 0);
		      else
			{
			  used = MANT_DIG - bits;
			  if (used > 0)
			    __mpn_lshift_1 (retval, RETURN_LIMB_SIZE, used, 0);
			}
		      bits += BITS_PER_MP_LIMB;
		    }
		  n1 = num[0];
		  n0 = 0;
		}
	    }
	  else
	    {
	      n1 = num[1];
	      n0 = num[0];
	    }

	  while (bits <= MANT_DIG)
	    {
	      mp_limb_t r;

	      if (n1 == d1)
		{
		  /* QUOT should be either 111..111 or 111..110.  We need
		     special treatment of this rare case as normal division
		     would give overflow.  */
		  quot = ~(mp_limb_t) 0;

		  r = n0 + d1;
		  if (r < d1)	/* Carry in the addition?  */
		    {
		      add_ssaaaa (n1, n0, r - d0, 0, 0, d0);
		      goto have_quot;
		    }
		  n1 = d0 - (d0 != 0);
		  n0 = -d0;
		}
	      else
		{
		  udiv_qrnnd (quot, r, n1, n0, d1);
		  umul_ppmm (n1, n0, d0, quot);
		}

	    q_test:
	      if (n1 > r || (n1 == r && n0 > 0))
		{
		  /* The estimated QUOT was too large.  */
		  --quot;

		  sub_ddmmss (n1, n0, n1, n0, 0, d0);
		  r += d1;
		  if (r >= d1)	/* If not carry, test QUOT again.  */
		    goto q_test;
		}
	      sub_ddmmss (n1, n0, r, 0, n1, n0);

	    have_quot:
	      got_limb;
	    }

	  return round_and_return (retval, exponent - 1, negative,
				   quot, BITS_PER_MP_LIMB - 1 - used,
				   more_bits || n1 != 0 || n0 != 0);
	}
      default:
	{
	  int i;
	  mp_limb_t cy, dX, d1, n0, n1;
	  mp_limb_t quot = 0;
	  int used = 0;

	  dX = den[densize - 1];
	  d1 = den[densize - 2];

	  /* The division does not work if the upper limb of the two-limb
	     numerator is greater than or equal to the denominator.  */
	  if (__mpn_cmp (num, &den[densize - numsize], numsize) >= 0)
	    num[numsize++] = 0;

	  if (numsize < densize)
	    {
	      mp_size_t empty = densize - numsize;
	      int i;

	      if (bits <= 0)
		exponent -= empty * BITS_PER_MP_LIMB;
	      else
		{
		  if (bits + empty * BITS_PER_MP_LIMB <= MANT_DIG)
		    {
		      /* We make a difference here because the compiler
			 cannot optimize the `else' case that good and
			 this reflects all currently used FLOAT types
			 and GMP implementations.  */
#if RETURN_LIMB_SIZE <= 2
		      assert (empty == 1);
		      __mpn_lshift_1 (retval, RETURN_LIMB_SIZE,
				      BITS_PER_MP_LIMB, 0);
#else
		      for (i = RETURN_LIMB_SIZE - 1; i >= empty; --i)
			retval[i] = retval[i - empty];
		      while (i >= 0)
			retval[i--] = 0;
#endif
		    }
		  else
		    {
		      used = MANT_DIG - bits;
		      if (used >= BITS_PER_MP_LIMB)
			{
			  int i;
			  (void) __mpn_lshift (&retval[used
						       / BITS_PER_MP_LIMB],
					       retval,
					       (RETURN_LIMB_SIZE
						- used / BITS_PER_MP_LIMB),
					       used % BITS_PER_MP_LIMB);
			  for (i = used / BITS_PER_MP_LIMB - 1; i >= 0; --i)
			    retval[i] = 0;
			}
		      else if (used > 0)
			__mpn_lshift_1 (retval, RETURN_LIMB_SIZE, used, 0);
		    }
		  bits += empty * BITS_PER_MP_LIMB;
		}
	      for (i = numsize; i > 0; --i)
		num[i + empty] = num[i - 1];
	      MPN_ZERO (num, empty + 1);
	    }
	  else
	    {
	      int i;
	      assert (numsize == densize);
	      for (i = numsize; i > 0; --i)
		num[i] = num[i - 1];
	      num[0] = 0;
	    }

	  den[densize] = 0;
	  n0 = num[densize];

	  while (bits <= MANT_DIG)
	    {
	      if (n0 == dX)
		/* This might over-estimate QUOT, but it's probably not
		   worth the extra code here to find out.  */
		quot = ~(mp_limb_t) 0;
	      else
		{
		  mp_limb_t r;

		  udiv_qrnnd (quot, r, n0, num[densize - 1], dX);
		  umul_ppmm (n1, n0, d1, quot);

		  while (n1 > r || (n1 == r && n0 > num[densize - 2]))
		    {
		      --quot;
		      r += dX;
		      if (r < dX) /* I.e. "carry in previous addition?" */
			break;
		      n1 -= n0 < d1;
		      n0 -= d1;
		    }
		}

	      /* Possible optimization: We already have (q * n0) and (1 * n1)
		 after the calculation of QUOT.  Taking advantage of this, we
		 could make this loop make two iterations less.  */

	      cy = __mpn_submul_1 (num, den, densize + 1, quot);

	      if (num[densize] != cy)
		{
		  cy = __mpn_add_n (num, num, den, densize);
		  assert (cy != 0);
		  --quot;
		}
	      n0 = num[densize] = num[densize - 1];
	      for (i = densize - 1; i > 0; --i)
		num[i] = num[i - 1];
	      num[0] = 0;

	      got_limb;
	    }

	  for (i = densize; i >= 0 && num[i] == 0; --i)
	    ;
	  return round_and_return (retval, exponent - 1, negative,
				   quot, BITS_PER_MP_LIMB - 1 - used,
				   more_bits || i >= 0);
	}
      }
  }

  /* NOTREACHED */
}
#if defined _LIBC && !defined USE_WIDE_CHAR
libc_hidden_def (____STRTOF_INTERNAL)
#endif

/* External user entry point.  */

FLOAT
#ifdef weak_function
weak_function
#endif
__STRTOF (const STRING_TYPE *nptr, STRING_TYPE **endptr, locale_t loc)
{
  return ____STRTOF_INTERNAL (nptr, endptr, 0, loc);
}
#if defined _LIBC
libc_hidden_def (__STRTOF)
libc_hidden_ver (__STRTOF, STRTOF)
#endif
weak_alias (__STRTOF, STRTOF)

#ifdef LONG_DOUBLE_COMPAT
# if LONG_DOUBLE_COMPAT(libc, GLIBC_2_1)
#  ifdef USE_WIDE_CHAR
compat_symbol (libc, __wcstod_l, __wcstold_l, GLIBC_2_1);
#  else
compat_symbol (libc, __strtod_l, __strtold_l, GLIBC_2_1);
#  endif
# endif
# if LONG_DOUBLE_COMPAT(libc, GLIBC_2_3)
#  ifdef USE_WIDE_CHAR
compat_symbol (libc, wcstod_l, wcstold_l, GLIBC_2_3);
#  else
compat_symbol (libc, strtod_l, strtold_l, GLIBC_2_3);
#  endif
# endif
#endif

#if BUILD_DOUBLE
# if __HAVE_FLOAT64 && !__HAVE_DISTINCT_FLOAT64
#  undef strtof64_l
#  undef wcstof64_l
#  ifdef USE_WIDE_CHAR
weak_alias (wcstod_l, wcstof64_l)
#  else
weak_alias (strtod_l, strtof64_l)
#  endif
# endif
# if __HAVE_FLOAT32X && !__HAVE_DISTINCT_FLOAT32X
#  undef strtof32x_l
#  undef wcstof32x_l
#  ifdef USE_WIDE_CHAR
weak_alias (wcstod_l, wcstof32x_l)
#  else
weak_alias (strtod_l, strtof32x_l)
#  endif
# endif
#endif
