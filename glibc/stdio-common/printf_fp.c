/* Floating point output for `printf'.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.
   Written by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1995.

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

/* The gmp headers need some configuration frobs.  */
#define HAVE_ALLOCA 1

#include <array_length.h>
#include <libioP.h>
#include <alloca.h>
#include <ctype.h>
#include <float.h>
#include <gmp-mparam.h>
#include <gmp.h>
#include <ieee754.h>
#include <stdlib/gmp-impl.h>
#include <stdlib/longlong.h>
#include <stdlib/fpioconst.h>
#include <locale/localeinfo.h>
#include <limits.h>
#include <math.h>
#include <printf.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <wchar.h>
#include <stdbool.h>
#include <rounding-mode.h>

#ifdef COMPILE_WPRINTF
# define CHAR_T        wchar_t
#else
# define CHAR_T        char
#endif

#include "_i18n_number.h"

#ifndef NDEBUG
# define NDEBUG			/* Undefine this for debugging assertions.  */
#endif
#include <assert.h>

#define PUT(f, s, n) _IO_sputn (f, s, n)
#define PAD(f, c, n) (wide ? _IO_wpadn (f, c, n) : _IO_padn (f, c, n))
#undef putc
#define putc(c, f) (wide \
		    ? (int)_IO_putwc_unlocked (c, f) : _IO_putc_unlocked (c, f))


/* Macros for doing the actual output.  */

#define outchar(ch)							      \
  do									      \
    {									      \
      const int outc = (ch);						      \
      if (putc (outc, fp) == EOF)					      \
	{								      \
	  if (buffer_malloced)						      \
	    {								      \
	      free (buffer);						      \
	      free (wbuffer);						      \
	    }								      \
	  return -1;							      \
	}								      \
      ++done;								      \
    } while (0)

#define PRINT(ptr, wptr, len)						      \
  do									      \
    {									      \
      size_t outlen = (len);						      \
      if (len > 20)							      \
	{								      \
	  if (PUT (fp, wide ? (const char *) wptr : ptr, outlen) != outlen)   \
	    {								      \
	      if (buffer_malloced)					      \
		{							      \
		  free (buffer);					      \
		  free (wbuffer);					      \
		}							      \
	      return -1;						      \
	    }								      \
	  ptr += outlen;						      \
	  done += outlen;						      \
	}								      \
      else								      \
	{								      \
	  if (wide)							      \
	    while (outlen-- > 0)					      \
	      outchar (*wptr++);					      \
	  else								      \
	    while (outlen-- > 0)					      \
	      outchar (*ptr++);						      \
	}								      \
    } while (0)

#define PADN(ch, len)							      \
  do									      \
    {									      \
      if (PAD (fp, ch, len) != len)					      \
	{								      \
	  if (buffer_malloced)						      \
	    {								      \
	      free (buffer);						      \
	      free (wbuffer);						      \
	    }								      \
	  return -1;							      \
	}								      \
      done += len;							      \
    }									      \
  while (0)

/* We use the GNU MP library to handle large numbers.

   An MP variable occupies a varying number of entries in its array.  We keep
   track of this number for efficiency reasons.  Otherwise we would always
   have to process the whole array.  */
#define MPN_VAR(name) mp_limb_t *name; mp_size_t name##size

#define MPN_ASSIGN(dst,src)						      \
  memcpy (dst, src, (dst##size = src##size) * sizeof (mp_limb_t))
#define MPN_GE(u,v) \
  (u##size > v##size || (u##size == v##size && __mpn_cmp (u, v, u##size) >= 0))

extern mp_size_t __mpn_extract_double (mp_ptr res_ptr, mp_size_t size,
				       int *expt, int *is_neg,
				       double value);
extern mp_size_t __mpn_extract_long_double (mp_ptr res_ptr, mp_size_t size,
					    int *expt, int *is_neg,
					    long double value);


static wchar_t *group_number (wchar_t *buf, wchar_t *bufend,
			      unsigned int intdig_no, const char *grouping,
			      wchar_t thousands_sep, int ngroups);

struct hack_digit_param
{
  /* Sign of the exponent.  */
  int expsign;
  /* The type of output format that will be used: 'e'/'E' or 'f'.  */
  int type;
  /* and the exponent.	*/
  int exponent;
  /* The fraction of the floting-point value in question  */
  MPN_VAR(frac);
  /* Scaling factor.  */
  MPN_VAR(scale);
  /* Temporary bignum value.  */
  MPN_VAR(tmp);
};

static wchar_t
hack_digit (struct hack_digit_param *p)
{
  mp_limb_t hi;

  if (p->expsign != 0 && p->type == 'f' && p->exponent-- > 0)
    hi = 0;
  else if (p->scalesize == 0)
    {
      hi = p->frac[p->fracsize - 1];
      p->frac[p->fracsize - 1] = __mpn_mul_1 (p->frac, p->frac,
	p->fracsize - 1, 10);
    }
  else
    {
      if (p->fracsize < p->scalesize)
	hi = 0;
      else
	{
	  hi = mpn_divmod (p->tmp, p->frac, p->fracsize,
	    p->scale, p->scalesize);
	  p->tmp[p->fracsize - p->scalesize] = hi;
	  hi = p->tmp[0];

	  p->fracsize = p->scalesize;
	  while (p->fracsize != 0 && p->frac[p->fracsize - 1] == 0)
	    --p->fracsize;
	  if (p->fracsize == 0)
	    {
	      /* We're not prepared for an mpn variable with zero
		 limbs.  */
	      p->fracsize = 1;
	      return L'0' + hi;
	    }
	}

      mp_limb_t _cy = __mpn_mul_1 (p->frac, p->frac, p->fracsize, 10);
      if (_cy != 0)
	p->frac[p->fracsize++] = _cy;
    }

  return L'0' + hi;
}

int
__printf_fp_l (FILE *fp, locale_t loc,
	       const struct printf_info *info,
	       const void *const *args)
{
  /* The floating-point value to output.  */
  union
    {
      double dbl;
      long double ldbl;
#if __HAVE_DISTINCT_FLOAT128
      _Float128 f128;
#endif
    }
  fpnum;

  /* Locale-dependent representation of decimal point.	*/
  const char *decimal;
  wchar_t decimalwc;

  /* Locale-dependent thousands separator and grouping specification.  */
  const char *thousands_sep = NULL;
  wchar_t thousands_sepwc = 0;
  const char *grouping;

  /* "NaN" or "Inf" for the special cases.  */
  const char *special = NULL;
  const wchar_t *wspecial = NULL;

  /* When _Float128 is enabled in the library and ABI-distinct from long
     double, we need mp_limbs enough for any of them.  */
#if __HAVE_DISTINCT_FLOAT128
# define GREATER_MANT_DIG FLT128_MANT_DIG
#else
# define GREATER_MANT_DIG LDBL_MANT_DIG
#endif
  /* We need just a few limbs for the input before shifting to the right
     position.	*/
  mp_limb_t fp_input[(GREATER_MANT_DIG + BITS_PER_MP_LIMB - 1)
		     / BITS_PER_MP_LIMB];
  /* We need to shift the contents of fp_input by this amount of bits.	*/
  int to_shift = 0;

  struct hack_digit_param p;
  /* Sign of float number.  */
  int is_neg = 0;

  /* Counter for number of written characters.	*/
  int done = 0;

  /* General helper (carry limb).  */
  mp_limb_t cy;

  /* Nonzero if this is output on a wide character stream.  */
  int wide = info->wide;

  /* Buffer in which we produce the output.  */
  wchar_t *wbuffer = NULL;
  char *buffer = NULL;
  /* Flag whether wbuffer and buffer are malloc'ed or not.  */
  int buffer_malloced = 0;

  p.expsign = 0;

  /* Figure out the decimal point character.  */
  if (info->extra == 0)
    {
      decimal = _nl_lookup (loc, LC_NUMERIC, DECIMAL_POINT);
      decimalwc = _nl_lookup_word
	(loc, LC_NUMERIC, _NL_NUMERIC_DECIMAL_POINT_WC);
    }
  else
    {
      decimal = _nl_lookup (loc, LC_MONETARY, MON_DECIMAL_POINT);
      if (*decimal == '\0')
	decimal = _nl_lookup (loc, LC_NUMERIC, DECIMAL_POINT);
      decimalwc = _nl_lookup_word (loc, LC_MONETARY,
				    _NL_MONETARY_DECIMAL_POINT_WC);
      if (decimalwc == L'\0')
	decimalwc = _nl_lookup_word (loc, LC_NUMERIC,
				      _NL_NUMERIC_DECIMAL_POINT_WC);
    }
  /* The decimal point character must not be zero.  */
  assert (*decimal != '\0');
  assert (decimalwc != L'\0');

  if (info->group)
    {
      if (info->extra == 0)
	grouping = _nl_lookup (loc, LC_NUMERIC, GROUPING);
      else
	grouping = _nl_lookup (loc, LC_MONETARY, MON_GROUPING);

      if (*grouping <= 0 || *grouping == CHAR_MAX)
	grouping = NULL;
      else
	{
	  /* Figure out the thousands separator character.  */
	  if (wide)
	    {
	      if (info->extra == 0)
		thousands_sepwc = _nl_lookup_word
		  (loc, LC_NUMERIC, _NL_NUMERIC_THOUSANDS_SEP_WC);
	      else
		thousands_sepwc =
		  _nl_lookup_word (loc, LC_MONETARY,
				    _NL_MONETARY_THOUSANDS_SEP_WC);
	    }
	  else
	    {
	      if (info->extra == 0)
		thousands_sep = _nl_lookup (loc, LC_NUMERIC, THOUSANDS_SEP);
	      else
		thousands_sep = _nl_lookup
		  (loc, LC_MONETARY, MON_THOUSANDS_SEP);
	    }

	  if ((wide && thousands_sepwc == L'\0')
	      || (! wide && *thousands_sep == '\0'))
	    grouping = NULL;
	  else if (thousands_sepwc == L'\0')
	    /* If we are printing multibyte characters and there is a
	       multibyte representation for the thousands separator,
	       we must ensure the wide character thousands separator
	       is available, even if it is fake.  */
	    thousands_sepwc = 0xfffffffe;
	}
    }
  else
    grouping = NULL;

#define PRINTF_FP_FETCH(FLOAT, VAR, SUFFIX, MANT_DIG)			\
  {									\
    (VAR) = *(const FLOAT *) args[0];					\
									\
    /* Check for special values: not a number or infinity.  */		\
    if (isnan (VAR))							\
      {									\
	is_neg = signbit (VAR);						\
	if (isupper (info->spec))					\
	  {								\
	    special = "NAN";						\
	    wspecial = L"NAN";						\
	  }								\
	else								\
	  {								\
	    special = "nan";						\
	    wspecial = L"nan";						\
	  }								\
      }									\
    else if (isinf (VAR))						\
      {									\
	is_neg = signbit (VAR);						\
	if (isupper (info->spec))					\
	  {								\
	    special = "INF";						\
	    wspecial = L"INF";						\
	  }								\
	else								\
	  {								\
	    special = "inf";						\
	    wspecial = L"inf";						\
	  }								\
      }									\
    else								\
      {									\
	p.fracsize = __mpn_extract_##SUFFIX				\
		     (fp_input, array_length (fp_input),		\
		      &p.exponent, &is_neg, VAR);			\
	to_shift = 1 + p.fracsize * BITS_PER_MP_LIMB - MANT_DIG;	\
      }									\
  }

  /* Fetch the argument value.	*/
#if __HAVE_DISTINCT_FLOAT128
  if (info->is_binary128)
    PRINTF_FP_FETCH (_Float128, fpnum.f128, float128, FLT128_MANT_DIG)
  else
#endif
#ifndef __NO_LONG_DOUBLE_MATH
  if (info->is_long_double && sizeof (long double) > sizeof (double))
    PRINTF_FP_FETCH (long double, fpnum.ldbl, long_double, LDBL_MANT_DIG)
  else
#endif
    PRINTF_FP_FETCH (double, fpnum.dbl, double, DBL_MANT_DIG)

#undef PRINTF_FP_FETCH

  if (special)
    {
      int width = info->width;

      if (is_neg || info->showsign || info->space)
	--width;
      width -= 3;

      if (!info->left && width > 0)
	PADN (' ', width);

      if (is_neg)
	outchar ('-');
      else if (info->showsign)
	outchar ('+');
      else if (info->space)
	outchar (' ');

      PRINT (special, wspecial, 3);

      if (info->left && width > 0)
	PADN (' ', width);

      return done;
    }


  /* We need three multiprecision variables.  Now that we have the p.exponent
     of the number we can allocate the needed memory.  It would be more
     efficient to use variables of the fixed maximum size but because this
     would be really big it could lead to memory problems.  */
  {
    mp_size_t bignum_size = ((abs (p.exponent) + BITS_PER_MP_LIMB - 1)
			     / BITS_PER_MP_LIMB
			     + (GREATER_MANT_DIG / BITS_PER_MP_LIMB > 2
				? 8 : 4))
			    * sizeof (mp_limb_t);
    p.frac = (mp_limb_t *) alloca (bignum_size);
    p.tmp = (mp_limb_t *) alloca (bignum_size);
    p.scale = (mp_limb_t *) alloca (bignum_size);
  }

  /* We now have to distinguish between numbers with positive and negative
     exponents because the method used for the one is not applicable/efficient
     for the other.  */
  p.scalesize = 0;
  if (p.exponent > 2)
    {
      /* |FP| >= 8.0.  */
      int scaleexpo = 0;
      int explog;
#if __HAVE_DISTINCT_FLOAT128
      if (info->is_binary128)
	explog = FLT128_MAX_10_EXP_LOG;
      else
	explog = LDBL_MAX_10_EXP_LOG;
#else
      explog = LDBL_MAX_10_EXP_LOG;
#endif
      int exp10 = 0;
      const struct mp_power *powers = &_fpioconst_pow10[explog + 1];
      int cnt_h, cnt_l, i;

      if ((p.exponent + to_shift) % BITS_PER_MP_LIMB == 0)
	{
	  MPN_COPY_DECR (p.frac + (p.exponent + to_shift) / BITS_PER_MP_LIMB,
			 fp_input, p.fracsize);
	  p.fracsize += (p.exponent + to_shift) / BITS_PER_MP_LIMB;
	}
      else
	{
	  cy = __mpn_lshift (p.frac
			     + (p.exponent + to_shift) / BITS_PER_MP_LIMB,
			     fp_input, p.fracsize,
			     (p.exponent + to_shift) % BITS_PER_MP_LIMB);
	  p.fracsize += (p.exponent + to_shift) / BITS_PER_MP_LIMB;
	  if (cy)
	    p.frac[p.fracsize++] = cy;
	}
      MPN_ZERO (p.frac, (p.exponent + to_shift) / BITS_PER_MP_LIMB);

      assert (powers > &_fpioconst_pow10[0]);
      do
	{
	  --powers;

	  /* The number of the product of two binary numbers with n and m
	     bits respectively has m+n or m+n-1 bits.	*/
	  if (p.exponent >= scaleexpo + powers->p_expo - 1)
	    {
	      if (p.scalesize == 0)
		{
#if __HAVE_DISTINCT_FLOAT128
		  if ((FLT128_MANT_DIG
			    > _FPIO_CONST_OFFSET * BITS_PER_MP_LIMB)
			   && info->is_binary128)
		    {
#define _FLT128_FPIO_CONST_SHIFT \
  (((FLT128_MANT_DIG + BITS_PER_MP_LIMB - 1) / BITS_PER_MP_LIMB) \
   - _FPIO_CONST_OFFSET)
		      /* 64bit const offset is not enough for
			 IEEE 854 quad long double (_Float128).  */
		      p.tmpsize = powers->arraysize + _FLT128_FPIO_CONST_SHIFT;
		      memcpy (p.tmp + _FLT128_FPIO_CONST_SHIFT,
			      &__tens[powers->arrayoff],
			      p.tmpsize * sizeof (mp_limb_t));
		      MPN_ZERO (p.tmp, _FLT128_FPIO_CONST_SHIFT);
		      /* Adjust p.exponent, as scaleexpo will be this much
			 bigger too.  */
		      p.exponent += _FLT128_FPIO_CONST_SHIFT * BITS_PER_MP_LIMB;
		    }
		  else
#endif /* __HAVE_DISTINCT_FLOAT128 */
#ifndef __NO_LONG_DOUBLE_MATH
		  if (LDBL_MANT_DIG > _FPIO_CONST_OFFSET * BITS_PER_MP_LIMB
		      && info->is_long_double)
		    {
#define _FPIO_CONST_SHIFT \
  (((LDBL_MANT_DIG + BITS_PER_MP_LIMB - 1) / BITS_PER_MP_LIMB) \
   - _FPIO_CONST_OFFSET)
		      /* 64bit const offset is not enough for
			 IEEE quad long double.  */
		      p.tmpsize = powers->arraysize + _FPIO_CONST_SHIFT;
		      memcpy (p.tmp + _FPIO_CONST_SHIFT,
			      &__tens[powers->arrayoff],
			      p.tmpsize * sizeof (mp_limb_t));
		      MPN_ZERO (p.tmp, _FPIO_CONST_SHIFT);
		      /* Adjust p.exponent, as scaleexpo will be this much
			 bigger too.  */
		      p.exponent += _FPIO_CONST_SHIFT * BITS_PER_MP_LIMB;
		    }
		  else
#endif
		    {
		      p.tmpsize = powers->arraysize;
		      memcpy (p.tmp, &__tens[powers->arrayoff],
			      p.tmpsize * sizeof (mp_limb_t));
		    }
		}
	      else
		{
		  cy = __mpn_mul (p.tmp, p.scale, p.scalesize,
				  &__tens[powers->arrayoff
					 + _FPIO_CONST_OFFSET],
				  powers->arraysize - _FPIO_CONST_OFFSET);
		  p.tmpsize = p.scalesize
		    + powers->arraysize - _FPIO_CONST_OFFSET;
		  if (cy == 0)
		    --p.tmpsize;
		}

	      if (MPN_GE (p.frac, p.tmp))
		{
		  int cnt;
		  MPN_ASSIGN (p.scale, p.tmp);
		  count_leading_zeros (cnt, p.scale[p.scalesize - 1]);
		  scaleexpo = (p.scalesize - 2) * BITS_PER_MP_LIMB - cnt - 1;
		  exp10 |= 1 << explog;
		}
	    }
	  --explog;
	}
      while (powers > &_fpioconst_pow10[0]);
      p.exponent = exp10;

      /* Optimize number representations.  We want to represent the numbers
	 with the lowest number of bytes possible without losing any
	 bytes. Also the highest bit in the scaling factor has to be set
	 (this is a requirement of the MPN division routines).  */
      if (p.scalesize > 0)
	{
	  /* Determine minimum number of zero bits at the end of
	     both numbers.  */
	  for (i = 0; p.scale[i] == 0 && p.frac[i] == 0; i++)
	    ;

	  /* Determine number of bits the scaling factor is misplaced.	*/
	  count_leading_zeros (cnt_h, p.scale[p.scalesize - 1]);

	  if (cnt_h == 0)
	    {
	      /* The highest bit of the scaling factor is already set.	So
		 we only have to remove the trailing empty limbs.  */
	      if (i > 0)
		{
		  MPN_COPY_INCR (p.scale, p.scale + i, p.scalesize - i);
		  p.scalesize -= i;
		  MPN_COPY_INCR (p.frac, p.frac + i, p.fracsize - i);
		  p.fracsize -= i;
		}
	    }
	  else
	    {
	      if (p.scale[i] != 0)
		{
		  count_trailing_zeros (cnt_l, p.scale[i]);
		  if (p.frac[i] != 0)
		    {
		      int cnt_l2;
		      count_trailing_zeros (cnt_l2, p.frac[i]);
		      if (cnt_l2 < cnt_l)
			cnt_l = cnt_l2;
		    }
		}
	      else
		count_trailing_zeros (cnt_l, p.frac[i]);

	      /* Now shift the numbers to their optimal position.  */
	      if (i == 0 && BITS_PER_MP_LIMB - cnt_h > cnt_l)
		{
		  /* We cannot save any memory.	 So just roll both numbers
		     so that the scaling factor has its highest bit set.  */

		  (void) __mpn_lshift (p.scale, p.scale, p.scalesize, cnt_h);
		  cy = __mpn_lshift (p.frac, p.frac, p.fracsize, cnt_h);
		  if (cy != 0)
		    p.frac[p.fracsize++] = cy;
		}
	      else if (BITS_PER_MP_LIMB - cnt_h <= cnt_l)
		{
		  /* We can save memory by removing the trailing zero limbs
		     and by packing the non-zero limbs which gain another
		     free one. */

		  (void) __mpn_rshift (p.scale, p.scale + i, p.scalesize - i,
				       BITS_PER_MP_LIMB - cnt_h);
		  p.scalesize -= i + 1;
		  (void) __mpn_rshift (p.frac, p.frac + i, p.fracsize - i,
				       BITS_PER_MP_LIMB - cnt_h);
		  p.fracsize -= p.frac[p.fracsize - i - 1] == 0 ? i + 1 : i;
		}
	      else
		{
		  /* We can only save the memory of the limbs which are zero.
		     The non-zero parts occupy the same number of limbs.  */

		  (void) __mpn_rshift (p.scale, p.scale + (i - 1),
				       p.scalesize - (i - 1),
				       BITS_PER_MP_LIMB - cnt_h);
		  p.scalesize -= i;
		  (void) __mpn_rshift (p.frac, p.frac + (i - 1),
				       p.fracsize - (i - 1),
				       BITS_PER_MP_LIMB - cnt_h);
		  p.fracsize -=
		    p.frac[p.fracsize - (i - 1) - 1] == 0 ? i : i - 1;
		}
	    }
	}
    }
  else if (p.exponent < 0)
    {
      /* |FP| < 1.0.  */
      int exp10 = 0;
      int explog;
#if __HAVE_DISTINCT_FLOAT128
      if (info->is_binary128)
	explog = FLT128_MAX_10_EXP_LOG;
      else
	explog = LDBL_MAX_10_EXP_LOG;
#else
      explog = LDBL_MAX_10_EXP_LOG;
#endif
      const struct mp_power *powers = &_fpioconst_pow10[explog + 1];

      /* Now shift the input value to its right place.	*/
      cy = __mpn_lshift (p.frac, fp_input, p.fracsize, to_shift);
      p.frac[p.fracsize++] = cy;
      assert (cy == 1 || (p.frac[p.fracsize - 2] == 0 && p.frac[0] == 0));

      p.expsign = 1;
      p.exponent = -p.exponent;

      assert (powers != &_fpioconst_pow10[0]);
      do
	{
	  --powers;

	  if (p.exponent >= powers->m_expo)
	    {
	      int i, incr, cnt_h, cnt_l;
	      mp_limb_t topval[2];

	      /* The __mpn_mul function expects the first argument to be
		 bigger than the second.  */
	      if (p.fracsize < powers->arraysize - _FPIO_CONST_OFFSET)
		cy = __mpn_mul (p.tmp, &__tens[powers->arrayoff
					    + _FPIO_CONST_OFFSET],
				powers->arraysize - _FPIO_CONST_OFFSET,
				p.frac, p.fracsize);
	      else
		cy = __mpn_mul (p.tmp, p.frac, p.fracsize,
				&__tens[powers->arrayoff + _FPIO_CONST_OFFSET],
				powers->arraysize - _FPIO_CONST_OFFSET);
	      p.tmpsize = p.fracsize + powers->arraysize - _FPIO_CONST_OFFSET;
	      if (cy == 0)
		--p.tmpsize;

	      count_leading_zeros (cnt_h, p.tmp[p.tmpsize - 1]);
	      incr = (p.tmpsize - p.fracsize) * BITS_PER_MP_LIMB
		     + BITS_PER_MP_LIMB - 1 - cnt_h;

	      assert (incr <= powers->p_expo);

	      /* If we increased the p.exponent by exactly 3 we have to test
		 for overflow.	This is done by comparing with 10 shifted
		 to the right position.	 */
	      if (incr == p.exponent + 3)
		{
		  if (cnt_h <= BITS_PER_MP_LIMB - 4)
		    {
		      topval[0] = 0;
		      topval[1]
			= ((mp_limb_t) 10) << (BITS_PER_MP_LIMB - 4 - cnt_h);
		    }
		  else
		    {
		      topval[0] = ((mp_limb_t) 10) << (BITS_PER_MP_LIMB - 4);
		      topval[1] = 0;
		      (void) __mpn_lshift (topval, topval, 2,
					   BITS_PER_MP_LIMB - cnt_h);
		    }
		}

	      /* We have to be careful when multiplying the last factor.
		 If the result is greater than 1.0 be have to test it
		 against 10.0.  If it is greater or equal to 10.0 the
		 multiplication was not valid.  This is because we cannot
		 determine the number of bits in the result in advance.  */
	      if (incr < p.exponent + 3
		  || (incr == p.exponent + 3
		      && (p.tmp[p.tmpsize - 1] < topval[1]
			  || (p.tmp[p.tmpsize - 1] == topval[1]
			      && p.tmp[p.tmpsize - 2] < topval[0]))))
		{
		  /* The factor is right.  Adapt binary and decimal
		     exponents.	 */
		  p.exponent -= incr;
		  exp10 |= 1 << explog;

		  /* If this factor yields a number greater or equal to
		     1.0, we must not shift the non-fractional digits down. */
		  if (p.exponent < 0)
		    cnt_h += -p.exponent;

		  /* Now we optimize the number representation.	 */
		  for (i = 0; p.tmp[i] == 0; ++i);
		  if (cnt_h == BITS_PER_MP_LIMB - 1)
		    {
		      MPN_COPY (p.frac, p.tmp + i, p.tmpsize - i);
		      p.fracsize = p.tmpsize - i;
		    }
		  else
		    {
		      count_trailing_zeros (cnt_l, p.tmp[i]);

		      /* Now shift the numbers to their optimal position.  */
		      if (i == 0 && BITS_PER_MP_LIMB - 1 - cnt_h > cnt_l)
			{
			  /* We cannot save any memory.	 Just roll the
			     number so that the leading digit is in a
			     separate limb.  */

			  cy = __mpn_lshift (p.frac, p.tmp, p.tmpsize,
			    cnt_h + 1);
			  p.fracsize = p.tmpsize + 1;
			  p.frac[p.fracsize - 1] = cy;
			}
		      else if (BITS_PER_MP_LIMB - 1 - cnt_h <= cnt_l)
			{
			  (void) __mpn_rshift (p.frac, p.tmp + i, p.tmpsize - i,
					       BITS_PER_MP_LIMB - 1 - cnt_h);
			  p.fracsize = p.tmpsize - i;
			}
		      else
			{
			  /* We can only save the memory of the limbs which
			     are zero.	The non-zero parts occupy the same
			     number of limbs.  */

			  (void) __mpn_rshift (p.frac, p.tmp + (i - 1),
					       p.tmpsize - (i - 1),
					       BITS_PER_MP_LIMB - 1 - cnt_h);
			  p.fracsize = p.tmpsize - (i - 1);
			}
		    }
		}
	    }
	  --explog;
	}
      while (powers != &_fpioconst_pow10[1] && p.exponent > 0);
      /* All factors but 10^-1 are tested now.	*/
      if (p.exponent > 0)
	{
	  int cnt_l;

	  cy = __mpn_mul_1 (p.tmp, p.frac, p.fracsize, 10);
	  p.tmpsize = p.fracsize;
	  assert (cy == 0 || p.tmp[p.tmpsize - 1] < 20);

	  count_trailing_zeros (cnt_l, p.tmp[0]);
	  if (cnt_l < MIN (4, p.exponent))
	    {
	      cy = __mpn_lshift (p.frac, p.tmp, p.tmpsize,
				 BITS_PER_MP_LIMB - MIN (4, p.exponent));
	      if (cy != 0)
		p.frac[p.tmpsize++] = cy;
	    }
	  else
	    (void) __mpn_rshift (p.frac, p.tmp, p.tmpsize, MIN (4, p.exponent));
	  p.fracsize = p.tmpsize;
	  exp10 |= 1;
	  assert (p.frac[p.fracsize - 1] < 10);
	}
      p.exponent = exp10;
    }
  else
    {
      /* This is a special case.  We don't need a factor because the
	 numbers are in the range of 1.0 <= |fp| < 8.0.  We simply
	 shift it to the right place and divide it by 1.0 to get the
	 leading digit.	 (Of course this division is not really made.)	*/
      assert (0 <= p.exponent && p.exponent < 3
	      && p.exponent + to_shift < BITS_PER_MP_LIMB);

      /* Now shift the input value to its right place.	*/
      cy = __mpn_lshift (p.frac, fp_input, p.fracsize, (p.exponent + to_shift));
      p.frac[p.fracsize++] = cy;
      p.exponent = 0;
    }

  {
    int width = info->width;
    wchar_t *wstartp, *wcp;
    size_t chars_needed;
    int expscale;
    int intdig_max, intdig_no = 0;
    int fracdig_min;
    int fracdig_max;
    int dig_max;
    int significant;
    int ngroups = 0;
    char spec = _tolower (info->spec);

    if (spec == 'e')
      {
	p.type = info->spec;
	intdig_max = 1;
	fracdig_min = fracdig_max = info->prec < 0 ? 6 : info->prec;
	chars_needed = 1 + 1 + (size_t) fracdig_max + 1 + 1 + 4;
	/*	       d   .	 ddd	     e	 +-  ddd  */
	dig_max = INT_MAX;		/* Unlimited.  */
	significant = 1;		/* Does not matter here.  */
      }
    else if (spec == 'f')
      {
	p.type = 'f';
	fracdig_min = fracdig_max = info->prec < 0 ? 6 : info->prec;
	dig_max = INT_MAX;		/* Unlimited.  */
	significant = 1;		/* Does not matter here.  */
	if (p.expsign == 0)
	  {
	    intdig_max = p.exponent + 1;
	    /* This can be really big!	*/  /* XXX Maybe malloc if too big? */
	    chars_needed = (size_t) p.exponent + 1 + 1 + (size_t) fracdig_max;
	  }
	else
	  {
	    intdig_max = 1;
	    chars_needed = 1 + 1 + (size_t) fracdig_max;
	  }
      }
    else
      {
	dig_max = info->prec < 0 ? 6 : (info->prec == 0 ? 1 : info->prec);
	if ((p.expsign == 0 && p.exponent >= dig_max)
	    || (p.expsign != 0 && p.exponent > 4))
	  {
	    if ('g' - 'G' == 'e' - 'E')
	      p.type = 'E' + (info->spec - 'G');
	    else
	      p.type = isupper (info->spec) ? 'E' : 'e';
	    fracdig_max = dig_max - 1;
	    intdig_max = 1;
	    chars_needed = 1 + 1 + (size_t) fracdig_max + 1 + 1 + 4;
	  }
	else
	  {
	    p.type = 'f';
	    intdig_max = p.expsign == 0 ? p.exponent + 1 : 0;
	    fracdig_max = dig_max - intdig_max;
	    /* We need space for the significant digits and perhaps
	       for leading zeros when < 1.0.  The number of leading
	       zeros can be as many as would be required for
	       exponential notation with a negative two-digit
	       p.exponent, which is 4.  */
	    chars_needed = (size_t) dig_max + 1 + 4;
	  }
	fracdig_min = info->alt ? fracdig_max : 0;
	significant = 0;		/* We count significant digits.	 */
      }

    if (grouping)
      {
	/* Guess the number of groups we will make, and thus how
	   many spaces we need for separator characters.  */
	ngroups = __guess_grouping (intdig_max, grouping);
	/* Allocate one more character in case rounding increases the
	   number of groups.  */
	chars_needed += ngroups + 1;
      }

    /* Allocate buffer for output.  We need two more because while rounding
       it is possible that we need two more characters in front of all the
       other output.  If the amount of memory we have to allocate is too
       large use `malloc' instead of `alloca'.  */
    if (__builtin_expect (chars_needed >= (size_t) -1 / sizeof (wchar_t) - 2
			  || chars_needed < fracdig_max, 0))
      {
	/* Some overflow occurred.  */
	__set_errno (ERANGE);
	return -1;
      }
    size_t wbuffer_to_alloc = (2 + chars_needed) * sizeof (wchar_t);
    buffer_malloced = ! __libc_use_alloca (wbuffer_to_alloc);
    if (__builtin_expect (buffer_malloced, 0))
      {
	wbuffer = (wchar_t *) malloc (wbuffer_to_alloc);
	if (wbuffer == NULL)
	  /* Signal an error to the caller.  */
	  return -1;
      }
    else
      wbuffer = (wchar_t *) alloca (wbuffer_to_alloc);
    wcp = wstartp = wbuffer + 2;	/* Let room for rounding.  */

    /* Do the real work: put digits in allocated buffer.  */
    if (p.expsign == 0 || p.type != 'f')
      {
	assert (p.expsign == 0 || intdig_max == 1);
	while (intdig_no < intdig_max)
	  {
	    ++intdig_no;
	    *wcp++ = hack_digit (&p);
	  }
	significant = 1;
	if (info->alt
	    || fracdig_min > 0
	    || (fracdig_max > 0 && (p.fracsize > 1 || p.frac[0] != 0)))
	  *wcp++ = decimalwc;
      }
    else
      {
	/* |fp| < 1.0 and the selected p.type is 'f', so put "0."
	   in the buffer.  */
	*wcp++ = L'0';
	--p.exponent;
	*wcp++ = decimalwc;
      }

    /* Generate the needed number of fractional digits.	 */
    int fracdig_no = 0;
    int added_zeros = 0;
    while (fracdig_no < fracdig_min + added_zeros
	   || (fracdig_no < fracdig_max && (p.fracsize > 1 || p.frac[0] != 0)))
      {
	++fracdig_no;
	*wcp = hack_digit (&p);
	if (*wcp++ != L'0')
	  significant = 1;
	else if (significant == 0)
	  {
	    ++fracdig_max;
	    if (fracdig_min > 0)
	      ++added_zeros;
	  }
      }

    /* Do rounding.  */
    wchar_t last_digit = wcp[-1] != decimalwc ? wcp[-1] : wcp[-2];
    wchar_t next_digit = hack_digit (&p);
    bool more_bits;
    if (next_digit != L'0' && next_digit != L'5')
      more_bits = true;
    else if (p.fracsize == 1 && p.frac[0] == 0)
      /* Rest of the number is zero.  */
      more_bits = false;
    else if (p.scalesize == 0)
      {
	/* Here we have to see whether all limbs are zero since no
	   normalization happened.  */
	size_t lcnt = p.fracsize;
	while (lcnt >= 1 && p.frac[lcnt - 1] == 0)
	  --lcnt;
	more_bits = lcnt > 0;
      }
    else
      more_bits = true;
    int rounding_mode = get_rounding_mode ();
    if (round_away (is_neg, (last_digit - L'0') & 1, next_digit >= L'5',
		    more_bits, rounding_mode))
      {
	wchar_t *wtp = wcp;

	if (fracdig_no > 0)
	  {
	    /* Process fractional digits.  Terminate if not rounded or
	       radix character is reached.  */
	    int removed = 0;
	    while (*--wtp != decimalwc && *wtp == L'9')
	      {
		*wtp = L'0';
		++removed;
	      }
	    if (removed == fracdig_min && added_zeros > 0)
	      --added_zeros;
	    if (*wtp != decimalwc)
	      /* Round up.  */
	      (*wtp)++;
	    else if (__builtin_expect (spec == 'g' && p.type == 'f' && info->alt
				       && wtp == wstartp + 1
				       && wstartp[0] == L'0',
				       0))
	      /* This is a special case: the rounded number is 1.0,
		 the format is 'g' or 'G', and the alternative format
		 is selected.  This means the result must be "1.".  */
	      --added_zeros;
	  }

	if (fracdig_no == 0 || *wtp == decimalwc)
	  {
	    /* Round the integer digits.  */
	    if (*(wtp - 1) == decimalwc)
	      --wtp;

	    while (--wtp >= wstartp && *wtp == L'9')
	      *wtp = L'0';

	    if (wtp >= wstartp)
	      /* Round up.  */
	      (*wtp)++;
	    else
	      /* It is more critical.  All digits were 9's.  */
	      {
		if (p.type != 'f')
		  {
		    *wstartp = '1';
		    p.exponent += p.expsign == 0 ? 1 : -1;

		    /* The above p.exponent adjustment could lead to 1.0e-00,
		       e.g. for 0.999999999.  Make sure p.exponent 0 always
		       uses + sign.  */
		    if (p.exponent == 0)
		      p.expsign = 0;
		  }
		else if (intdig_no == dig_max)
		  {
		    /* This is the case where for p.type %g the number fits
		       really in the range for %f output but after rounding
		       the number of digits is too big.	 */
		    *--wstartp = decimalwc;
		    *--wstartp = L'1';

		    if (info->alt || fracdig_no > 0)
		      {
			/* Overwrite the old radix character.  */
			wstartp[intdig_no + 2] = L'0';
			++fracdig_no;
		      }

		    fracdig_no += intdig_no;
		    intdig_no = 1;
		    fracdig_max = intdig_max - intdig_no;
		    ++p.exponent;
		    /* Now we must print the p.exponent.	*/
		    p.type = isupper (info->spec) ? 'E' : 'e';
		  }
		else
		  {
		    /* We can simply add another another digit before the
		       radix.  */
		    *--wstartp = L'1';
		    ++intdig_no;
		  }

		/* While rounding the number of digits can change.
		   If the number now exceeds the limits remove some
		   fractional digits.  */
		if (intdig_no + fracdig_no > dig_max)
		  {
		    wcp -= intdig_no + fracdig_no - dig_max;
		    fracdig_no -= intdig_no + fracdig_no - dig_max;
		  }
	      }
	  }
      }

    /* Now remove unnecessary '0' at the end of the string.  */
    while (fracdig_no > fracdig_min + added_zeros && *(wcp - 1) == L'0')
      {
	--wcp;
	--fracdig_no;
      }
    /* If we eliminate all fractional digits we perhaps also can remove
       the radix character.  */
    if (fracdig_no == 0 && !info->alt && *(wcp - 1) == decimalwc)
      --wcp;

    if (grouping)
      {
	/* Rounding might have changed the number of groups.  We allocated
	   enough memory but we need here the correct number of groups.  */
	if (intdig_no != intdig_max)
	  ngroups = __guess_grouping (intdig_no, grouping);

	/* Add in separator characters, overwriting the same buffer.  */
	wcp = group_number (wstartp, wcp, intdig_no, grouping, thousands_sepwc,
			    ngroups);
      }

    /* Write the p.exponent if it is needed.  */
    if (p.type != 'f')
      {
	if (__glibc_unlikely (p.expsign != 0 && p.exponent == 4 && spec == 'g'))
	  {
	    /* This is another special case.  The p.exponent of the number is
	       really smaller than -4, which requires the 'e'/'E' format.
	       But after rounding the number has an p.exponent of -4.  */
	    assert (wcp >= wstartp + 1);
	    assert (wstartp[0] == L'1');
	    __wmemcpy (wstartp, L"0.0001", 6);
	    wstartp[1] = decimalwc;
	    if (wcp >= wstartp + 2)
	      {
		__wmemset (wstartp + 6, L'0', wcp - (wstartp + 2));
		wcp += 4;
	      }
	    else
	      wcp += 5;
	  }
	else
	  {
	    *wcp++ = (wchar_t) p.type;
	    *wcp++ = p.expsign ? L'-' : L'+';

	    /* Find the magnitude of the p.exponent.	*/
	    expscale = 10;
	    while (expscale <= p.exponent)
	      expscale *= 10;

	    if (p.exponent < 10)
	      /* Exponent always has at least two digits.  */
	      *wcp++ = L'0';
	    else
	      do
		{
		  expscale /= 10;
		  *wcp++ = L'0' + (p.exponent / expscale);
		  p.exponent %= expscale;
		}
	      while (expscale > 10);
	    *wcp++ = L'0' + p.exponent;
	  }
      }

    /* Compute number of characters which must be filled with the padding
       character.  */
    if (is_neg || info->showsign || info->space)
      --width;
    width -= wcp - wstartp;

    if (!info->left && info->pad != '0' && width > 0)
      PADN (info->pad, width);

    if (is_neg)
      outchar ('-');
    else if (info->showsign)
      outchar ('+');
    else if (info->space)
      outchar (' ');

    if (!info->left && info->pad == '0' && width > 0)
      PADN ('0', width);

    {
      char *buffer_end = NULL;
      char *cp = NULL;
      char *tmpptr;

      if (! wide)
	{
	  /* Create the single byte string.  */
	  size_t decimal_len;
	  size_t thousands_sep_len;
	  wchar_t *copywc;
	  size_t factor;
	  if (info->i18n)
	    factor = _nl_lookup_word (loc, LC_CTYPE, _NL_CTYPE_MB_CUR_MAX);
	  else
	    factor = 1;

	  decimal_len = strlen (decimal);

	  if (thousands_sep == NULL)
	    thousands_sep_len = 0;
	  else
	    thousands_sep_len = strlen (thousands_sep);

	  size_t nbuffer = (2 + chars_needed * factor + decimal_len
			    + ngroups * thousands_sep_len);
	  if (__glibc_unlikely (buffer_malloced))
	    {
	      buffer = (char *) malloc (nbuffer);
	      if (buffer == NULL)
		{
		  /* Signal an error to the caller.  */
		  free (wbuffer);
		  return -1;
		}
	    }
	  else
	    buffer = (char *) alloca (nbuffer);
	  buffer_end = buffer + nbuffer;

	  /* Now copy the wide character string.  Since the character
	     (except for the decimal point and thousands separator) must
	     be coming from the ASCII range we can esily convert the
	     string without mapping tables.  */
	  for (cp = buffer, copywc = wstartp; copywc < wcp; ++copywc)
	    if (*copywc == decimalwc)
	      cp = (char *) __mempcpy (cp, decimal, decimal_len);
	    else if (*copywc == thousands_sepwc)
	      cp = (char *) __mempcpy (cp, thousands_sep, thousands_sep_len);
	    else
	      *cp++ = (char) *copywc;
	}

      tmpptr = buffer;
      if (__glibc_unlikely (info->i18n))
	{
#ifdef COMPILE_WPRINTF
	  wstartp = _i18n_number_rewrite (wstartp, wcp,
					  wbuffer + wbuffer_to_alloc);
	  wcp = wbuffer + wbuffer_to_alloc;
	  assert ((uintptr_t) wbuffer <= (uintptr_t) wstartp);
	  assert ((uintptr_t) wstartp
		  < (uintptr_t) wbuffer + wbuffer_to_alloc);
#else
	  tmpptr = _i18n_number_rewrite (tmpptr, cp, buffer_end);
	  cp = buffer_end;
	  assert ((uintptr_t) buffer <= (uintptr_t) tmpptr);
	  assert ((uintptr_t) tmpptr < (uintptr_t) buffer_end);
#endif
	}

      PRINT (tmpptr, wstartp, wide ? wcp - wstartp : cp - tmpptr);

      /* Free the memory if necessary.  */
      if (__glibc_unlikely (buffer_malloced))
	{
	  free (buffer);
	  free (wbuffer);
	  /* Avoid a double free if the subsequent PADN encounters an
	     I/O error.  */
	  buffer = NULL;
	  wbuffer = NULL;
	}
    }

    if (info->left && width > 0)
      PADN (info->pad, width);
  }
  return done;
}
libc_hidden_def (__printf_fp_l)

int
___printf_fp (FILE *fp, const struct printf_info *info,
	      const void *const *args)
{
  return __printf_fp_l (fp, _NL_CURRENT_LOCALE, info, args);
}
ldbl_hidden_def (___printf_fp, __printf_fp)
ldbl_strong_alias (___printf_fp, __printf_fp)


/* Return the number of extra grouping characters that will be inserted
   into a number with INTDIG_MAX integer digits.  */

unsigned int
__guess_grouping (unsigned int intdig_max, const char *grouping)
{
  unsigned int groups;

  /* We treat all negative values like CHAR_MAX.  */

  if (*grouping == CHAR_MAX || *grouping <= 0)
    /* No grouping should be done.  */
    return 0;

  groups = 0;
  while (intdig_max > (unsigned int) *grouping)
    {
      ++groups;
      intdig_max -= *grouping++;

      if (*grouping == CHAR_MAX
#if CHAR_MIN < 0
	  || *grouping < 0
#endif
	  )
	/* No more grouping should be done.  */
	break;
      else if (*grouping == 0)
	{
	  /* Same grouping repeats.  */
	  groups += (intdig_max - 1) / grouping[-1];
	  break;
	}
    }

  return groups;
}

/* Group the INTDIG_NO integer digits of the number in [BUF,BUFEND).
   There is guaranteed enough space past BUFEND to extend it.
   Return the new end of buffer.  */

static wchar_t *
group_number (wchar_t *buf, wchar_t *bufend, unsigned int intdig_no,
	      const char *grouping, wchar_t thousands_sep, int ngroups)
{
  wchar_t *p;

  if (ngroups == 0)
    return bufend;

  /* Move the fractional part down.  */
  __wmemmove (buf + intdig_no + ngroups, buf + intdig_no,
	      bufend - (buf + intdig_no));

  p = buf + intdig_no + ngroups - 1;
  do
    {
      unsigned int len = *grouping++;
      do
	*p-- = buf[--intdig_no];
      while (--len > 0);
      *p-- = thousands_sep;

      if (*grouping == CHAR_MAX
#if CHAR_MIN < 0
	  || *grouping < 0
#endif
	  )
	/* No more grouping should be done.  */
	break;
      else if (*grouping == 0)
	/* Same grouping repeats.  */
	--grouping;
    } while (intdig_no > (unsigned int) *grouping);

  /* Copy the remaining ungrouped digits.  */
  do
    *p-- = buf[--intdig_no];
  while (p > buf);

  return bufend + ngroups;
}
