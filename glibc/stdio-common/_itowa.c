/* Internal function for converting integers to ASCII.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Torbjorn Granlund <tege@matematik.su.se>
   and Ulrich Drepper <drepper@gnu.org>.

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

#include <gmp-mparam.h>
#include <gmp.h>
#include <limits.h>
#include <stdlib/gmp-impl.h>
#include <stdlib/longlong.h>

#include <_itowa.h>


/* Canonize environment.  For some architectures not all values might
   be defined in the GMP header files.  */
#ifndef UMUL_TIME
# define UMUL_TIME 1
#endif
#ifndef UDIV_TIME
# define UDIV_TIME 3
#endif

/* Control memory layout.  */
#ifdef PACK
# undef PACK
# define PACK __attribute__ ((packed))
#else
# define PACK
#endif


/* Declare local types.  */
struct base_table_t
{
#if (UDIV_TIME > 2 * UMUL_TIME)
  mp_limb_t base_multiplier;
#endif
  char flag;
  char post_shift;
#if BITS_PER_MP_LIMB == 32
  struct
    {
      char normalization_steps;
      char ndigits;
      mp_limb_t base PACK;
#if UDIV_TIME > 2 * UMUL_TIME
      mp_limb_t base_ninv PACK;
#endif
    } big;
#endif
};

/* To reduce the memory needed we include some fields of the tables
   only conditionally.  */
#if UDIV_TIME > 2 * UMUL_TIME
# define SEL1(X) X,
# define SEL2(X) ,X
#else
# define SEL1(X)
# define SEL2(X)
#endif

/* Factor table for the different bases.  */
extern const struct base_table_t _itoa_base_table[] attribute_hidden;

/* Lower-case digits.  */
extern const wchar_t _itowa_lower_digits[] attribute_hidden;
/* Upper-case digits.  */
extern const wchar_t _itowa_upper_digits[] attribute_hidden;


#if _ITOA_NEEDED
wchar_t *
_itowa (unsigned long long int value, wchar_t *buflim, unsigned int base,
	int upper_case)
{
  const wchar_t *digits = (upper_case
			   ? _itowa_upper_digits : _itowa_lower_digits);
  wchar_t *bp = buflim;
  const struct base_table_t *brec = &_itoa_base_table[base - 2];

  switch (base)
    {
# define RUN_2N(BITS) \
      do								      \
	{								      \
	  /* `unsigned long long int' always has 64 bits.  */		      \
	  mp_limb_t work_hi = value >> (64 - BITS_PER_MP_LIMB);		      \
									      \
	  if (BITS_PER_MP_LIMB == 32)					      \
	    {								      \
	      if (work_hi != 0)						      \
		{							      \
		  mp_limb_t work_lo;					      \
		  int cnt;						      \
									      \
		  work_lo = value & 0xfffffffful;			      \
		  for (cnt = BITS_PER_MP_LIMB / BITS; cnt > 0; --cnt)	      \
		    {							      \
		      *--bp = digits[work_lo & ((1ul << BITS) - 1)];	      \
		      work_lo >>= BITS;					      \
		    }							      \
		  if (BITS_PER_MP_LIMB % BITS != 0)			      \
		    {							      \
		      work_lo						      \
			|= ((work_hi					      \
			     & ((1 << (BITS - BITS_PER_MP_LIMB%BITS))	      \
				- 1))					      \
			    << BITS_PER_MP_LIMB % BITS);		      \
		      work_hi >>= BITS - BITS_PER_MP_LIMB % BITS;	      \
		      if (work_hi == 0)					      \
			work_hi = work_lo;				      \
		      else						      \
			*--bp = digits[work_lo];			      \
		    }							      \
		}							      \
	      else							      \
		work_hi = value & 0xfffffffful;				      \
	    }								      \
	  do								      \
	    {								      \
	      *--bp = digits[work_hi & ((1 << BITS) - 1)];		      \
	      work_hi >>= BITS;						      \
	    }								      \
	  while (work_hi != 0);						      \
	}								      \
      while (0)
    case 8:
      RUN_2N (3);
      break;

    case 16:
      RUN_2N (4);
      break;

    default:
      {
# if BITS_PER_MP_LIMB == 64
	mp_limb_t base_multiplier = brec->base_multiplier;
	if (brec->flag)
	  while (value != 0)
	    {
	      mp_limb_t quo, rem, x;
	      mp_limb_t dummy __attribute__ ((unused));

	      umul_ppmm (x, dummy, value, base_multiplier);
	      quo = (x + ((value - x) >> 1)) >> (brec->post_shift - 1);
	      rem = value - quo * base;
	      *--bp = digits[rem];
	      value = quo;
	    }
	else
	  while (value != 0)
	    {
	      mp_limb_t quo, rem, x;
	      mp_limb_t dummy __attribute__ ((unused));

	      umul_ppmm (x, dummy, value, base_multiplier);
	      quo = x >> brec->post_shift;
	      rem = value - quo * base;
	      *--bp = digits[rem];
	      value = quo;
	    }
# endif
# if BITS_PER_MP_LIMB == 32
	mp_limb_t t[3];
	int n;

	/* First convert x0 to 1-3 words in base s->big.base.
	   Optimize for frequent cases of 32 bit numbers.  */
	if ((mp_limb_t) (value >> 32) >= 1)
	  {
# if UDIV_TIME > 2 * UMUL_TIME || UDIV_NEEDS_NORMALIZATION
	    int big_normalization_steps = brec->big.normalization_steps;
	    mp_limb_t big_base_norm
	      = brec->big.base << big_normalization_steps;
# endif
	    if ((mp_limb_t) (value >> 32) >= brec->big.base)
	      {
		mp_limb_t x1hi, x1lo, r;
		/* If you want to optimize this, take advantage of
		   that the quotient in the first udiv_qrnnd will
		   always be very small.  It might be faster just to
		   subtract in a tight loop.  */

# if UDIV_TIME > 2 * UMUL_TIME
		mp_limb_t x, xh, xl;

		if (big_normalization_steps == 0)
		  xh = 0;
		else
		  xh = (mp_limb_t) (value >> (64 - big_normalization_steps));
		xl = (mp_limb_t) (value >> (32 - big_normalization_steps));
		udiv_qrnnd_preinv (x1hi, r, xh, xl, big_base_norm,
				   brec->big.base_ninv);

		xl = ((mp_limb_t) value) << big_normalization_steps;
		udiv_qrnnd_preinv (x1lo, x, r, xl, big_base_norm,
				   brec->big.base_ninv);
		t[2] = x >> big_normalization_steps;

		if (big_normalization_steps == 0)
		  xh = x1hi;
		else
		  xh = ((x1hi << big_normalization_steps)
			| (x1lo >> (32 - big_normalization_steps)));
		xl = x1lo << big_normalization_steps;
		udiv_qrnnd_preinv (t[0], x, xh, xl, big_base_norm,
				   brec->big.base_ninv);
		t[1] = x >> big_normalization_steps;
# elif UDIV_NEEDS_NORMALIZATION
		mp_limb_t x, xh, xl;

		if (big_normalization_steps == 0)
		  xh = 0;
		else
		  xh = (mp_limb_t) (value >> 64 - big_normalization_steps);
		xl = (mp_limb_t) (value >> 32 - big_normalization_steps);
		udiv_qrnnd (x1hi, r, xh, xl, big_base_norm);

		xl = ((mp_limb_t) value) << big_normalization_steps;
		udiv_qrnnd (x1lo, x, r, xl, big_base_norm);
		t[2] = x >> big_normalization_steps;

		if (big_normalization_steps == 0)
		  xh = x1hi;
		else
		  xh = ((x1hi << big_normalization_steps)
			| (x1lo >> 32 - big_normalization_steps));
		xl = x1lo << big_normalization_steps;
		udiv_qrnnd (t[0], x, xh, xl, big_base_norm);
		t[1] = x >> big_normalization_steps;
# else
		udiv_qrnnd (x1hi, r, 0, (mp_limb_t) (value >> 32),
			    brec->big.base);
		udiv_qrnnd (x1lo, t[2], r, (mp_limb_t) value, brec->big.base);
		udiv_qrnnd (t[0], t[1], x1hi, x1lo, brec->big.base);
# endif
		n = 3;
	      }
	    else
	      {
# if UDIV_TIME > 2 * UMUL_TIME
		mp_limb_t x;

		value <<= brec->big.normalization_steps;
		udiv_qrnnd_preinv (t[0], x, (mp_limb_t) (value >> 32),
				   (mp_limb_t) value, big_base_norm,
				   brec->big.base_ninv);
		t[1] = x >> brec->big.normalization_steps;
# elif UDIV_NEEDS_NORMALIZATION
		mp_limb_t x;

		value <<= big_normalization_steps;
		udiv_qrnnd (t[0], x, (mp_limb_t) (value >> 32),
			    (mp_limb_t) value, big_base_norm);
		t[1] = x >> big_normalization_steps;
# else
		udiv_qrnnd (t[0], t[1], (mp_limb_t) (value >> 32),
			    (mp_limb_t) value, brec->big.base);
# endif
		n = 2;
	      }
	  }
	else
	  {
	    t[0] = value;
	    n = 1;
	  }

	/* Convert the 1-3 words in t[], word by word, to ASCII.  */
	do
	  {
	    mp_limb_t ti = t[--n];
	    int ndig_for_this_limb = 0;

# if UDIV_TIME > 2 * UMUL_TIME
	    mp_limb_t base_multiplier = brec->base_multiplier;
	    if (brec->flag)
	      while (ti != 0)
		{
		  mp_limb_t quo, rem, x;
		  mp_limb_t dummy __attribute__ ((unused));

		  umul_ppmm (x, dummy, ti, base_multiplier);
		  quo = (x + ((ti - x) >> 1)) >> (brec->post_shift - 1);
		  rem = ti - quo * base;
		  *--bp = digits[rem];
		  ti = quo;
		  ++ndig_for_this_limb;
		}
	    else
	      while (ti != 0)
		{
		  mp_limb_t quo, rem, x;
		  mp_limb_t dummy __attribute__ ((unused));

		  umul_ppmm (x, dummy, ti, base_multiplier);
		  quo = x >> brec->post_shift;
		  rem = ti - quo * base;
		  *--bp = digits[rem];
		  ti = quo;
		  ++ndig_for_this_limb;
		}
# else
	    while (ti != 0)
	      {
		mp_limb_t quo, rem;

		quo = ti / base;
		rem = ti % base;
		*--bp = digits[rem];
		ti = quo;
		++ndig_for_this_limb;
	      }
# endif
	    /* If this wasn't the most significant word, pad with zeros.  */
	    if (n != 0)
	      while (ndig_for_this_limb < brec->big.ndigits)
		{
		  *--bp = '0';
		  ++ndig_for_this_limb;
		}
	  }
	while (n != 0);
# endif
      }
      break;
    }

  return bp;
}
#endif
