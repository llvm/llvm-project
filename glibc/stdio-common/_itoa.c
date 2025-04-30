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

#include <_itoa.h>


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


/* We do not compile _itoa if we always can use _itoa_word.  */
#if _ITOA_NEEDED
/* Local variables.  */
const struct base_table_t _itoa_base_table[] attribute_hidden =
{
# if BITS_PER_MP_LIMB == 64
  /*  2 */ {SEL1(0ull) 1, 1},
  /*  3 */ {SEL1(0xaaaaaaaaaaaaaaabull) 0, 1},
  /*  4 */ {SEL1(0ull) 1, 2},
  /*  5 */ {SEL1(0xcccccccccccccccdull) 0, 2},
  /*  6 */ {SEL1(0xaaaaaaaaaaaaaaabull) 0, 2},
  /*  7 */ {SEL1(0x2492492492492493ull) 1, 3},
  /*  8 */ {SEL1(0ull) 1, 3},
  /*  9 */ {SEL1(0xe38e38e38e38e38full) 0, 3},
  /* 10 */ {SEL1(0xcccccccccccccccdull) 0, 3},
  /* 11 */ {SEL1(0x2e8ba2e8ba2e8ba3ull) 0, 1},
  /* 12 */ {SEL1(0xaaaaaaaaaaaaaaabull) 0, 3},
  /* 13 */ {SEL1(0x4ec4ec4ec4ec4ec5ull) 0, 2},
  /* 14 */ {SEL1(0x2492492492492493ull) 1, 4},
  /* 15 */ {SEL1(0x8888888888888889ull) 0, 3},
  /* 16 */ {SEL1(0ull) 1, 4},
  /* 17 */ {SEL1(0xf0f0f0f0f0f0f0f1ull) 0, 4},
  /* 18 */ {SEL1(0xe38e38e38e38e38full) 0, 4},
  /* 19 */ {SEL1(0xd79435e50d79435full) 0, 4},
  /* 20 */ {SEL1(0xcccccccccccccccdull) 0, 4},
  /* 21 */ {SEL1(0x8618618618618619ull) 1, 5},
  /* 22 */ {SEL1(0x2e8ba2e8ba2e8ba3ull) 0, 2},
  /* 23 */ {SEL1(0x642c8590b21642c9ull) 1, 5},
  /* 24 */ {SEL1(0xaaaaaaaaaaaaaaabull) 0, 4},
  /* 25 */ {SEL1(0x47ae147ae147ae15ull) 1, 5},
  /* 26 */ {SEL1(0x4ec4ec4ec4ec4ec5ull) 0, 3},
  /* 27 */ {SEL1(0x97b425ed097b425full) 0, 4},
  /* 28 */ {SEL1(0x2492492492492493ull) 1, 5},
  /* 29 */ {SEL1(0x1a7b9611a7b9611bull) 1, 5},
  /* 30 */ {SEL1(0x8888888888888889ull) 0, 4},
  /* 31 */ {SEL1(0x0842108421084211ull) 1, 5},
  /* 32 */ {SEL1(0ull) 1, 5},
  /* 33 */ {SEL1(0x0f83e0f83e0f83e1ull) 0, 1},
  /* 34 */ {SEL1(0xf0f0f0f0f0f0f0f1ull) 0, 5},
  /* 35 */ {SEL1(0xea0ea0ea0ea0ea0full) 0, 5},
  /* 36 */ {SEL1(0xe38e38e38e38e38full) 0, 5}
# endif
# if BITS_PER_MP_LIMB == 32
  /*  2 */ {SEL1(0ul) 1, 1, {0, 31, 0x80000000ul SEL2(0xfffffffful)}},
  /*  3 */ {SEL1(0xaaaaaaabul) 0, 1, {0, 20, 0xcfd41b91ul SEL2(0x3b563c24ul)}},
  /*  4 */ {SEL1(0ul) 1, 2, {1, 15, 0x40000000ul SEL2(0xfffffffful)}},
  /*  5 */ {SEL1(0xcccccccdul) 0, 2, {1, 13, 0x48c27395ul SEL2(0xc25c2684ul)}},
  /*  6 */ {SEL1(0xaaaaaaabul) 0, 2, {0, 12, 0x81bf1000ul SEL2(0xf91bd1b6ul)}},
  /*  7 */ {SEL1(0x24924925ul) 1, 3, {1, 11, 0x75db9c97ul SEL2(0x1607a2cbul)}},
  /*  8 */ {SEL1(0ul) 1, 3, {1, 10, 0x40000000ul SEL2(0xfffffffful)}},
  /*  9 */ {SEL1(0x38e38e39ul) 0, 1, {0, 10, 0xcfd41b91ul SEL2(0x3b563c24ul)}},
  /* 10 */ {SEL1(0xcccccccdul) 0, 3, {2, 9, 0x3b9aca00ul SEL2(0x12e0be82ul)}},
  /* 11 */ {SEL1(0xba2e8ba3ul) 0, 3, {0, 9, 0x8c8b6d2bul SEL2(0xd24cde04ul)}},
  /* 12 */ {SEL1(0xaaaaaaabul) 0, 3, {3, 8, 0x19a10000ul SEL2(0x3fa39ab5ul)}},
  /* 13 */ {SEL1(0x4ec4ec4ful) 0, 2, {2, 8, 0x309f1021ul SEL2(0x50f8ac5ful)}},
  /* 14 */ {SEL1(0x24924925ul) 1, 4, {1, 8, 0x57f6c100ul SEL2(0x74843b1eul)}},
  /* 15 */ {SEL1(0x88888889ul) 0, 3, {0, 8, 0x98c29b81ul SEL2(0xad0326c2ul)}},
  /* 16 */ {SEL1(0ul) 1, 4, {3, 7, 0x10000000ul SEL2(0xfffffffful)}},
  /* 17 */ {SEL1(0xf0f0f0f1ul) 0, 4, {3, 7, 0x18754571ul SEL2(0x4ef0b6bdul)}},
  /* 18 */ {SEL1(0x38e38e39ul) 0, 2, {2, 7, 0x247dbc80ul SEL2(0xc0fc48a1ul)}},
  /* 19 */ {SEL1(0xaf286bcbul) 1, 5, {2, 7, 0x3547667bul SEL2(0x33838942ul)}},
  /* 20 */ {SEL1(0xcccccccdul) 0, 4, {1, 7, 0x4c4b4000ul SEL2(0xad7f29abul)}},
  /* 21 */ {SEL1(0x86186187ul) 1, 5, {1, 7, 0x6b5a6e1dul SEL2(0x313c3d15ul)}},
  /* 22 */ {SEL1(0xba2e8ba3ul) 0, 4, {0, 7, 0x94ace180ul SEL2(0xb8cca9e0ul)}},
  /* 23 */ {SEL1(0xb21642c9ul) 0, 4, {0, 7, 0xcaf18367ul SEL2(0x42ed6de9ul)}},
  /* 24 */ {SEL1(0xaaaaaaabul) 0, 4, {4, 6, 0x0b640000ul SEL2(0x67980e0bul)}},
  /* 25 */ {SEL1(0x51eb851ful) 0, 3, {4, 6, 0x0e8d4a51ul SEL2(0x19799812ul)}},
  /* 26 */ {SEL1(0x4ec4ec4ful) 0, 3, {3, 6, 0x1269ae40ul SEL2(0xbce85396ul)}},
  /* 27 */ {SEL1(0x2f684bdbul) 1, 5, {3, 6, 0x17179149ul SEL2(0x62c103a9ul)}},
  /* 28 */ {SEL1(0x24924925ul) 1, 5, {3, 6, 0x1cb91000ul SEL2(0x1d353d43ul)}},
  /* 29 */ {SEL1(0x8d3dcb09ul) 0, 4, {2, 6, 0x23744899ul SEL2(0xce1deceaul)}},
  /* 30 */ {SEL1(0x88888889ul) 0, 4, {2, 6, 0x2b73a840ul SEL2(0x790fc511ul)}},
  /* 31 */ {SEL1(0x08421085ul) 1, 5, {2, 6, 0x34e63b41ul SEL2(0x35b865a0ul)}},
  /* 32 */ {SEL1(0ul) 1, 5, {1, 6, 0x40000000ul SEL2(0xfffffffful)}},
  /* 33 */ {SEL1(0x3e0f83e1ul) 0, 3, {1, 6, 0x4cfa3cc1ul SEL2(0xa9aed1b3ul)}},
  /* 34 */ {SEL1(0xf0f0f0f1ul) 0, 5, {1, 6, 0x5c13d840ul SEL2(0x63dfc229ul)}},
  /* 35 */ {SEL1(0xd41d41d5ul) 1, 6, {1, 6, 0x6d91b519ul SEL2(0x2b0fee30ul)}},
  /* 36 */ {SEL1(0x38e38e39ul) 0, 3, {0, 6, 0x81bf1000ul SEL2(0xf91bd1b6ul)}}
# endif
};
#endif

char *
_itoa_word (_ITOA_WORD_TYPE value, char *buflim,
	    unsigned int base, int upper_case)
{
  const char *digits = (upper_case
			? _itoa_upper_digits
			: _itoa_lower_digits);

  switch (base)
    {
#define SPECIAL(Base)							      \
    case Base:								      \
      do								      \
	*--buflim = digits[value % Base];				      \
      while ((value /= Base) != 0);					      \
      break

      SPECIAL (10);
      SPECIAL (16);
      SPECIAL (8);
    default:
      do
	*--buflim = digits[value % base];
      while ((value /= base) != 0);
    }
  return buflim;
}
#undef SPECIAL


#if _ITOA_NEEDED
char *
_itoa (unsigned long long int value, char *buflim, unsigned int base,
       int upper_case)
{
  const char *digits = (upper_case
			? _itoa_upper_digits
			: _itoa_lower_digits);
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
		      *--buflim = digits[work_lo & ((1ul << BITS) - 1)];      \
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
			*--buflim = digits[work_lo];			      \
		    }							      \
		}							      \
	      else							      \
		work_hi = value & 0xfffffffful;				      \
	    }								      \
	  do								      \
	    {								      \
	      *--buflim = digits[work_hi & ((1 << BITS) - 1)];		      \
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
	char *bufend = buflim;
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
	      *--buflim = digits[rem];
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
	      *--buflim = digits[rem];
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
#  if UDIV_TIME > 2 * UMUL_TIME || UDIV_NEEDS_NORMALIZATION
	    int big_normalization_steps = brec->big.normalization_steps;
	    mp_limb_t big_base_norm
	      = brec->big.base << big_normalization_steps;
#  endif
	    if ((mp_limb_t) (value >> 32) >= brec->big.base)
	      {
		mp_limb_t x1hi, x1lo, r;
		/* If you want to optimize this, take advantage of
		   that the quotient in the first udiv_qrnnd will
		   always be very small.  It might be faster just to
		   subtract in a tight loop.  */

#  if UDIV_TIME > 2 * UMUL_TIME
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
#  elif UDIV_NEEDS_NORMALIZATION
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
#  else
		udiv_qrnnd (x1hi, r, 0, (mp_limb_t) (value >> 32),
			    brec->big.base);
		udiv_qrnnd (x1lo, t[2], r, (mp_limb_t) value, brec->big.base);
		udiv_qrnnd (t[0], t[1], x1hi, x1lo, brec->big.base);
#  endif
		n = 3;
	      }
	    else
	      {
#  if UDIV_TIME > 2 * UMUL_TIME
		mp_limb_t x;

		value <<= brec->big.normalization_steps;
		udiv_qrnnd_preinv (t[0], x, (mp_limb_t) (value >> 32),
				   (mp_limb_t) value, big_base_norm,
				   brec->big.base_ninv);
		t[1] = x >> brec->big.normalization_steps;
#  elif UDIV_NEEDS_NORMALIZATION
		mp_limb_t x;

		value <<= big_normalization_steps;
		udiv_qrnnd (t[0], x, (mp_limb_t) (value >> 32),
			    (mp_limb_t) value, big_base_norm);
		t[1] = x >> big_normalization_steps;
#  else
		udiv_qrnnd (t[0], t[1], (mp_limb_t) (value >> 32),
			    (mp_limb_t) value, brec->big.base);
#  endif
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

#  if UDIV_TIME > 2 * UMUL_TIME
	    mp_limb_t base_multiplier = brec->base_multiplier;
	    if (brec->flag)
	      while (ti != 0)
		{
		  mp_limb_t quo, rem, x;
		  mp_limb_t dummy __attribute__ ((unused));

		  umul_ppmm (x, dummy, ti, base_multiplier);
		  quo = (x + ((ti - x) >> 1)) >> (brec->post_shift - 1);
		  rem = ti - quo * base;
		  *--buflim = digits[rem];
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
		  *--buflim = digits[rem];
		  ti = quo;
		  ++ndig_for_this_limb;
		}
#  else
	    while (ti != 0)
	      {
		mp_limb_t quo, rem;

		quo = ti / base;
		rem = ti % base;
		*--buflim = digits[rem];
		ti = quo;
		++ndig_for_this_limb;
	      }
#  endif
	    /* If this wasn't the most significant word, pad with zeros.  */
	    if (n != 0)
	      while (ndig_for_this_limb < brec->big.ndigits)
		{
		  *--buflim = '0';
		  ++ndig_for_this_limb;
		}
	  }
	while (n != 0);
# endif
	if (buflim == bufend)
	  *--buflim = '0';
      }
      break;
    }

  return buflim;
}
#endif

char *
_fitoa_word (_ITOA_WORD_TYPE value, char *buf, unsigned int base,
	     int upper_case)
{
  char tmpbuf[sizeof (value) * 4];	      /* Worst case length: base 2.  */
  char *cp = _itoa_word (value, tmpbuf + sizeof (value) * 4, base, upper_case);
  while (cp < tmpbuf + sizeof (value) * 4)
    *buf++ = *cp++;
  return buf;
}

#if _ITOA_NEEDED
char *
_fitoa (unsigned long long value, char *buf, unsigned int base, int upper_case)
{
  char tmpbuf[sizeof (value) * 4];	      /* Worst case length: base 2.  */
  char *cp = _itoa (value, tmpbuf + sizeof (value) * 4, base, upper_case);
  while (cp < tmpbuf + sizeof (value) * 4)
    *buf++ = *cp++;
  return buf;
}
#endif
