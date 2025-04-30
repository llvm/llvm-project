/* Print floating point number in hexadecimal notation according to ISO C99.
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

#define PRINT_FPHEX_LONG_DOUBLE \
do {									      \
      /* We have 105 bits of mantissa plus one implicit digit.  Since	      \
	 106 bits are representable without rest using hexadecimal	      \
	 digits we use only the implicit digits for the number before	      \
	 the decimal point.  */						      \
      unsigned long long int num0, num1;				      \
      unsigned long long hi, lo;					      \
      int ediff;							      \
      union ibm_extended_long_double u;					      \
      u.ld = fpnum.ldbl;						      \
									      \
      assert (sizeof (long double) == 16);				      \
									      \
      lo = ((long long)u.d[1].ieee.mantissa0 << 32) | u.d[1].ieee.mantissa1;  \
      hi = ((long long)u.d[0].ieee.mantissa0 << 32) | u.d[0].ieee.mantissa1;  \
      lo <<= 7; /* pre-shift lo to match ieee854.  */			      \
      /* If the lower double is not a denormal or zero then set the hidden    \
	 53rd bit.  */							      \
      if (u.d[1].ieee.exponent != 0)					      \
	lo |= (1ULL << (52 + 7));					      \
      else								      \
	lo <<= 1;							      \
      /* The lower double is normalized separately from the upper.  We	      \
	 may need to adjust the lower manitissa to reflect this.  */	      \
      ediff = u.d[0].ieee.exponent - u.d[1].ieee.exponent - 53;		      \
      if (ediff > 63)							      \
	lo = 0;								      \
      else if (ediff > 0)						      \
	lo = lo >> ediff;						      \
      else if (ediff < 0)						      \
	lo = lo << -ediff;						      \
      if (u.d[0].ieee.negative != u.d[1].ieee.negative			      \
	  && lo != 0)							      \
	{								      \
	  lo = (1ULL << 60) - lo;					      \
	  if (hi == 0L)							      \
	    {								      \
	      /* we have a borrow from the hidden bit, so shift left 1.  */   \
	      hi = 0xffffffffffffeLL | (lo >> 59);			      \
	      lo = 0xfffffffffffffffLL & (lo << 1);			      \
	      u.d[0].ieee.exponent--;					      \
	    }								      \
	  else								      \
	    hi--;							      \
        }								      \
      num1 = (hi << 60) | lo;						      \
      num0 = hi >> 4;							      \
									      \
      zero_mantissa = (num0|num1) == 0;					      \
									      \
      if (sizeof (unsigned long int) > 6)				      \
	{								      \
	  numstr = _itoa_word (num1, numbuf + sizeof numbuf, 16,	      \
			       info->spec == 'A');			      \
	  wnumstr = _itowa_word (num1,					      \
				 wnumbuf + sizeof (wnumbuf) / sizeof (wchar_t),\
				 16, info->spec == 'A');		      \
	}								      \
      else								      \
	{								      \
	  numstr = _itoa (num1, numbuf + sizeof numbuf, 16,		      \
			  info->spec == 'A');				      \
	  wnumstr = _itowa (num1,					      \
			    wnumbuf + sizeof (wnumbuf) / sizeof (wchar_t),    \
			    16, info->spec == 'A');			      \
	}								      \
									      \
      while (numstr > numbuf + (sizeof numbuf - 64 / 4))		      \
	{								      \
	  *--numstr = '0';						      \
	  *--wnumstr = L'0';						      \
	}								      \
									      \
      if (sizeof (unsigned long int) > 6)				      \
	{								      \
	  numstr = _itoa_word (num0, numstr, 16, info->spec == 'A');	      \
	  wnumstr = _itowa_word (num0, wnumstr, 16, info->spec == 'A');	      \
	}								      \
      else								      \
	{								      \
	  numstr = _itoa (num0, numstr, 16, info->spec == 'A');		      \
	  wnumstr = _itowa (num0, wnumstr, 16, info->spec == 'A');	      \
	}								      \
									      \
      /* Fill with zeroes.  */						      \
      while (numstr > numbuf + (sizeof numbuf - 112 / 4))		      \
	{								      \
	  *--numstr = '0';						      \
	  *--wnumstr = L'0';						      \
	}								      \
									      \
      leading = u.d[0].ieee.exponent == 0 ? '0' : '1';			      \
									      \
      exponent = u.d[0].ieee.exponent;					      \
									      \
      if (exponent == 0)						      \
	{								      \
	  if (zero_mantissa)						      \
	    expnegative = 0;						      \
	  else								      \
	    {								      \
	      /* This is a denormalized number.  */			      \
	      expnegative = 1;						      \
	      exponent = IEEE754_DOUBLE_BIAS - 1;			      \
	    }								      \
	}								      \
      else if (exponent >= IEEE754_DOUBLE_BIAS)				      \
	{								      \
	  expnegative = 0;						      \
	  exponent -= IEEE754_DOUBLE_BIAS;				      \
	}								      \
      else								      \
	{								      \
	  expnegative = 1;						      \
	  exponent = -(exponent - IEEE754_DOUBLE_BIAS);			      \
	}								      \
} while (0)

#include <stdio-common/printf_fphex.c>
