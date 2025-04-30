/* Print floating point number in hexadecimal notation according to ISO C99.
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

#ifndef LONG_DOUBLE_DENORM_BIAS
# define LONG_DOUBLE_DENORM_BIAS (IEEE854_LONG_DOUBLE_BIAS - 1)
#endif

#define PRINT_FPHEX_LONG_DOUBLE \
do {									      \
      /* The "strange" 80 bit format on ix86 and m68k has an explicit	      \
	 leading digit in the 64 bit mantissa.  */			      \
      unsigned long long int num;					      \
      union ieee854_long_double u;					      \
      u.d = fpnum.ldbl;							      \
									      \
      num = (((unsigned long long int) u.ieee.mantissa0) << 32		      \
	     | u.ieee.mantissa1);					      \
									      \
      zero_mantissa = num == 0;						      \
									      \
      if (sizeof (unsigned long int) > 6)				      \
	{								      \
	  numstr = _itoa_word (num, numbuf + sizeof numbuf, 16,		      \
			       info->spec == 'A');			      \
	  wnumstr = _itowa_word (num,					      \
				 wnumbuf + sizeof (wnumbuf) / sizeof (wchar_t),\
				 16, info->spec == 'A');		      \
	}								      \
      else								      \
	{								      \
	  numstr = _itoa (num, numbuf + sizeof numbuf, 16, info->spec == 'A');\
	  wnumstr = _itowa (num,					      \
			    wnumbuf + sizeof (wnumbuf) / sizeof (wchar_t),    \
			    16, info->spec == 'A');			      \
	}								      \
									      \
      /* Fill with zeroes.  */						      \
      while (numstr > numbuf + (sizeof numbuf - 64 / 4))		      \
	{								      \
	  *--numstr = '0';						      \
	  *--wnumstr = L'0';						      \
	}								      \
									      \
      /* We use a full nibble for the leading digit.  */		      \
      leading = *numstr++;						      \
      wnumstr++;							      \
									      \
      /* We have 3 bits from the mantissa in the leading nibble.	      \
	 Therefore we are here using `IEEE854_LONG_DOUBLE_BIAS + 3'.  */      \
      exponent = u.ieee.exponent;					      \
									      \
      if (exponent == 0)						      \
	{								      \
	  if (zero_mantissa)						      \
	    expnegative = 0;						      \
	  else								      \
	    {								      \
	      /* This is a denormalized number.  */			      \
	      expnegative = 1;						      \
	      /* This is a hook for the m68k long double format, where the    \
		 exponent bias is the same for normalized and denormalized    \
		 numbers.  */						      \
	      exponent = LONG_DOUBLE_DENORM_BIAS + 3;			      \
	    }								      \
	}								      \
      else if (exponent >= IEEE854_LONG_DOUBLE_BIAS + 3)		      \
	{								      \
	  expnegative = 0;						      \
	  exponent -= IEEE854_LONG_DOUBLE_BIAS + 3;			      \
	}								      \
      else								      \
	{								      \
	  expnegative = 1;						      \
	  exponent = -(exponent - (IEEE854_LONG_DOUBLE_BIAS + 3));	      \
	}								      \
} while (0)

#include <stdio-common/printf_fphex.c>
