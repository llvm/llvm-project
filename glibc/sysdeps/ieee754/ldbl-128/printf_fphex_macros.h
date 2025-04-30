/* Macro to print floating point numbers in hexadecimal notation.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#define PRINT_FPHEX(FLOAT, VAR, IEEE854_UNION, IEEE854_BIAS)		      \
do {									      \
      /* We have 112 bits of mantissa plus one implicit digit.  Since	      \
	 112 bits are representable without rest using hexadecimal	      \
	 digits we use only the implicit digits for the number before	      \
	 the decimal point.  */						      \
      unsigned long long int num0, num1;				      \
      union IEEE854_UNION u;						      \
      u.d = VAR;							      \
									      \
      assert (sizeof (FLOAT) == 16);					      \
									      \
      num0 = (((unsigned long long int) u.ieee.mantissa0) << 32		      \
	     | u.ieee.mantissa1);					      \
      num1 = (((unsigned long long int) u.ieee.mantissa2) << 32		      \
	     | u.ieee.mantissa3);					      \
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
      leading = u.ieee.exponent == 0 ? '0' : '1';			      \
									      \
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
	      exponent = IEEE854_BIAS - 1;				      \
	    }								      \
	}								      \
      else if (exponent >= IEEE854_BIAS)				      \
	{								      \
	  expnegative = 0;						      \
	  exponent -= IEEE854_BIAS;					      \
	}								      \
      else								      \
	{								      \
	  expnegative = 1;						      \
	  exponent = -(exponent - IEEE854_BIAS);			      \
	}								      \
} while (0)
