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

#include <array_length.h>
#include <ctype.h>
#include <ieee754.h>
#include <math.h>
#include <printf.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <_itoa.h>
#include <_itowa.h>
#include <locale/localeinfo.h>
#include <stdbool.h>
#include <rounding-mode.h>

#if __HAVE_DISTINCT_FLOAT128
# include "ieee754_float128.h"
# include <ldbl-128/printf_fphex_macros.h>
# define PRINT_FPHEX_FLOAT128 \
   PRINT_FPHEX (_Float128, fpnum.flt128, ieee854_float128, \
		IEEE854_FLOAT128_BIAS)
#endif

/* #define NDEBUG 1*/		/* Undefine this for debugging assertions.  */
#include <assert.h>

#include <libioP.h>
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
	return -1;							      \
      ++done;								      \
    } while (0)

#define PRINT(ptr, wptr, len)						      \
  do									      \
    {									      \
      size_t outlen = (len);						      \
      if (wide)								      \
	while (outlen-- > 0)						      \
	  outchar (*wptr++);						      \
      else								      \
	while (outlen-- > 0)						      \
	  outchar (*ptr++);						      \
    } while (0)

#define PADN(ch, len)							      \
  do									      \
    {									      \
      if (PAD (fp, ch, len) != len)					      \
	return -1;							      \
      done += len;							      \
    }									      \
  while (0)

#ifndef MIN
# define MIN(a,b) ((a)<(b)?(a):(b))
#endif


int
__printf_fphex (FILE *fp,
		const struct printf_info *info,
		const void *const *args)
{
  /* The floating-point value to output.  */
  union
    {
      union ieee754_double dbl;
      long double ldbl;
#if __HAVE_DISTINCT_FLOAT128
      _Float128 flt128;
#endif
    }
  fpnum;

  /* Locale-dependent representation of decimal point.	*/
  const char *decimal;
  wchar_t decimalwc;

  /* "NaN" or "Inf" for the special cases.  */
  const char *special = NULL;
  const wchar_t *wspecial = NULL;

  /* Buffer for the generated number string for the mantissa.  The
     maximal size for the mantissa is 128 bits.  */
  char numbuf[32];
  char *numstr;
  char *numend;
  wchar_t wnumbuf[32];
  wchar_t *wnumstr;
  wchar_t *wnumend;
  int negative;

  /* The maximal exponent of two in decimal notation has 5 digits.  */
  char expbuf[5];
  char *expstr;
  wchar_t wexpbuf[5];
  wchar_t *wexpstr;
  int expnegative;
  int exponent;

  /* Non-zero is mantissa is zero.  */
  int zero_mantissa;

  /* The leading digit before the decimal point.  */
  char leading;

  /* Precision.  */
  int precision = info->prec;

  /* Width.  */
  int width = info->width;

  /* Number of characters written.  */
  int done = 0;

  /* Nonzero if this is output on a wide character stream.  */
  int wide = info->wide;


  /* Figure out the decimal point character.  */
  if (info->extra == 0)
    {
      decimal = _NL_CURRENT (LC_NUMERIC, DECIMAL_POINT);
      decimalwc = _NL_CURRENT_WORD (LC_NUMERIC, _NL_NUMERIC_DECIMAL_POINT_WC);
    }
  else
    {
      decimal = _NL_CURRENT (LC_MONETARY, MON_DECIMAL_POINT);
      decimalwc = _NL_CURRENT_WORD (LC_MONETARY,
				    _NL_MONETARY_DECIMAL_POINT_WC);
    }
  /* The decimal point character must never be zero.  */
  assert (*decimal != '\0' && decimalwc != L'\0');

#define PRINTF_FPHEX_FETCH(FLOAT, VAR)					\
  {									\
    (VAR) = *(const FLOAT *) args[0];					\
									\
    /* Check for special values: not a number or infinity.  */		\
    if (isnan (VAR))							\
      {									\
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
    else								\
      {									\
	if (isinf (VAR))						\
	  {								\
	    if (isupper (info->spec))					\
	      {								\
		special = "INF";					\
		wspecial = L"INF";					\
	      }								\
	    else							\
	      {								\
		special = "inf";					\
		wspecial = L"inf";					\
	      }								\
	  }								\
      }									\
    negative = signbit (VAR);						\
  }

  /* Fetch the argument value.	*/
#if __HAVE_DISTINCT_FLOAT128
  if (info->is_binary128)
    PRINTF_FPHEX_FETCH (_Float128, fpnum.flt128)
  else
#endif
#ifndef __NO_LONG_DOUBLE_MATH
  if (info->is_long_double && sizeof (long double) > sizeof (double))
    PRINTF_FPHEX_FETCH (long double, fpnum.ldbl)
  else
#endif
    PRINTF_FPHEX_FETCH (double, fpnum.dbl.d)

#undef PRINTF_FPHEX_FETCH

  if (special)
    {
      int width = info->width;

      if (negative || info->showsign || info->space)
	--width;
      width -= 3;

      if (!info->left && width > 0)
	PADN (' ', width);

      if (negative)
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

#if __HAVE_DISTINCT_FLOAT128
  if (info->is_binary128)
    PRINT_FPHEX_FLOAT128;
  else
#endif
  if (info->is_long_double == 0 || sizeof (double) == sizeof (long double))
    {
      /* We have 52 bits of mantissa plus one implicit digit.  Since
	 52 bits are representable without rest using hexadecimal
	 digits we use only the implicit digits for the number before
	 the decimal point.  */
      unsigned long long int num;

      num = (((unsigned long long int) fpnum.dbl.ieee.mantissa0) << 32
	     | fpnum.dbl.ieee.mantissa1);

      zero_mantissa = num == 0;

      if (sizeof (unsigned long int) > 6)
	{
	  wnumstr = _itowa_word (num, wnumbuf + (sizeof wnumbuf) / sizeof (wchar_t), 16,
				 info->spec == 'A');
	  numstr = _itoa_word (num, numbuf + sizeof numbuf, 16,
			       info->spec == 'A');
	}
      else
	{
	  wnumstr = _itowa (num, wnumbuf + sizeof wnumbuf / sizeof (wchar_t), 16,
			    info->spec == 'A');
	  numstr = _itoa (num, numbuf + sizeof numbuf, 16,
			  info->spec == 'A');
	}

      /* Fill with zeroes.  */
      while (wnumstr > wnumbuf + (sizeof wnumbuf - 52) / sizeof (wchar_t))
	{
	  *--wnumstr = L'0';
	  *--numstr = '0';
	}

      leading = fpnum.dbl.ieee.exponent == 0 ? '0' : '1';

      exponent = fpnum.dbl.ieee.exponent;

      if (exponent == 0)
	{
	  if (zero_mantissa)
	    expnegative = 0;
	  else
	    {
	      /* This is a denormalized number.  */
	      expnegative = 1;
	      exponent = IEEE754_DOUBLE_BIAS - 1;
	    }
	}
      else if (exponent >= IEEE754_DOUBLE_BIAS)
	{
	  expnegative = 0;
	  exponent -= IEEE754_DOUBLE_BIAS;
	}
      else
	{
	  expnegative = 1;
	  exponent = -(exponent - IEEE754_DOUBLE_BIAS);
	}
    }
#ifdef PRINT_FPHEX_LONG_DOUBLE
  else
    PRINT_FPHEX_LONG_DOUBLE;
#endif

  /* Look for trailing zeroes.  */
  if (! zero_mantissa)
    {
      wnumend = array_end (wnumbuf);
      numend = array_end (numbuf);
      while (wnumend[-1] == L'0')
	{
	  --wnumend;
	  --numend;
	}

      bool do_round_away = false;

      if (precision != -1 && precision < numend - numstr)
	{
	  char last_digit = precision > 0 ? numstr[precision - 1] : leading;
	  char next_digit = numstr[precision];
	  int last_digit_value = (last_digit >= 'A' && last_digit <= 'F'
				  ? last_digit - 'A' + 10
				  : (last_digit >= 'a' && last_digit <= 'f'
				     ? last_digit - 'a' + 10
				     : last_digit - '0'));
	  int next_digit_value = (next_digit >= 'A' && next_digit <= 'F'
				  ? next_digit - 'A' + 10
				  : (next_digit >= 'a' && next_digit <= 'f'
				     ? next_digit - 'a' + 10
				     : next_digit - '0'));
	  bool more_bits = ((next_digit_value & 7) != 0
			    || precision + 1 < numend - numstr);
	  int rounding_mode = get_rounding_mode ();
	  do_round_away = round_away (negative, last_digit_value & 1,
				      next_digit_value >= 8, more_bits,
				      rounding_mode);
	}

      if (precision == -1)
	precision = numend - numstr;
      else if (do_round_away)
	{
	  /* Round up.  */
	  int cnt = precision;
	  while (--cnt >= 0)
	    {
	      char ch = numstr[cnt];
	      /* We assume that the digits and the letters are ordered
		 like in ASCII.  This is true for the rest of GNU, too.  */
	      if (ch == '9')
		{
		  wnumstr[cnt] = (wchar_t) info->spec;
		  numstr[cnt] = info->spec;	/* This is tricky,
						   think about it!  */
		  break;
		}
	      else if (tolower (ch) < 'f')
		{
		  ++numstr[cnt];
		  ++wnumstr[cnt];
		  break;
		}
	      else
		{
		  numstr[cnt] = '0';
		  wnumstr[cnt] = L'0';
		}
	    }
	  if (cnt < 0)
	    {
	      /* The mantissa so far was fff...f  Now increment the
		 leading digit.  Here it is again possible that we
		 get an overflow.  */
	      if (leading == '9')
		leading = info->spec;
	      else if (tolower (leading) < 'f')
		++leading;
	      else
		{
		  leading = '1';
		  if (expnegative)
		    {
		      exponent -= 4;
		      if (exponent <= 0)
			{
			  exponent = -exponent;
			  expnegative = 0;
			}
		    }
		  else
		    exponent += 4;
		}
	    }
	}
    }
  else
    {
      if (precision == -1)
	precision = 0;
      numend = numstr;
      wnumend = wnumstr;
    }

  /* Now we can compute the exponent string.  */
  expstr = _itoa_word (exponent, expbuf + sizeof expbuf, 10, 0);
  wexpstr = _itowa_word (exponent,
			 wexpbuf + sizeof wexpbuf / sizeof (wchar_t), 10, 0);

  /* Now we have all information to compute the size.  */
  width -= ((negative || info->showsign || info->space)
	    /* Sign.  */
	    + 2    + 1 + 0 + precision + 1 + 1
	    /* 0x    h   .   hhh         P   ExpoSign.  */
	    + ((expbuf + sizeof expbuf) - expstr));
	    /* Exponent.  */

  /* Count the decimal point.
     A special case when the mantissa or the precision is zero and the `#'
     is not given.  In this case we must not print the decimal point.  */
  if (precision > 0 || info->alt)
    width -= wide ? 1 : strlen (decimal);

  if (!info->left && info->pad != '0' && width > 0)
    PADN (' ', width);

  if (negative)
    outchar ('-');
  else if (info->showsign)
    outchar ('+');
  else if (info->space)
    outchar (' ');

  outchar ('0');
  if ('X' - 'A' == 'x' - 'a')
    outchar (info->spec + ('x' - 'a'));
  else
    outchar (info->spec == 'A' ? 'X' : 'x');

  if (!info->left && info->pad == '0' && width > 0)
    PADN ('0', width);

  outchar (leading);

  if (precision > 0 || info->alt)
    {
      const wchar_t *wtmp = &decimalwc;
      PRINT (decimal, wtmp, wide ? 1 : strlen (decimal));
    }

  if (precision > 0)
    {
      ssize_t tofill = precision - (numend - numstr);
      PRINT (numstr, wnumstr, MIN (numend - numstr, precision));
      if (tofill > 0)
	PADN ('0', tofill);
    }

  if ('P' - 'A' == 'p' - 'a')
    outchar (info->spec + ('p' - 'a'));
  else
    outchar (info->spec == 'A' ? 'P' : 'p');

  outchar (expnegative ? '-' : '+');

  PRINT (expstr, wexpstr, (expbuf + sizeof expbuf) - expstr);

  if (info->left && info->pad != '0' && width > 0)
    PADN (info->pad, width);

  return done;
}
