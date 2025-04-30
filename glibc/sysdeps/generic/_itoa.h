/* Internal function for converting integers to ASCII.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#ifndef _ITOA_H
#define _ITOA_H

#include <limits.h>

/* When long long is different from long, by default, _itoa_word is
   provided to convert long to ASCII and _itoa is provided to convert
   long long.  A sysdeps _itoa.h can define _ITOA_NEEDED to 0 and define
   _ITOA_WORD_TYPE to unsigned long long int to override it so that
   _itoa_word is changed to convert long long to ASCII and _itoa is
   mapped to _itoa_word.  */

#ifndef _ITOA_NEEDED
# define _ITOA_NEEDED		(LONG_MAX != LLONG_MAX)
#endif
#ifndef _ITOA_WORD_TYPE
# define _ITOA_WORD_TYPE	unsigned long int
#endif


/* Convert VALUE into ASCII in base BASE (2..36).
   Write backwards starting the character just before BUFLIM.
   Return the address of the first (left-to-right) character in the number.
   Use upper case letters iff UPPER_CASE is nonzero.  */

extern char *_itoa (unsigned long long int value, char *buflim,
		    unsigned int base, int upper_case) attribute_hidden;

extern const char _itoa_upper_digits[];
extern const char _itoa_lower_digits[];
#if IS_IN (libc) || IS_IN (rtld)
hidden_proto (_itoa_upper_digits)
hidden_proto (_itoa_lower_digits)
#endif

#if IS_IN (libc)
extern char *_itoa_word (_ITOA_WORD_TYPE value, char *buflim,
			 unsigned int base,
			 int upper_case) attribute_hidden;
#else
static inline char * __attribute__ ((unused, always_inline))
_itoa_word (_ITOA_WORD_TYPE value, char *buflim,
	    unsigned int base, int upper_case)
{
  const char *digits = (upper_case
			? _itoa_upper_digits
			: _itoa_lower_digits);

  switch (base)
    {
# define SPECIAL(Base)							      \
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
# undef SPECIAL
#endif

/* Similar to the _itoa functions, but output starts at buf and pointer
   after the last written character is returned.  */
extern char *_fitoa_word (_ITOA_WORD_TYPE value, char *buf,
			  unsigned int base,
			  int upper_case) attribute_hidden;
extern char *_fitoa (unsigned long long value, char *buf, unsigned int base,
		     int upper_case) attribute_hidden;

#if !_ITOA_NEEDED
/* No need for special long long versions.  */
# define _itoa(value, buf, base, upper_case) \
  _itoa_word (value, buf, base, upper_case)
# define _fitoa(value, buf, base, upper_case) \
  _fitoa_word (value, buf, base, upper_case)
#endif

#endif	/* itoa.h */
