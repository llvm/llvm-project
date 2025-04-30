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

#ifndef _ITOWA_H
#define _ITOWA_H	1
#include <features.h>
#include <wchar.h>
#include <_itoa.h>

/* Convert VALUE into ASCII in base BASE (2..36).
   Write backwards starting the character just before BUFLIM.
   Return the address of the first (left-to-right) character in the number.
   Use upper case letters iff UPPER_CASE is nonzero.  */

extern wchar_t *_itowa (unsigned long long int value, wchar_t *buflim,
			unsigned int base, int upper_case);

static inline wchar_t *
__attribute__ ((unused, always_inline))
_itowa_word (_ITOA_WORD_TYPE value, wchar_t *buflim,
	     unsigned int base, int upper_case)
{
  extern const wchar_t _itowa_upper_digits[] attribute_hidden;
  extern const wchar_t _itowa_lower_digits[] attribute_hidden;
  const wchar_t *digits = (upper_case
			   ? _itowa_upper_digits : _itowa_lower_digits);
  wchar_t *bp = buflim;

  switch (base)
    {
#define SPECIAL(Base)							      \
    case Base:								      \
      do								      \
	*--bp = digits[value % Base];					      \
      while ((value /= Base) != 0);					      \
      break

      SPECIAL (10);
      SPECIAL (16);
      SPECIAL (8);
    default:
      do
	*--bp = digits[value % base];
      while ((value /= base) != 0);
    }
  return bp;
}
#undef SPECIAL

#if !_ITOA_NEEDED
/* No need for special long long versions.  */
# define _itowa(value, buf, base, upper_case) \
  _itowa_word (value, buf, base, upper_case)
#endif

#endif	/* itowa.h */
