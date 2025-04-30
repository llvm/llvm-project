/* Access functions for JISX0201 conversion.
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

#ifndef _JIS0201_H
#define _JIS0201_H	1

#include <stdint.h>

/* Conversion table.  */
extern const uint32_t __jisx0201_to_ucs4[];


static inline uint32_t
__attribute ((always_inline))
jisx0201_to_ucs4 (char ch)
{
  uint32_t val = __jisx0201_to_ucs4[(unsigned char) ch];

  if (val == 0 && ch != '\0')
    val = __UNKNOWN_10646_CHAR;

  return val;
}


static inline size_t
__attribute ((always_inline))
ucs4_to_jisx0201 (uint32_t wch, unsigned char *s)
{
  unsigned char ch;

  if (wch == 0xa5)
    ch = '\x5c';
  else if (wch == 0x203e)
    ch = '\x7e';
  else if (wch < 0x7e && wch != 0x5c)
    ch = wch;
  else if (wch >= 0xff61 && wch <= 0xff9f)
    ch = wch - 0xfec0;
  else
    return __UNKNOWN_10646_CHAR;

  s[0] = ch;
  return 1;
}

#endif /* jis0201.h */
