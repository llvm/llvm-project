/* Access functions for JISX0208 conversion.
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

#ifndef _JIS0208_H
#define _JIS0208_H	1

#include <gconv.h>
#include <stdint.h>

/* Struct for table with indices in UCS mapping table.  */
struct jisx0208_ucs_idx
{
  uint16_t start;
  uint16_t end;
  uint16_t idx;
};


/* Conversion table.  */
extern const uint16_t __jis0208_to_ucs[];

#define JIS0208_LAT1_MIN 0xa2
#define JIS0208_LAT1_MAX 0xf7
extern const char __jisx0208_from_ucs4_lat1[JIS0208_LAT1_MAX + 1
					    - JIS0208_LAT1_MIN][2];
extern const char __jisx0208_from_ucs4_greek[0xc1][2];
extern const struct jisx0208_ucs_idx __jisx0208_from_ucs_idx[];
extern const char __jisx0208_from_ucs_tab[][2];


static inline uint32_t
__attribute ((always_inline))
jisx0208_to_ucs4 (const unsigned char **s, size_t avail, unsigned char offset)
{
  unsigned char ch = *(*s);
  unsigned char ch2;
  int idx;

  if (ch < offset || (ch - offset) <= 0x20)
    return __UNKNOWN_10646_CHAR;

  if (avail < 2)
    return 0;

  ch2 = (*s)[1];
  if (ch2 < offset || (ch2 - offset) <= 0x20 || (ch2 - offset) >= 0x7f)
    return __UNKNOWN_10646_CHAR;

  idx = (ch - 0x21 - offset) * 94 + (ch2 - 0x21 - offset);
  if (idx >= 0x1e80)
    return __UNKNOWN_10646_CHAR;

  (*s) += 2;

  return __jis0208_to_ucs[idx] ?: ((*s) -= 2, __UNKNOWN_10646_CHAR);
}


static inline size_t
__attribute ((always_inline))
ucs4_to_jisx0208 (uint32_t wch, unsigned char *s, size_t avail)
{
  unsigned int ch = (unsigned int) wch;
  const char *cp;

  if (avail < 2)
    return 0;

  if (ch >= JIS0208_LAT1_MIN && ch <= JIS0208_LAT1_MAX)
    cp = __jisx0208_from_ucs4_lat1[ch - JIS0208_LAT1_MIN];
  else if (ch >= 0x391 && ch <= 0x451)
    cp = __jisx0208_from_ucs4_greek[ch - 0x391];
  else
    {
      const struct jisx0208_ucs_idx *rp = __jisx0208_from_ucs_idx;

      if (ch >= 0xffff)
	return __UNKNOWN_10646_CHAR;
      while (ch > rp->end)
	++rp;
      if (ch >= rp->start)
	cp = __jisx0208_from_ucs_tab[rp->idx + ch - rp->start];
      else
	return __UNKNOWN_10646_CHAR;
    }

  if (cp[0] == '\0')
    return __UNKNOWN_10646_CHAR;

  s[0] = cp[0];
  s[1] = cp[1];

  return 2;
}

#endif /* jis0208.h */
