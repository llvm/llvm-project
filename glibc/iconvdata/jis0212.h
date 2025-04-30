/* Access functions for JISX0212 conversion.
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

#ifndef _JIS0212_H
#define _JIS0212_H	1

#include <assert.h>
#include <gconv.h>
#include <stdint.h>


/* Struct for table with indices in mapping table.  */
struct jisx0212_idx
{
  uint16_t start;
  uint16_t end;
  uint16_t idx;
};

/* Conversion table.  */
extern const struct jisx0212_idx __jisx0212_to_ucs_idx[];
extern const uint16_t __jisx0212_to_ucs[];

extern const struct jisx0212_idx __jisx0212_from_ucs_idx[];
extern const char __jisx0212_from_ucs[][2];


static inline uint32_t
__attribute ((always_inline))
jisx0212_to_ucs4 (const unsigned char **s, size_t avail, unsigned char offset)
{
  const struct jisx0212_idx *rp = __jisx0212_to_ucs_idx;
  unsigned char ch = *(*s);
  unsigned char ch2;
  uint32_t wch = 0;
  int idx;

  if (ch < offset || (ch - offset) < 0x22 || (ch - offset) > 0x6d)
    return __UNKNOWN_10646_CHAR;

  if (avail < 2)
    return 0;

  ch2 = (*s)[1];
  if (ch2 < offset || (ch2 - offset) <= 0x20 || (ch2 - offset) >= 0x7f)
    return __UNKNOWN_10646_CHAR;

  idx = (ch - offset - 0x21) * 94 + (ch2 - offset - 0x21);

  while (idx > rp->end)
    ++rp;
  if (idx >= rp->start)
    wch = __jisx0212_to_ucs[rp->idx + idx - rp->start];

  if (wch != L'\0')
    (*s) += 2;
  else
    wch = __UNKNOWN_10646_CHAR;

  return wch;
}


static inline size_t
__attribute ((always_inline))
ucs4_to_jisx0212 (uint32_t wch, unsigned char *s, size_t avail)
{
  const struct jisx0212_idx *rp = __jisx0212_from_ucs_idx;
  unsigned int ch = (unsigned int) wch;
  const char *cp;

  if (ch >= 0xffff)
    return __UNKNOWN_10646_CHAR;
  while (ch > rp->end)
    ++rp;
  if (ch >= rp->start)
    cp = __jisx0212_from_ucs[rp->idx + ch - rp->start];
  else
    return __UNKNOWN_10646_CHAR;

  if (cp[0] == '\0')
    return __UNKNOWN_10646_CHAR;

  s[0] = cp[0];
  assert (cp[1] != '\0');
  if (avail < 2)
    return 0;

  s[1] = cp[1];
  return 2;
}

#endif /* jis0212.h */
