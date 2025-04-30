/* Access functions for CNS 11643, plane 2 handling.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#include <stdint.h>
#include <gconv.h>

/* Table for CNS 11643, plane 2 to UCS4 conversion.  */
extern const uint16_t __cns11643l2_to_ucs4_tab[];


static inline uint32_t
__attribute ((always_inline))
cns11643l2_to_ucs4 (const unsigned char **s, size_t avail,
		    unsigned char offset)
{
  unsigned char ch = *(*s);
  unsigned char ch2;
  int idx;

  if (ch < offset || (ch - offset) <= 0x20 || (ch - offset) > 0x7d)
    return __UNKNOWN_10646_CHAR;

  if (avail < 2)
    return 0;

  ch2 = (*s)[1];
  if ((ch2 - offset) <= 0x20 || (ch2 - offset) >= 0x7f)
    return __UNKNOWN_10646_CHAR;

  idx = (ch - 0x21 - offset) * 94 + (ch2 - 0x21 - offset);
  if (idx > 0x1de1)
    return __UNKNOWN_10646_CHAR;

  (*s) += 2;

  return __cns11643l2_to_ucs4_tab[idx] ?: ((*s) -= 2, __UNKNOWN_10646_CHAR);
}


/* The table which contains the CNS 11643 level 2 mappings.  */
extern const char __cns11643_from_ucs4p0_tab[][3];


static inline size_t
__attribute ((always_inline))
ucs4_to_cns11643l2 (uint32_t wch, unsigned char *s, size_t avail)
{
  unsigned int ch = (unsigned int) wch;
  const char *cp = NULL;

  if (ch >= 0x4e07 && ch <= 0x9fa4)
    {
      cp = __cns11643_from_ucs4p0_tab[ch - 0x3400];
      if (cp[0] == '\2')
	++cp;
      else
	cp = NULL;
    }

  if (cp == NULL)
    return __UNKNOWN_10646_CHAR;

  if (avail < 2)
    return 0;

  s[0] = cp[0];
  s[1] = cp[1];

  return 2;
}
