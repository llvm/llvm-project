/* Functions for JISX0213 conversion.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <bruno@clisp.org>, 2002.

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

#ifndef _JISX0213_H
#define _JISX0213_H	1

#include <stdint.h>

extern const uint16_t __jisx0213_to_ucs_combining[][2];
extern const uint16_t __jisx0213_to_ucs_main[120 * 94];
extern const uint32_t __jisx0213_to_ucs_pagestart[];
extern const int16_t __jisx0213_from_ucs_level1[2715];
extern const uint16_t __jisx0213_from_ucs_level2[];

#define NELEMS(arr) (sizeof (arr) / sizeof (arr[0]))

static inline uint32_t
__attribute ((always_inline))
jisx0213_to_ucs4 (unsigned int row, unsigned int col)
{
  uint32_t val;

  if (row >= 0x121 && row <= 0x17e)
    row -= 289;
  else if (row == 0x221)
    row -= 451;
  else if (row >= 0x223 && row <= 0x225)
    row -= 452;
  else if (row == 0x228)
    row -= 454;
  else if (row >= 0x22c && row <= 0x22f)
    row -= 457;
  else if (row >= 0x26e && row <= 0x27e)
    row -= 519;
  else
    return 0x0000;

  if (col >= 0x21 && col <= 0x7e)
    col -= 0x21;
  else
    return 0x0000;

  val = __jisx0213_to_ucs_main[row * 94 + col];
  val = __jisx0213_to_ucs_pagestart[val >> 8] + (val & 0xff);
  if (val == 0xfffd)
    val = 0x0000;
  return val;
}

static inline uint16_t
__attribute ((always_inline))
ucs4_to_jisx0213 (uint32_t ucs)
{
  if (ucs < NELEMS (__jisx0213_from_ucs_level1) << 6)
    {
      int index1 = __jisx0213_from_ucs_level1[ucs >> 6];
      if (index1 >= 0)
	return __jisx0213_from_ucs_level2[(index1 << 6) + (ucs & 0x3f)];
    }
  return 0x0000;
}

static inline int
__attribute ((always_inline))
jisx0213_added_in_2004_p (uint16_t val)
{
  /* From JISX 0213:2000 to JISX 0213:2004, 10 characters were added to
     plane 1, and plane 2 was left unchanged.  See ISO-IR-233.  */
  switch (val >> 8)
    {
    case 0x2e:
      return val == 0x2e21;
    case 0x2f:
      return val == 0x2f7e;
    case 0x4f:
      return val == 0x4f54 || val == 0x4f7e;
    case 0x74:
      return val == 0x7427;
    case 0x7e:
      return val >= 0x7e7a && val <= 0x7e7e;
    default:
      return 0;
    }
}

#endif /* _JISX0213_H */
