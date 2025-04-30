/* Tables for conversion to and from ISO-IR-165.
   converting from UCS using gaps.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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

#ifndef _ISO_IR_165_H
#define _ISO_IR_165_H	1

#include <gconv.h>
#include <stdint.h>

struct gap
{
  uint16_t start;
  uint16_t end;
  int32_t idx;
};

/* Table for ISO-IR-165 (CCITT Chinese) to UCS4 conversion.  */
#define ISOIR165_FROMSIZE	0x2284
extern const uint16_t __isoir165_to_tab[ISOIR165_FROMSIZE];


/* XXX If we at some point need an offset value to decode the byte
   sequences another parameter can be added.  */
static inline uint32_t
__attribute ((always_inline))
isoir165_to_ucs4 (const unsigned char **s, size_t avail)
{
  unsigned char ch = *(*s);
  unsigned char ch2;
  uint32_t res;

  if (ch <= 0x20 || ch >= 0x7f)
    return __UNKNOWN_10646_CHAR;

  if (avail < 2)
    return 0;

  ch2 = (*s)[1];
  if (ch2 <= 0x20 || ch2 >= 0x7f)
    return __UNKNOWN_10646_CHAR;

  res = __isoir165_to_tab[(ch - 0x21) * 94 + (ch2 - 0x21)];
  if (res == 0)
    return __UNKNOWN_10646_CHAR;

  *s += 2;
  return res;
}


/* Tables for ISO-IR-165 (CCITT Chinese) from UCS4 conversion.  */
extern const struct gap __isoir165_from_idx[];
extern const char __isoir165_from_tab[];

static inline size_t
__attribute ((always_inline))
ucs4_to_isoir165 (uint32_t wch, unsigned char *s, size_t avail)
{
  unsigned int ch = (unsigned int) wch;
  const char *cp;
  const struct gap *rp = __isoir165_from_idx;

  if (ch > 0xffe5)
    /* This is an illegal character.  */
    return __UNKNOWN_10646_CHAR;

  while (ch > rp->end)
    ++rp;
  if (ch < rp->start)
    /* This is an illegal character.  */
    return __UNKNOWN_10646_CHAR;

  /* The two bytes following the index given in this record give the
     encoding in ISO-IR-165.  Unless the bytes are zero.  */
  cp = &__isoir165_from_tab[(ch + rp->idx) * 2];
  if (*cp == '\0')
    /* This is an illegal character.  */
    return __UNKNOWN_10646_CHAR;

  if (avail < 2)
    return 0;

  s[0] = cp[0];
  s[1] = cp[1];

  return 2;
}

#endif	/* iso-ir-165.h */
