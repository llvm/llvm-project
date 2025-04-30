/* Access functions for KS C 5601-1992 based encoding conversion.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#ifndef _KSC5601_H
#define _KSC5601_H	1

#define KSC5601_HANGUL 2350
#define KSC5601_HANJA  4888
#define KSC5601_SYMBOL  989

#include <gconv.h>
#include <stdint.h>

/* Structure to map from UCS to KSC.  This structure should be packed
   on all platforms.  */
struct map
{
  uint16_t ucs;
  char val[2];
};

/* Conversion table.  */
extern const uint16_t __ksc5601_hangul_to_ucs[KSC5601_HANGUL];
extern const uint16_t __ksc5601_sym_to_ucs[];
extern const struct map __ksc5601_sym_from_ucs[KSC5601_SYMBOL];
extern const uint16_t __ksc5601_hanja_to_ucs[KSC5601_HANJA];
extern const struct map __ksc5601_hanja_from_ucs[KSC5601_HANJA];


static inline uint32_t
__attribute ((always_inline))
ksc5601_to_ucs4 (const unsigned char **s, size_t avail, unsigned char offset)
{
  unsigned char ch = **s;
  unsigned char ch2;
  int idx;

  if (avail < 2)
    return 0;

  /* row 94(0x7e) and row 41(0x49) are user-defined area in KS C 5601 */

  if (ch < offset || (ch - offset) <= 0x20 || (ch - offset) >= 0x7e
      || (ch - offset) == 0x49)
    return __UNKNOWN_10646_CHAR;

  ch2 = (*s)[1];
  if (ch2 < offset || (ch2 - offset) <= 0x20 || (ch2 - offset) >= 0x7f)
    return __UNKNOWN_10646_CHAR;

  idx = (ch - offset - 0x21) * 94 + (ch2 - offset - 0x21);

  /* 1410 = 15 * 94 , 3760 = 40 * 94
     Hangul in KS C 5601 : row 16 - row 40 */

  *s += 2;

  if (idx >= 1410 && idx < 1410 + KSC5601_HANGUL)
    return (__ksc5601_hangul_to_ucs[idx - 1410]
	    ?: (*s -= 2, __UNKNOWN_10646_CHAR));
  else if (idx >= 3854)
    /* Hanja : row 42 - row 93 : 3854 = 94 * (42-1) */
   return (__ksc5601_hanja_to_ucs[idx - 3854]
	   ?: (*s -= 2, __UNKNOWN_10646_CHAR));
  else if (idx <= 1114)
    return __ksc5601_sym_to_ucs[idx] ?: (*s -= 2, __UNKNOWN_10646_CHAR);

  *s -= 2;
  return __UNKNOWN_10646_CHAR;
}

static inline size_t
__attribute ((always_inline))
ucs4_to_ksc5601_hangul (uint32_t wch, unsigned char *s, size_t avail)
{
  int l = 0;
  int u = KSC5601_HANGUL - 1;
  uint32_t try;

  while (l <= u)
    {
      int m = (l + u) / 2;
      try = (uint32_t) __ksc5601_hangul_to_ucs[m];
      if (try > wch)
	u = m - 1;
      else if (try < wch)
	l= m + 1;
      else
	{
	  if (avail < 2)
	    return 0;

	  s[0] = (m / 94) + 0x30;
	  s[1] = (m % 94) + 0x21;

	  return 2;
	}
    }

  return __UNKNOWN_10646_CHAR;
}


static inline size_t
__attribute ((always_inline))
ucs4_to_ksc5601_hanja (uint32_t wch, unsigned char *s, size_t avail)
{
  int l = 0;
  int u = KSC5601_HANJA - 1;
  uint32_t try;

  while (l <= u)
    {
      int m = (l + u) / 2;
      try = (uint32_t) __ksc5601_hanja_from_ucs[m].ucs;
      if (try > wch)
	u=m-1;
      else if (try < wch)
	l = m + 1;
      else
	{
	  if (avail < 2)
	    return 0;

	  s[0] = __ksc5601_hanja_from_ucs[m].val[0];
	  s[1] = __ksc5601_hanja_from_ucs[m].val[1];

	  return 2;
	}
    }

  return __UNKNOWN_10646_CHAR;
}

static inline  size_t
__attribute ((always_inline))
ucs4_to_ksc5601_sym (uint32_t wch, unsigned char *s, size_t avail)
{
  int l = 0;
  int u = KSC5601_SYMBOL - 1;
  uint32_t try;

  while (l <= u)
    {
      int m = (l + u) / 2;
      try = __ksc5601_sym_from_ucs[m].ucs;
      if (try > wch)
	u = m - 1;
      else if (try < wch)
	l = m + 1;
      else
	{
	  if (avail < 2)
	    return 0;

	  s[0] = __ksc5601_sym_from_ucs[m].val[0];
	  s[1] = __ksc5601_sym_from_ucs[m].val[1];

	  return 2;
	}
    }

  return __UNKNOWN_10646_CHAR;
}


static inline size_t
__attribute ((always_inline))
ucs4_to_ksc5601 (uint32_t wch, unsigned char *s, size_t avail)
{
  if (wch >= 0xac00 && wch <= 0xd7a3)
    return ucs4_to_ksc5601_hangul (wch, s, avail);
  else if ((wch >= 0x4e00 && wch <= 0x9fff)
	   || (wch >= 0xf900 && wch <= 0xfa0b))
    return ucs4_to_ksc5601_hanja (wch, s, avail);
  else
    return ucs4_to_ksc5601_sym (wch, s, avail);
}

#endif /* ksc5601.h */
