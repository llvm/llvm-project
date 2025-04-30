/* Access functions for GB2312 conversion.
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

#ifndef _GB2312_H
#define _GB2312_H	1

#include <gconv.h>
#include <stdint.h>
#include <assert.h>

/* Conversion table.  */
extern const uint16_t __gb2312_to_ucs[];


static inline uint32_t
__attribute ((always_inline))
gb2312_to_ucs4 (const unsigned char **s, size_t avail, unsigned char offset)
{
  unsigned char ch = *(*s);
  unsigned char ch2;
  int idx;

  if (ch < offset || (ch - offset) <= 0x20 || (ch - offset) > 0x77)
    return __UNKNOWN_10646_CHAR;

  if (avail < 2)
    return 0;

  ch2 = (*s)[1];
  if ((ch2 - offset) <= 0x20 || (ch2 - offset) >= 0x7f)
    return __UNKNOWN_10646_CHAR;

  idx = (ch - 0x21 - offset) * 94 + (ch2 - 0x21 - offset);
  if (idx > 0x1ff1)
    return __UNKNOWN_10646_CHAR;

  (*s) += 2;

  return __gb2312_to_ucs[idx] ?: ((*s) -= 2, __UNKNOWN_10646_CHAR);
}


extern const char __gb2312_from_ucs4_tab1[][2];
extern const char __gb2312_from_ucs4_tab2[][2];
extern const char __gb2312_from_ucs4_tab3[][2];
extern const char __gb2312_from_ucs4_tab4[][2];
extern const char __gb2312_from_ucs4_tab5[][2];
extern const char __gb2312_from_ucs4_tab6[][2];
extern const char __gb2312_from_ucs4_tab7[][2];
extern const char __gb2312_from_ucs4_tab8[][2];
extern const char __gb2312_from_ucs4_tab9[][2];

static inline size_t
__attribute ((always_inline))
ucs4_to_gb2312 (uint32_t wch, unsigned char *s, size_t avail)
{
  unsigned int ch = (unsigned int) wch;
  char buf[2];
  const char *cp = buf;

  switch (ch)
    {
    case 0xa4 ... 0x101:
      cp = __gb2312_from_ucs4_tab1[ch - 0xa4];
      break;
    case 0x113:
      cp = "\x28\x25";
      break;
    case 0x11b:
      cp = "\x28\x27";
      break;
    case 0x12b:
      cp = "\x28\x29";
      break;
    case 0x14d:
      cp = "\x28\x2d";
      break;
    case 0x16b:
      cp = "\x28\x31";
      break;
    case 0x1ce:
      cp = "\x28\x23";
      break;
    case 0x1d0:
      cp = "\x28\x2b";
      break;
    case 0x1d2:
      cp = "\x28\x2f";
      break;
    case 0x1d4:
      cp = "\x28\x33";
      break;
    case 0x1d6:
      cp = "\x28\x35";
      break;
    case 0x1d8:
      cp = "\x28\x36";
      break;
    case 0x1da:
      cp = "\x28\x37";
      break;
    case 0x1dc:
      cp = "\x28\x38";
      break;
    case 0x2c7:
      cp = "\x21\x26";
      break;
    case 0x2c9:
      cp = "\x21\x25";
      break;
    case 0x391 ... 0x3c9:
      cp = __gb2312_from_ucs4_tab2[ch - 0x391];
      break;
    case 0x401 ... 0x451:
      cp = __gb2312_from_ucs4_tab3[ch - 0x401];
      break;
    case 0x2015 ... 0x203b:
      cp = __gb2312_from_ucs4_tab4[ch - 0x2015];
      break;
    case 0x2103 ... 0x22a5:
      cp = __gb2312_from_ucs4_tab5[ch - 0x2103];
      break;
    case 0x2312:
      cp = "\x21\x50";
      break;
    case 0x2460 ... 0x249b:
      cp = __gb2312_from_ucs4_tab6[ch - 0x2460];
      break;
    case 0x2500 ... 0x254b:
      buf[0] = '\x29';
      buf[1] = '\x24' + (ch % 256);
      break;
    case 0x25a0:
      cp = "\x21\x76";
      break;
    case 0x25a1:
      cp = "\x21\x75";
      break;
    case 0x25b2:
      cp = "\x21\x78";
      break;
    case 0x25b3:
      cp = "\x21\x77";
      break;
    case 0x25c6:
      cp = "\x21\x74";
      break;
    case 0x25c7:
      cp = "\x21\x73";
      break;
    case 0x25cb:
      cp = "\x21\x70";
      break;
    case 0x25ce:
      cp = "\x21\x72";
      break;
    case 0x25cf:
      cp = "\x21\x71";
      break;
    case 0x2605:
      cp = "\x21\x6f";
      break;
    case 0x2606:
      cp = "\x21\x6e";
      break;
    case 0x2640:
      cp = "\x21\x62";
      break;
    case 0x2642:
      cp = "\x21\x61";
      break;
    case 0x3000 ... 0x3129:
      cp = __gb2312_from_ucs4_tab7[ch - 0x3000];
      break;
    case 0x3220 ... 0x3229:
      buf[0] = '\x22';
      buf[1] = '\x65' + (ch - 0x3220);
      break;
    case 0x4e00 ... 0x9fa0:
      cp = __gb2312_from_ucs4_tab8[ch - 0x4e00];
      break;
    case 0xff01 ... 0xff5e:
      cp = __gb2312_from_ucs4_tab9[ch - 0xff01];
      break;
    case 0xffe0:
      cp = "\x21\x69";
      break;
    case 0xffe1:
      cp = "\x21\x6a";
      break;
    case 0xffe3:
      cp = "\x23\x7e";
      break;
    case 0xffe5:
      cp = "\x23\x24";
      break;
    default:
      return __UNKNOWN_10646_CHAR;
    }

  if (cp[0] == '\0')
    return __UNKNOWN_10646_CHAR;

  assert (cp[1] != '\0');

  if (avail < 2)
    return 0;

  s[0] = cp[0];
  s[1] = cp[1];

  return 2;
}

#endif	/* gb2312.h */
