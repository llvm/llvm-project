/* Conversion to and from ARMSCII-8
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

#include <dlfcn.h>
#include <stdint.h>

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"ARMSCII-8//"
#define FROM_LOOP		from_armscii_8
#define TO_LOOP			to_armscii_8
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define MIN_NEEDED_FROM		1
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0


static const uint16_t map_from_armscii_8[0xfe - 0xa2 + 1] =
  {
    0x0587, 0x0589, 0x0029, 0x0028, 0x00bb, 0x00ab, 0x2014, 0x002e,
    0x055d, 0x002c, 0x002d, 0x058a, 0x2026, 0x055c, 0x055b, 0x055e,
    0x0531, 0x0561, 0x0532, 0x0562, 0x0533, 0x0563, 0x0534, 0x0564,
    0x0535, 0x0565, 0x0536, 0x0566, 0x0537, 0x0567, 0x0538, 0x0568,
    0x0539, 0x0569, 0x053a, 0x056a, 0x053b, 0x056b, 0x053c, 0x056c,
    0x053d, 0x056d, 0x053e, 0x056e, 0x053f, 0x056f, 0x0540, 0x0570,
    0x0541, 0x0571, 0x0542, 0x0572, 0x0543, 0x0573, 0x0544, 0x0574,
    0x0545, 0x0575, 0x0546, 0x0576, 0x0547, 0x0577, 0x0548, 0x0578,
    0x0549, 0x0579, 0x054a, 0x057a, 0x054b, 0x057b, 0x054c, 0x057c,
    0x054d, 0x057d, 0x054e, 0x057e, 0x054f, 0x057f, 0x0550, 0x0580,
    0x0551, 0x0581, 0x0552, 0x0582, 0x0553, 0x0583, 0x0554, 0x0584,
    0x0555, 0x0585, 0x0556, 0x0586, 0x055a
  };


/* First define the conversion function from ARMSCII-8 to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint_fast8_t ch = *inptr;						      \
									      \
    if (ch <= 0xa0)							      \
      {									      \
        /* Upto and including 0xa0 the ARMSCII-8 corresponds to Unicode.  */  \
        *((uint32_t *) outptr) = ch;					      \
        outptr += sizeof (uint32_t);					      \
      }									      \
    else if (ch >= 0xa2 && ch <= 0xfe)					      \
      {									      \
        /* Use the table.  */						      \
        *((uint32_t *) outptr) = map_from_armscii_8[ch - 0xa2];		      \
        outptr += sizeof (uint32_t);					      \
      }									      \
    else								      \
      {									      \
	/* This is an illegal character.  */				      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
									      \
    ++inptr;								      \
  }
#define LOOP_NEED_FLAGS
#define ONEBYTE_BODY \
  {									      \
    if (c <= 0xa0)							      \
      /* Upto and including 0xa0 the ARMSCII-8 corresponds to Unicode.  */    \
      return c;								      \
    else if (c >= 0xa2 && c <= 0xfe)					      \
      /* Use the table.  */						      \
      return map_from_armscii_8[c - 0xa2];				      \
    else								      \
      return WEOF;							      \
  }
#include <iconv/loop.c>


static const unsigned char map_to_armscii_8[0x58a - 0x531 + 1] =
  {
    0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe, 0xc0,
    0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0,
    0xd2, 0xd4, 0xd6, 0xd8, 0xda, 0xdc, 0xde, 0xe0,
    0xe2, 0xe4, 0xe6, 0xe8, 0xea, 0xec, 0xee, 0xf0,
    0xf2, 0xf4, 0xf6, 0xf8, 0xfa, 0xfc, 0x00, 0x00,
    0x00, 0xfe, 0xb0, 0xaf, 0xaa, 0xb1, 0x00, 0x00,
    0xb3, 0xb5, 0xb7, 0xb9, 0xbb, 0xbd, 0xbf, 0xc1,
    0xc3, 0xc5, 0xc7, 0xc9, 0xcb, 0xcd, 0xcf, 0xd1,
    0xd3, 0xd5, 0xd7, 0xd9, 0xdb, 0xdd, 0xdf, 0xe1,
    0xe3, 0xe5, 0xe7, 0xe9, 0xeb, 0xed, 0xef, 0xf1,
    0xf3, 0xf5, 0xf7, 0xf9, 0xfb, 0xfd, 0xa2, 0x00,
    0xa3, 0xad
  };


/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch = *((const uint32_t *) inptr);				      \
									      \
    if (ch <= 0xa0)							      \
      /* Upto and including 0xa0 the ARMSCII-8 corresponds to Unicode.  */    \
      *outptr = (unsigned char) ch;					      \
    else if (ch == 0xab)						      \
      *outptr = 0xa7;							      \
    else if (ch == 0xbb)						      \
      *outptr = 0xa6;							      \
    else if (ch >= 0x531 && ch <= 0x58a)				      \
      {									      \
	unsigned char oc = map_to_armscii_8[ch - 0x531];		      \
									      \
	if (oc == 0)							      \
	  /* No valid mapping.  */					      \
	  goto err;							      \
									      \
	*outptr = oc;							      \
      }									      \
    else if (ch == 0x2014)						      \
      *outptr = 0xa8;							      \
    else if (ch == 0x2026)						      \
      *outptr = 0xae;							      \
    else								      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
									      \
	/* We have an illegal character.  */				      \
      err:								      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
    ++outptr;								      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
