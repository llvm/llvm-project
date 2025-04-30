/* Conversion to and from ISO 11548-1.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997,
	Samuel Thibault <samuel.thibault@ens-lyon.org>, 2005.

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
#define CHARSET_NAME		"ISO_11548-1//"
#define FROM_LOOP		from_iso11548_1
#define TO_LOOP			to_iso11548_1
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define MIN_NEEDED_FROM		1
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0

#define BRAILLE_UCS_BASE	0x2800

/* First define the conversion function from ISO 11548-1 to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  *((uint32_t *) outptr) = BRAILLE_UCS_BASE | (*inptr++);		      \
  outptr += sizeof (uint32_t);
#define ONEBYTE_BODY \
  {									      \
    return BRAILLE_UCS_BASE | c;					      \
  }
#include <iconv/loop.c>


/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch = *((const uint32_t *) inptr);				      \
    if (__glibc_unlikely ((ch & 0xffffff00u) != BRAILLE_UCS_BASE))	      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
									      \
	/* We have an illegal character.  */				      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
    else								      \
      *outptr++ = (unsigned char) (ch & 0xff);				      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
