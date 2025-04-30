/* Generic conversion to and from 8bit charsets.
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

#define FROM_LOOP		from_generic
#define TO_LOOP			to_generic
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define MIN_NEEDED_FROM		1
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0


/* First define the conversion function from the 8bit charset to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = to_ucs4[*inptr];					      \
									      \
    if (HAS_HOLES && __builtin_expect (ch == L'\0', 0) && *inptr != '\0')     \
      {									      \
	/* This is an illegal character.  */				      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
    ++inptr;								      \
  }
#define LOOP_NEED_FLAGS
#define ONEBYTE_BODY \
  {									      \
    uint32_t ch = to_ucs4[c];						      \
									      \
    if (HAS_HOLES && __builtin_expect (ch == L'\0', 0) && c != '\0')	      \
      return WEOF;							      \
    else								      \
      return ch;							      \
  }
#include <iconv/loop.c>


/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch = get32 (inptr);					      \
									      \
    if (__builtin_expect (ch >= sizeof (from_ucs4) / sizeof (from_ucs4[0]), 0)\
	|| (__builtin_expect (from_ucs4[ch], '\1') == '\0' && ch != 0))	      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
									      \
	/* This is an illegal character.  */				      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
									      \
    *outptr++ = from_ucs4[ch];						      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
