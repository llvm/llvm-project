/* Generic conversion to and from 8bit charsets,
   converting from UCS using gaps.
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

struct gap
{
  uint16_t start;
  uint16_t end;
  int32_t idx;
};

/* Now we can include the tables.  */
#include TABLES

#ifndef NONNUL
# define NONNUL(c)	((c) != '\0')
#endif


#define FROM_LOOP		from_gap
#define TO_LOOP			to_gap
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
    if (HAS_HOLES && __builtin_expect (ch == L'\0', 0) && NONNUL (*inptr))    \
      {									      \
	/* This is an illegal character.  */				      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
    else								      \
      {									      \
	put32 (outptr, ch);						      \
	outptr += 4;							      \
      }									      \
									      \
    ++inptr;								      \
  }
#define LOOP_NEED_FLAGS
#define ONEBYTE_BODY \
  {									      \
    uint32_t ch = to_ucs4[c];						      \
									      \
    if (HAS_HOLES && __builtin_expect (ch == L'\0', 0) && NONNUL (c))	      \
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
    const struct gap *rp = from_idx;					      \
    uint32_t ch = get32 (inptr);					      \
    unsigned char res;							      \
									      \
    if (__glibc_unlikely (ch >= 0xffff))				      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
	rp = NULL;							      \
      }									      \
    else								      \
      while (ch > rp->end)						      \
	++rp;								      \
    if (__builtin_expect (rp == NULL, 0)				      \
	|| __builtin_expect (ch < rp->start, 0)				      \
	|| (res = from_ucs4[ch + rp->idx],				      \
	    __builtin_expect (res, '\1') == '\0' && ch != 0))		      \
      {									      \
	/* This is an illegal character.  */				      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
									      \
    *outptr++ = res;							      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
