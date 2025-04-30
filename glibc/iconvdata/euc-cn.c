/* Mapping tables for EUC-CN handling.
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

#include <dlfcn.h>
#include <gb2312.h>
#include <stdint.h>

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"EUC-CN//"
#define FROM_LOOP		from_euc_cn
#define TO_LOOP			to_euc_cn
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define MIN_NEEDED_FROM		1
#define MAX_NEEDED_FROM		2
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0


/* First define the conversion function from EUC-CN to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
									      \
    if (ch <= 0x7f)							      \
      ++inptr;								      \
    else								      \
      if ((__builtin_expect (ch <= 0xa0, 0) && ch != 0x8e && ch != 0x8f)      \
	  || __builtin_expect (ch > 0xfe, 0))				      \
	{								      \
	  /* This is illegal.  */					      \
	  STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	}								      \
      else								      \
	{								      \
	  /* Two or more byte character.  First test whether the	      \
	     next byte is also available.  */				      \
	  const unsigned char *endp;					      \
									      \
	  if (__glibc_unlikely (inptr + 1 >= inend))			      \
	    {								      \
	      /* The second character is not available.  Store		      \
		 the intermediate result.  */				      \
	      result = __GCONV_INCOMPLETE_INPUT;			      \
	      break;							      \
	    }								      \
									      \
	  ch = inptr[1];						      \
									      \
	  /* All second bytes of a multibyte character must be >= 0xa1. */    \
	  if (__glibc_unlikely (ch < 0xa1))				      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
									      \
	  /* This is code set 1: GB 2312-80.  */			      \
	  endp = inptr;							      \
									      \
	  ch = gb2312_to_ucs4 (&endp, 2, 0x80);				      \
	  if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	    {								      \
	      /* This is an illegal character.  */			      \
	      STANDARD_FROM_LOOP_ERR_HANDLER (2);			      \
	    }								      \
									      \
	  inptr += 2;							      \
	}								      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define ONEBYTE_BODY \
  {									      \
    if (c < 0x80)							      \
      return c;								      \
    else								      \
      return WEOF;							      \
  }
#include <iconv/loop.c>


/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch = get32 (inptr);					      \
									      \
    if (ch <= L'\x7f')							      \
      /* It's plain ASCII.  */						      \
      *outptr++ = (unsigned char) ch;					      \
    else								      \
      {									      \
	size_t found;							      \
									      \
	found = ucs4_to_gb2312 (ch, outptr, outend - outptr);		      \
	if (__builtin_expect (found, 1) != 0)				      \
	  {								      \
	    if (__builtin_expect (found, 0) == __UNKNOWN_10646_CHAR)	      \
	      {								      \
		UNICODE_TAG_HANDLER (ch, 4);				      \
									      \
		/* Illegal character.  */				      \
		STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
	      }								      \
									      \
	    /* It's a GB 2312 character, adjust it for EUC-CN.  */	      \
	    *outptr++ += 0x80;						      \
	    *outptr++ += 0x80;						      \
	  }								      \
	else								      \
	  {								      \
	    /* We ran out of space.  */					      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
      }									      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
