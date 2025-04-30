/* Mapping tables for EUC-KR handling.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jungshik Shin <jshin@pantheon.yale.edu>
   and Ulrich Drepper <drepper@cygnus.com>, 1998.

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
#include <ksc5601.h>


static inline void
__attribute ((always_inline))
euckr_from_ucs4 (uint32_t ch, unsigned char *cp)
{
  if (ch > 0x9f)
    {
      if (__builtin_expect (ch, 0) == 0x20a9)
	{
	  /* Half-width Korean Currency WON sign.  There is no
             equivalent in EUC-KR.  Some mappings use \x5c because
             this is what some old Korean ASCII variants used but this
             is causing problems.  We map it to the FULL WIDTH WON SIGN.  */
	  cp[0] = '\xa3';
	  cp[1] = '\xdc';
	}
      else if (__builtin_expect (ucs4_to_ksc5601 (ch, cp, 2), 0)
	  != __UNKNOWN_10646_CHAR)
	{
	  cp[0] |= 0x80;
	  cp[1] |= 0x80;
	}
      else
	cp[0] = cp[1] = '\0';
    }
  else
    {
      /* There is no mapping for U005c but we nevertheless map it to
	 \x5c.  */
      cp[0] = (unsigned char) ch;
      cp[1] = '\0';
    }
}


/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"EUC-KR//"
#define FROM_LOOP		from_euc_kr
#define TO_LOOP			to_euc_kr
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define MIN_NEEDED_FROM		1
#define MAX_NEEDED_FROM		2
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0


/* First define the conversion function from EUC-KR to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
									      \
    if (ch <= 0x9f)							      \
      ++inptr;								      \
    else if (__glibc_unlikely (ch == 0xa0))				      \
      {									      \
	/* This is illegal.  */						      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
    else								      \
      {									      \
	/* Two-byte character.  First test whether the next byte	      \
	   is also available.  */					      \
	ch = ksc5601_to_ucs4 (&inptr, inend - inptr, 0x80);		      \
	if (__glibc_unlikely (ch == 0))					      \
	  {								      \
	    /* The second byte is not available.  */			      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	  /* This is an illegal character.  */				      \
	  STANDARD_FROM_LOOP_ERR_HANDLER (2);				      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define ONEBYTE_BODY \
  {									      \
    if (c <= 0x9f)							      \
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
    unsigned char cp[2];						      \
									      \
    /* Decomposing Hangul syllables not available in KS C 5601 into	      \
       Jamos should be considered either here or in euckr_from_ucs4() */      \
    euckr_from_ucs4 (ch, cp);						      \
									      \
    if (__builtin_expect (cp[0], '\1') == '\0' && ch != 0)		      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
									      \
	/* Illegal character.  */					      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
									      \
    *outptr++ = cp[0];							      \
    /* Now test for a possible second byte and write this if possible.  */    \
    if (cp[1] != '\0')							      \
      {									      \
	if (__glibc_unlikely (outptr >= outend))			      \
	  {								      \
	    /* The result does not fit into the buffer.  */		      \
	    --outptr;							      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	*outptr++ = cp[1];						      \
      }									      \
									      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
