/* Mapping tables for EUC-JP handling.
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
#include <stdint.h>
#include <gconv.h>
#include <jis0201.h>
#include <jis0208.h>
#include <jis0212.h>

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"EUC-JP//"
#define FROM_LOOP		from_euc_jp
#define TO_LOOP			to_euc_jp
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define MIN_NEEDED_FROM		1
#define MAX_NEEDED_FROM		3
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0


/* First define the conversion function from EUC-JP to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
									      \
    if (ch < 0x8e || (ch >= 0x90 && ch <= 0x9f))			      \
      ++inptr;								      \
    else if (ch == 0xff)						      \
      {									      \
	/* This is illegal.  */						      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
    else								      \
      {									      \
	/* Two or more byte character.  First test whether the next	      \
	   byte is also available.  */					      \
	int ch2;							      \
									      \
	if (__glibc_unlikely (inptr + 1 >= inend))			      \
	  {								      \
	    /* The second byte is not available.  Store the		      \
	       intermediate result.  */					      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
									      \
	ch2 = inptr[1];							      \
									      \
	/* All second bytes of a multibyte character must be >= 0xa1. */      \
	if (__glibc_unlikely (ch2 < 0xa1))				      \
	  STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
									      \
	if (ch == 0x8e)							      \
	  {								      \
	    /* This is code set 2: half-width katakana.  */		      \
	    ch = jisx0201_to_ucs4 (ch2);				      \
	    if (__builtin_expect (ch, 0) == __UNKNOWN_10646_CHAR)	      \
	      STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
									      \
	    inptr += 2;							      \
	  }								      \
	else								      \
	  {								      \
	    const unsigned char *endp;					      \
									      \
	    if (ch == 0x8f)						      \
	      {								      \
		/* This is code set 3: JIS X 0212-1990.  */		      \
		endp = inptr + 1;					      \
									      \
		ch = jisx0212_to_ucs4 (&endp, inend - endp, 0x80);	      \
	      }								      \
	    else							      \
	      {								      \
		/* This is code set 1: JIS X 0208.  */			      \
		endp = inptr;						      \
									      \
		ch = jisx0208_to_ucs4 (&endp, inend - inptr, 0x80);	      \
	      }								      \
									      \
	    if (__builtin_expect (ch, 1) == 0)				      \
	      {								      \
		/* Not enough input available.  */			      \
		result = __GCONV_INCOMPLETE_INPUT;			      \
		break;							      \
	      }								      \
	    if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	      /* Illegal character.  */					      \
	      STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
									      \
	    inptr = endp;						      \
	  }								      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
  }
#define ONEBYTE_BODY \
  {									      \
    if (c < 0x8e || (c >= 0x90 && c <= 0x9f))				      \
      return c;								      \
    else								      \
      return WEOF;							      \
  }
#define LOOP_NEED_FLAGS
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
    if (ch < 0x8e || (ch >= 0x90 && ch <= 0x9f))			      \
      /* It's plain ASCII or C1.  */					      \
      *outptr++ = ch;							      \
    else if (ch == 0xa5)						      \
      /* YEN sign => backslash  */					      \
      *outptr++ = 0x5c;							      \
    else if (ch == 0x203e)						      \
      /* overscore => asciitilde */					      \
      *outptr++ = 0x7e;							      \
    else								      \
      {									      \
	/* Try the JIS character sets.  */				      \
	size_t found;							      \
									      \
	/* See whether we have room for at least two characters.  */	      \
	if (__glibc_unlikely (outptr + 1 >= outend))			      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
									      \
	found = ucs4_to_jisx0201 (ch, outptr + 1);			      \
	if (found != __UNKNOWN_10646_CHAR)				      \
	  {								      \
	    /* Yes, it's a JIS 0201 character.  Store the shift byte.  */     \
	    *outptr = 0x8e;						      \
	    outptr += 2;						      \
	  }								      \
	else								      \
	  {								      \
	    /* No JIS 0201 character.  */				      \
	    found = ucs4_to_jisx0208 (ch, outptr, 2);			      \
	    /* Please note that we always have enough room for the output. */ \
	    if (found != __UNKNOWN_10646_CHAR)				      \
	      {								      \
		/* It's a JIS 0208 character, adjust it for EUC-JP.  */	      \
		*outptr++ += 0x80;					      \
		*outptr++ += 0x80;					      \
	      }								      \
	    else							      \
	      {								      \
		/* No JIS 0208 character.  */				      \
		found = ucs4_to_jisx0212 (ch, outptr + 1,		      \
					  outend - outptr - 1);		      \
		  							      \
		if (__builtin_expect (found, 1) == 0)			      \
		  {							      \
		    /* We ran out of space.  */				      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
		else if (__builtin_expect (found, 0) != __UNKNOWN_10646_CHAR) \
		  {							      \
		    /* It's a JIS 0212 character, adjust it for EUC-JP.  */   \
		    *outptr++ = 0x8f;					      \
		    *outptr++ += 0x80;					      \
		    *outptr++ += 0x80;					      \
		  }							      \
		else							      \
		  {							      \
		    UNICODE_TAG_HANDLER (ch, 4);			      \
									      \
		    /* Illegal character.  */				      \
		    STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
		  }							      \
	      }								      \
	  }								      \
      }									      \
									      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
