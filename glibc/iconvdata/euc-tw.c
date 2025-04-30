/* Mapping tables for EUC-TW handling.
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
#include <cns11643l1.h>
#include <cns11643.h>

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"EUC-TW//"
#define FROM_LOOP		from_euc_tw
#define TO_LOOP			to_euc_tw
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define MIN_NEEDED_FROM		1
#define MAX_NEEDED_FROM		4
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0


/* First define the conversion function from EUC-TW to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
    									      \
    if (ch <= 0x7f)							      \
      /* Plain ASCII.  */						      \
      ++inptr;								      \
    else if ((ch <= 0xa0 || ch > 0xfe) && ch != 0x8e)			      \
      {									      \
	/* This is illegal.  */						      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
    else								      \
      {									      \
	/* Two or more byte character.  First test whether the next byte      \
	   is also available.  */					      \
	uint32_t ch2;							      \
									      \
	if (inptr + 1 >= inend)						      \
	  {								      \
	    /* The second byte is not available.  Store the intermediate      \
	       result.  */						      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
									      \
	ch2 = *(inptr + 1);						      \
									      \
	/* All second bytes of a multibyte character must be >= 0xa1. */      \
	if (ch2 < 0xa1 || ch2 == 0xff)					      \
	  STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
									      \
	if (ch == 0x8e)							      \
	  {								      \
	    /* This is code set 2: CNS 11643, planes 1 to 16.  */	      \
	    const unsigned char *endp = inptr + 1;			      \
									      \
	    ch = cns11643_to_ucs4 (&endp, inend - inptr - 1, 0x80);	      \
									      \
	    if (ch == 0)						      \
	      {								      \
		/* The third or fourth byte is not available.  Store	      \
		   the intermediate result.  */				      \
		result = __GCONV_INCOMPLETE_INPUT;			      \
		break;							      \
	      }								      \
									      \
	    if (ch == __UNKNOWN_10646_CHAR)				      \
	      /* Illegal input.  */					      \
	      STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
									      \
	    inptr += 4;							      \
	  }								      \
	else								      \
	  {								      \
	    /* This is code set 1: CNS 11643, plane 1.  */		      \
	    const unsigned char *endp = inptr;				      \
									      \
	    ch = cns11643l1_to_ucs4 (&endp, inend - inptr, 0x80);	      \
	    /* Please note that we need not test for the missing input	      \
	       characters here anymore.  */				      \
	    if (ch == __UNKNOWN_10646_CHAR)				      \
	      /* Illegal input.  */					      \
	      STANDARD_FROM_LOOP_ERR_HANDLER (2);			      \
									      \
	    inptr += 2;							      \
	  }								      \
      }									      \
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
    if (ch <= 0x7f)							      \
      /* It's plain ASCII.  */						      \
      *outptr++ = ch;							      \
    else								      \
      {									      \
	/* Try the CNS 11643 planes.  */				      \
	size_t found;							      \
									      \
	found = ucs4_to_cns11643l1 (ch, outptr, outend - outptr);	      \
	if (__builtin_expect (found, 1) == 0)				      \
	  {								      \
	    /* We ran out of space.  */					      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	if (__builtin_expect (found, 1) != __UNKNOWN_10646_CHAR)	      \
	  {								      \
	    /* It's a CNS 11643, plane 1 character, adjust it for EUC-TW.  */ \
	    *outptr++ += 0x80;						      \
	    *outptr++ += 0x80;						      \
	  }								      \
	else								      \
	  {								      \
	    /* No CNS 11643, plane 1 character.  */			      \
									      \
	    found = ucs4_to_cns11643 (ch, outptr + 1, outend - outptr - 1);   \
	    if (__builtin_expect (found, 1) == 0)			      \
	      {								      \
		/* We ran out of space.  */				      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    if (__builtin_expect (found, 0) == __UNKNOWN_10646_CHAR)	      \
	      {								      \
		UNICODE_TAG_HANDLER (ch, 4);				      \
									      \
		/* Illegal character.  */				      \
		STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
	      }								      \
									      \
	    /* It's a CNS 11643 character, adjust it for EUC-TW.  */	      \
	    *outptr++ = '\x8e';						      \
	    *outptr++ += 0xa0;						      \
	    *outptr++ += 0x80;						      \
	    *outptr++ += 0x80;						      \
	  }								      \
      }									      \
									      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
