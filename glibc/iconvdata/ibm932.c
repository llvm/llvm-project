/* Conversion from and to IBM932.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Masahide Washizawa <washi@jp.ibm.com>, 2000.

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
#include <stdbool.h>
#include "ibm932.h"

#define FROM	0
#define TO	1

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME	"IBM932//"
#define FROM_LOOP	from_ibm932
#define TO_LOOP		to_ibm932
#define ONE_DIRECTION	0

/* Definitions of initialization and destructor function.  */
#define DEFINE_INIT	1
#define DEFINE_FINI	1

#define MIN_NEEDED_FROM	1
#define MAX_NEEDED_FROM	2
#define MIN_NEEDED_TO	4

/* First, define the conversion function from IBM-932 to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    const struct gap *rp2 = __ibm932db_to_ucs4_idx;			      \
    uint32_t ch = *inptr;						      \
    uint32_t res;							      \
									      \
    if (__builtin_expect (ch == 0x80, 0)				      \
	|| __builtin_expect (ch == 0xa0, 0)				      \
	|| __builtin_expect (ch == 0xfd, 0)				      \
	|| __builtin_expect (ch == 0xfe, 0)				      \
	|| __builtin_expect (ch == 0xff, 0))				      \
      {									      \
	/* This is an illegal character.  */				      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
									      \
    /* Use the IBM932 table for single byte.  */			      \
    res = __ibm932sb_to_ucs4[ch];					      \
    if (__builtin_expect (res == 0, 0) && ch != 0)			      \
      {									      \
	/* Use the IBM932 table for double byte.  */			      \
	if (__glibc_unlikely (inptr + 1 >= inend))			      \
	  {								      \
	    /* The second character is not available.			      \
	       Store the intermediate result.  */			      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
									      \
	ch = (ch * 0x100) + inptr[1];					      \
	/* ch was less than 0xfd.  */					      \
	assert (ch < 0xfd00);						      \
	while (ch > rp2->end)						      \
	  ++rp2;							      \
									      \
	if (__builtin_expect (ch < rp2->start, 0)			      \
	    || (res = __ibm932db_to_ucs4[ch + rp2->idx],		      \
	    __builtin_expect (res, '\1') == 0 && ch !=0))		      \
	  {								      \
	    /* This is an illegal character.  */			      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (2);				      \
	  }								      \
	else								      \
	  {								      \
	    put32 (outptr, res);					      \
	    outptr += 4;						      \
	    inptr += 2;							      \
	  }								      \
      }									      \
    else								      \
      {									      \
	if (res == 0xa5)						      \
	  res = 0x5c;							      \
	else if (res == 0x203e)						      \
	  res = 0x7e;							      \
	put32 (outptr, res);						      \
	outptr += 4;							      \
	inptr++;							      \
      }									      \
  }
#define LOOP_NEED_FLAGS
#define ONEBYTE_BODY \
  {									      \
    if (c == 0x80 || c == 0xa0 || c >= 0xfd)				      \
      return WEOF;							      \
    uint32_t res = __ibm932sb_to_ucs4[c];				      \
    if (res == 0 && c != 0)						      \
      return WEOF;							      \
    if (res == 0xa5)						              \
      res = 0x5c;							      \
    else if (res == 0x203e)						      \
      res = 0x7e;							      \
    return res;								      \
  }
#include <iconv/loop.c>

/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    const struct gap *rp = __ucs4_to_ibm932sb_idx;			      \
    unsigned char sc;							      \
    uint32_t ch = get32 (inptr);					      \
    bool found = true;							      \
    uint32_t i;								      \
    uint32_t low;							      \
    uint32_t high;							      \
    uint16_t pccode;							      \
									      \
    if (__glibc_unlikely (ch >= 0xffff))				      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
	rp = NULL;							      \
      }									      \
    else								      \
      while (ch > rp->end)						      \
	++rp;								      \
									      \
    /* Use the UCS4 table for single byte.  */				      \
    if (__builtin_expect (rp == NULL, 0)				      \
	|| __builtin_expect (ch < rp->start, 0)				      \
	|| (sc = __ucs4_to_ibm932sb[ch + rp->idx],			      \
	__builtin_expect (sc, '\1') == '\0' && ch != L'\0'))		      \
      {									      \
									      \
	/* Use the UCS4 table for double byte.  */			      \
	found = false;							      \
	low = 0;							      \
	high = (sizeof (__ucs4_to_ibm932db) >> 1)			      \
		/ sizeof (__ucs4_to_ibm932db[0][FROM]);			      \
	pccode = ch;							      \
	if (__glibc_likely (rp != NULL))				      \
	  while (low < high)						      \
	    {								      \
	      i = (low + high) >> 1;					      \
	      if (pccode < __ucs4_to_ibm932db[i][FROM])			      \
		high = i;						      \
	      else if (pccode > __ucs4_to_ibm932db[i][FROM])		      \
		low = i + 1;						      \
	      else 							      \
		{							      \
		  pccode = __ucs4_to_ibm932db[i][TO];			      \
		  found = true;						      \
		  break;						      \
		}							      \
	    }								      \
	if (found) 							      \
	  {								      \
	    if (__glibc_unlikely (outptr + 2 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    *outptr++ = pccode >> 8 & 0xff;				      \
	    *outptr++ = pccode & 0xff;					      \
	  }								      \
	else								      \
	  {								      \
	    /* This is an illegal character.  */			      \
	    STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
	  }								      \
      }									      \
    else								      \
      {									      \
	if (__glibc_unlikely (outptr + 1 > outend))			      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	if (ch == 0x5c)							      \
	  *outptr++ = 0x5c;						      \
	else if (ch == 0x7e)						      \
	  *outptr++ = 0x7e;						      \
	else								      \
	  *outptr++ = sc;						      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>

/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
