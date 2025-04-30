/* Conversion from and to IBM933.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Masahide Washizawa <washi@yamato.ibm.co.jp>, 2000.

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

/* IBM933 is designed for the representation of Korean using a stateful
   EBCDIC encoding scheme.  It is also known as CCSID 933 or CP933. See:
   https://www-01.ibm.com/software/globalization/ccsid/ccsid933.html */

#include <dlfcn.h>
#include <stdint.h>
#include <wchar.h>
#include <byteswap.h>
#include "ibm933.h"

/* The shift sequences for this charset (it does not use ESC).  */
#define SI 		0x0F  /* Shift In, host code to turn DBCS off.  */
#define SO 		0x0E  /* Shift Out, host code to turn DBCS on.  */

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME	"IBM933//"
#define FROM_LOOP	from_ibm933
#define TO_LOOP		to_ibm933
#define ONE_DIRECTION			0
#define FROM_LOOP_MIN_NEEDED_FROM	1
#define FROM_LOOP_MAX_NEEDED_FROM	2
#define FROM_LOOP_MIN_NEEDED_TO		4
#define FROM_LOOP_MAX_NEEDED_TO		4
#define TO_LOOP_MIN_NEEDED_FROM		4
#define TO_LOOP_MAX_NEEDED_FROM		4
#define TO_LOOP_MIN_NEEDED_TO		1
#define TO_LOOP_MAX_NEEDED_TO		3
#define PREPARE_LOOP \
  int save_curcs;							      \
  int *curcsp = &data->__statep->__count;
#define EXTRA_LOOP_ARGS		, curcsp

/* Definitions of initialization and destructor function.  */
#define DEFINE_INIT	1
#define DEFINE_FINI	1


/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  if ((data->__statep->__count & ~7) != sb)				      \
    {									      \
      if (FROM_DIRECTION)						      \
	data->__statep->__count &= 7;					      \
      else								      \
	{								      \
	  /* We are not in the initial state.  To switch back we have	      \
	     to emit `SI'.  */						      \
	  if (__glibc_unlikely (outbuf >= outend))			      \
	    /* We don't have enough room in the output buffer.  */	      \
	    status = __GCONV_FULL_OUTPUT;				      \
	  else								      \
	    {								      \
	      /* Write out the shift sequence.  */			      \
	      *outbuf++ = SI;						      \
	      data->__statep->__count &= 7;				      \
	    }								      \
	}								      \
    }


/* Since we might have to reset input pointer we must be able to save
   and retore the state.  */
#define SAVE_RESET_STATE(Save) \
  if (Save)								      \
    save_curcs = *curcsp;						      \
  else									      \
    *curcsp = save_curcs


/* Current codeset type.  */
enum
{
  sb = 0,
  db = 64
};

/* First, define the conversion function from IBM-933 to UCS4.  */
#define MIN_NEEDED_INPUT	FROM_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	FROM_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	FROM_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	FROM_LOOP_MAX_NEEDED_TO
#define LOOPFCT 		FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
    uint32_t res;							      \
									      \
    if (__builtin_expect (ch, 0) == SO)					      \
      {									      \
	/* Shift OUT, change to DBCS converter (redundant escape okay).  */   \
	curcs = db;							      \
	++inptr;							      \
	continue;							      \
      }									      \
    else if (__builtin_expect (ch, 0) == SI)				      \
      {									      \
	/* Shift IN, change to SBCS converter (redundant escape okay).  */    \
	curcs = sb;							      \
	++inptr;							      \
	continue;							      \
      }									      \
									      \
    if (curcs == sb)							      \
      {									      \
	/* Use the IBM933 table for single byte.  */			      \
	res = __ibm933sb_to_ucs4[ch];					      \
	if (__builtin_expect (res, L'\1') == L'\0' && ch != '\0')	      \
	  {								      \
	    /* This is an illegal character.  */			      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
	else								      \
	  {								      \
	    put32 (outptr, res);					      \
	    outptr += 4;						      \
	  }								      \
	++inptr;							      \
      }									      \
    else								      \
      {									      \
	const struct gap *rp2 = __ibm933db_to_ucs4_idx;			      \
									      \
	assert (curcs == db);						      \
									      \
	/* Use the IBM933 table for double byte.  */			      \
	if (__glibc_unlikely (inptr + 1 >= inend))			      \
	  {								      \
	    /* The second character is not available.  Store the	      \
	       intermediate result. */					      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
									      \
	ch = (ch * 0x100) + inptr[1];					      \
	while (ch > rp2->end)						      \
	  ++rp2;							      \
									      \
	if (__builtin_expect (rp2->start == 0xffff, 0)			      \
	    || __builtin_expect (ch < rp2->start, 0)			      \
	    || (res = __ibm933db_to_ucs4[ch + rp2->idx],		      \
		__builtin_expect (res, L'\1') == L'\0' && ch != '\0'))	      \
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
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *curcsp
#define INIT_PARAMS		int curcs = *curcsp & ~7
#define UPDATE_PARAMS		*curcsp = curcs
#include <iconv/loop.c>

/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	TO_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	TO_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	TO_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	TO_LOOP_MAX_NEEDED_TO
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch = get32 (inptr);					      \
    const struct gap *rp1 = __ucs4_to_ibm933sb_idx;			      \
    const struct gap *rp2 = __ucs4_to_ibm933db_idx;			      \
									      \
    if (__glibc_unlikely (ch >= 0xffff))				      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
									      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
									      \
    while (ch > rp1->end)						      \
      ++rp1;								      \
									      \
    /* Use the UCS4 table for single byte.  */				      \
    unsigned char sbconv;						      \
    if (__builtin_expect (ch < rp1->start, 0)				      \
	|| (sbconv = __ucs4_to_ibm933sb[ch + rp1->idx],			      \
	    __builtin_expect (sbconv, L'\1') == L'\0' && ch != '\0'))	      \
      {									      \
	/* Use the UCS4 table for double byte.  */			      \
	while (ch > rp2->end)						      \
	  ++rp2;							      \
									      \
	const char *cp;							      \
	if (__builtin_expect (ch < rp2->start, 0)			      \
	    || (cp = __ucs4_to_ibm933db[ch + rp2->idx],			      \
		__builtin_expect (cp[0], L'\1')==L'\0' && ch != '\0'))	      \
	  {								      \
	    /* This is an illegal character.  */			      \
	    STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
	  }								      \
	else								      \
	  {								      \
	    if (curcs == sb)						      \
	      {								      \
		if (__glibc_unlikely (outptr + 1 > outend))		      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
		*outptr++ = SO;						      \
		curcs = db;						      \
	      }								      \
									      \
	    if (__glibc_unlikely (outptr + 2 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    *outptr++ = cp[0];						      \
	    *outptr++ = cp[1];						      \
	  }								      \
      }									      \
    else								      \
      {									      \
	if (curcs == db)						      \
	  {								      \
	    if (__glibc_unlikely (outptr + 1 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    *outptr++ = SI;						      \
	    curcs = sb;							      \
	  }								      \
									      \
	if (__glibc_unlikely (outptr + 1 > outend))			      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	*outptr++ = sbconv;						      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *curcsp
#define INIT_PARAMS		int curcs = *curcsp & ~7
#define REINIT_PARAMS		curcs = *curcsp & ~7
#define UPDATE_PARAMS		*curcsp = curcs
#include <iconv/loop.c>

/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
