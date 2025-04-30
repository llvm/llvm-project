/* Conversion module for ISO-2022-KR.
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
#include <gconv.h>
#include <stdint.h>
#include <string.h>
#include "ksc5601.h"

#include <assert.h>

/* This makes obvious what everybody knows: 0x1b is the Esc character.  */
#define ESC	0x1b

/* The shift sequences for this charset (it does not use ESC).  */
#define SI	0x0f
#define SO	0x0e

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"ISO-2022-KR//"
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define FROM_LOOP		from_iso2022kr_loop
#define TO_LOOP			to_iso2022kr_loop
#define MIN_NEEDED_FROM		1
#define MAX_NEEDED_FROM		4
#define MIN_NEEDED_TO		4
#define MAX_NEEDED_TO		4
#define ONE_DIRECTION		0
#define PREPARE_LOOP \
  int save_set;								      \
  int *setp = &data->__statep->__count;					      \
  if (!FROM_DIRECTION && !data->__internal_use				      \
      && data->__invocation_counter == 0)				      \
    {									      \
      /* Emit the designator sequence.  */				      \
      if (outbuf + 4 > outend)						      \
	return __GCONV_FULL_OUTPUT;					      \
									      \
      *outbuf++ = ESC;							      \
      *outbuf++ = '$';							      \
      *outbuf++ = ')';							      \
      *outbuf++ = 'C';							      \
    }
#define EXTRA_LOOP_ARGS		, setp


/* The COUNT element of the state keeps track of the currently selected
   character set.  The possible values are:  */
enum
{
  ASCII_set = 0,
  KSC5601_set = 8
};


/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  if (data->__statep->__count != ASCII_set)				      \
    {									      \
      if (FROM_DIRECTION)						      \
	{								      \
	  /* It's easy, we don't have to emit anything, we just reset the     \
	     state for the input.  */					      \
	  data->__statep->__count &= 7;					      \
	  data->__statep->__count |= ASCII_set;				      \
	}								      \
      else								      \
	{								      \
	  /* We are not in the initial state.  To switch back we have	      \
	     to emit `SI'.  */						      \
	  if (__glibc_unlikely (outbuf == outend))			      \
	    /* We don't have enough room in the output buffer.  */	      \
	    status = __GCONV_FULL_OUTPUT;				      \
	  else								      \
	    {								      \
	      /* Write out the shift sequence.  */			      \
	      *outbuf++ = SI;						      \
	      data->__statep->__count = ASCII_set;			      \
	    }								      \
	}								      \
    }


/* Since we might have to reset input pointer we must be able to save
   and retore the state.  */
#define SAVE_RESET_STATE(Save) \
  if (Save)								      \
    save_set = *setp;							      \
  else									      \
    *setp = save_set


/* First define the conversion function from ISO-2022-KR to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
									      \
    /* This is a 7bit character set, disallow all 8bit characters.  */	      \
    if (__glibc_unlikely (ch > 0x7f))					      \
      STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
									      \
    /* Recognize escape sequences.  */					      \
    if (__builtin_expect (ch, 0) == ESC)				      \
      {									      \
	/* We don't really have to handle escape sequences since all the      \
	   switching is done using the SI and SO bytes.  But we have to	      \
	   recognize `Esc $ ) C' since this is a kind of flag for this	      \
	   encoding.  We simply ignore it.  */				      \
	if (__builtin_expect (inptr + 2 > inend, 0)			      \
	    || (inptr[1] == '$'						      \
		&& (__builtin_expect (inptr + 3 > inend, 0)		      \
		    || (inptr[2] == ')'					      \
			&& __builtin_expect (inptr + 4 > inend, 0)))))	      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	if (inptr[1] == '$' && inptr[2] == ')' && inptr[3] == 'C')	      \
	  {								      \
	    /* Yeah, yeah, we know this is ISO 2022-KR.  */		      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
      }									      \
    else if (__builtin_expect (ch, 0) == SO)				      \
      {									      \
	/* Switch to use KSC.  */					      \
	++inptr;							      \
	set = KSC5601_set;						      \
	continue;							      \
      }									      \
    else if (__builtin_expect (ch, 0) == SI)				      \
      {									      \
	/* Switch to use ASCII.  */					      \
	++inptr;							      \
	set = ASCII_set;						      \
	continue;							      \
      }									      \
									      \
    if (set == ASCII_set)						      \
      {									      \
	/* Almost done, just advance the input pointer.  */		      \
	++inptr;							      \
      }									      \
    else								      \
      {									      \
	assert (set == KSC5601_set);					      \
									      \
	/* Use the KSC 5601 table.  */					      \
	ch = ksc5601_to_ucs4 (&inptr, inend - inptr, 0);		      \
									      \
	if (__glibc_unlikely (ch == 0))					      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	else if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	  {								      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *setp
#define INIT_PARAMS		int set = *setp
#define UPDATE_PARAMS		*setp = set
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
    /* First see whether we can write the character using the currently	      \
       selected character set.  */					      \
    if (ch < 0x80)							      \
      {									      \
	if (set != ASCII_set)						      \
	  {								      \
	    *outptr++ = SI;						      \
	    set = ASCII_set;						      \
	    if (__glibc_unlikely (outptr == outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	  }								      \
 									      \
	*outptr++ = ch;							      \
      }									      \
    else								      \
      {									      \
	unsigned char buf[2];						      \
	/* Fake initialization to keep gcc quiet.  */			      \
	asm ("" : "=m" (buf));						      \
									      \
	size_t written = ucs4_to_ksc5601 (ch, buf, 2);			      \
	if (__builtin_expect (written, 0) == __UNKNOWN_10646_CHAR)	      \
	  {								      \
	    UNICODE_TAG_HANDLER (ch, 4);				      \
									      \
	    /* Illegal character.  */					      \
	    STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
	  }								      \
	else								      \
	  {								      \
	    assert (written == 2);					      \
									      \
	    /* We use KSC 5601.  */					      \
	    if (set != KSC5601_set)					      \
	      {								      \
		*outptr++ = SO;						      \
		set = KSC5601_set;					      \
	      }								      \
									      \
	    if (__glibc_unlikely (outptr + 2 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    *outptr++ = buf[0];						      \
	    *outptr++ = buf[1];						      \
	  }								      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *setp
#define INIT_PARAMS		int set = *setp
#define REINIT_PARAMS		set = *setp
#define UPDATE_PARAMS		*setp = set
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
