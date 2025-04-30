/* Conversion module for ISO-2022-CN.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1999.

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
#include "gb2312.h"
#include "cns11643l1.h"
#include "cns11643l2.h"

#include <assert.h>

/* This makes obvious what everybody knows: 0x1b is the Esc character.  */
#define ESC	0x1b

/* We have single-byte shift-in and shift-out sequences, and the single
   shift sequence SS2 which replaces the SS2 designation for the next
   two bytes.  */
#define SI	0x0f
#define SO	0x0e
#define SS2_0	ESC
#define SS2_1	0x4e

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"ISO-2022-CN//"
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define FROM_LOOP		from_iso2022cn_loop
#define TO_LOOP			to_iso2022cn_loop
#define ONE_DIRECTION			0
#define FROM_LOOP_MIN_NEEDED_FROM	1
#define FROM_LOOP_MAX_NEEDED_FROM	4
#define FROM_LOOP_MIN_NEEDED_TO		4
#define FROM_LOOP_MAX_NEEDED_TO		4
#define TO_LOOP_MIN_NEEDED_FROM		4
#define TO_LOOP_MAX_NEEDED_FROM		4
#define TO_LOOP_MIN_NEEDED_TO		1
#define TO_LOOP_MAX_NEEDED_TO		6
#define PREPARE_LOOP \
  int save_set;								      \
  int *setp = &data->__statep->__count;
#define EXTRA_LOOP_ARGS		, setp


/* The COUNT element of the state keeps track of the currently selected
   character set.  The possible values are:  */
enum
{
  ASCII_set = 0,
  GB2312_set = 8,
  CNS11643_1_set = 16,
  CNS11643_2_set = 24,
  CURRENT_SEL_MASK = 24,
  GB2312_ann = 32,
  CNS11643_1_ann = 64,
  CNS11643_2_ann = 128,
  CURRENT_ANN_MASK = 224
};


/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  if (data->__statep->__count != ASCII_set)				      \
    {									      \
      if (FROM_DIRECTION)						      \
	/* It's easy, we don't have to emit anything, we just reset the	      \
	   state for the input.  */					      \
	data->__statep->__count = ASCII_set;				      \
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


/* First define the conversion function from ISO-2022-CN to UCS4.  */
#define MIN_NEEDED_INPUT	FROM_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	FROM_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	FROM_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	FROM_LOOP_MAX_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
									      \
    /* This is a 7bit character set, disallow all 8bit characters.  */	      \
    if (__glibc_unlikely (ch >= 0x7f))					      \
      STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
									      \
    /* Recognize escape sequences.  */					      \
    if (__builtin_expect (ch, 0) == ESC)				      \
      {									      \
	/* There are two kinds of escape sequences we have to handle:	      \
	   - those announcing the use of GB and CNS characters on the	      \
	     line; we can simply ignore them				      \
	   - the initial byte of the SS2 sequence.			      \
	*/								      \
	if (__builtin_expect (inptr + 2 > inend, 0)			      \
	    || (inptr[1] == '$'						      \
		&& (__builtin_expect (inptr + 3 > inend, 0)		      \
		    || (inptr[2] == ')'					      \
			&& __builtin_expect (inptr + 4 > inend, 0))	      \
		    || (inptr[2] == '*'					      \
			&& __builtin_expect (inptr + 4 > inend, 0))))	      \
	    || (inptr[1] == SS2_1					      \
		&& __builtin_expect (inptr + 4 > inend, 0)))		      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	if (inptr[1] == '$'						      \
	    && ((inptr[2] == ')' && (inptr[3] == 'A' || inptr[3] == 'G'))     \
		|| (inptr[2] == '*' && inptr[3] == 'H')))		      \
	  {								      \
	    /* OK, we accept those character sets.  */			      \
	    if (inptr[3] == 'A')					      \
	      ann = GB2312_ann;						      \
	    else if (inptr[3] == 'G')					      \
	      ann = CNS11643_1_ann;					      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
      }									      \
    else if (__builtin_expect (ch, 0) == SO)				      \
      {									      \
	/* Switch to use GB2312 or CNS 11643 plane 1, depending on which      \
	   S0 designation came last.  The only problem is what to do with     \
	   faulty input files where no designator came.			      \
	   XXX For now I'll default to use GB2312.  If this is not the	      \
	   best behaviour (e.g., we should flag an error) let me know.  */    \
	++inptr;							      \
	set = ann == CNS11643_1_ann ? CNS11643_1_set : GB2312_set;	      \
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
    if (__builtin_expect (ch, 0) == ESC && inptr[1] == SS2_1)		      \
      {									      \
	/* This is a character from CNS 11643 plane 2.			      \
	   XXX We could test here whether the use of this character	      \
	   set was announced.  */					      \
	inptr += 2;							      \
	ch = cns11643l2_to_ucs4 (&inptr, 2, 0);				      \
	if (__builtin_expect (ch, 0) == __UNKNOWN_10646_CHAR)		      \
	  {								      \
	    inptr -= 2;							      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (2);				      \
	  }								      \
      }									      \
    else if (set == ASCII_set)						      \
      {									      \
	/* Almost done, just advance the input pointer.  */		      \
	++inptr;							      \
      }									      \
    else								      \
      {									      \
	/* That's pretty easy, we have a dedicated functions for this.  */    \
	if (set == GB2312_set)						      \
	  ch = gb2312_to_ucs4 (&inptr, inend - inptr, 0);		      \
	else								      \
	  {								      \
	    assert (set == CNS11643_1_set);				      \
	    ch = cns11643l1_to_ucs4 (&inptr, inend - inptr, 0);		      \
	  }								      \
									      \
	if (__builtin_expect (ch, 1) == 0)				      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	else if (__builtin_expect (ch, 1) == __UNKNOWN_10646_CHAR)	      \
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
#define INIT_PARAMS		int set = *setp & CURRENT_SEL_MASK; \
				int ann = *setp & CURRENT_ANN_MASK
#define UPDATE_PARAMS		*setp = set | ann
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
									      \
	/* At the end of the line we have to clear the `ann' flags since      \
	   every line must contain this information again.  */		      \
	if (ch == L'\n')						      \
	  ann = 0;							      \
      }									      \
    else								      \
      {									      \
	unsigned char buf[2];						      \
	/* Fake initialization to keep gcc quiet.  */			      \
	asm ("" : "=m" (buf));						      \
									      \
	int used;							      \
	size_t written = 0;						      \
									      \
	if (set == GB2312_set || (ann & CNS11643_1_ann) == 0)		      \
	  {								      \
	    written = ucs4_to_gb2312 (ch, buf, 2);			      \
	    used = GB2312_set;						      \
	  }								      \
	else								      \
	  {								      \
	    written = ucs4_to_cns11643l1 (ch, buf, 2);			      \
	    used = CNS11643_1_set;					      \
	  }								      \
									      \
	if (written == __UNKNOWN_10646_CHAR)				      \
	  {								      \
	    /* Cannot convert it using the currently selected SO set.	      \
	       Next try the SS2 set.  */				      \
	    written = ucs4_to_cns11643l2 (ch, buf, 2);			      \
	    if (written != __UNKNOWN_10646_CHAR)			      \
	      /* Yep, that worked.  */					      \
	      used = CNS11643_2_set;					      \
	    else							      \
	      {								      \
		/* Well, see whether we have to change the SO set.  */	      \
		if (used == GB2312_set)					      \
		  written = ucs4_to_cns11643l1 (ch, buf, 2);		      \
		else							      \
		  written = ucs4_to_gb2312 (ch, buf, 2);		      \
									      \
		if (__builtin_expect (written, 0) != __UNKNOWN_10646_CHAR)    \
		  /* Oh well, then switch SO.  */			      \
		  used = GB2312_set + CNS11643_1_set - used;		      \
		else							      \
		  {							      \
		    UNICODE_TAG_HANDLER (ch, 4);			      \
									      \
		    /* Even this does not work.  Error.  */		      \
		    STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
		  }							      \
	      }								      \
	  }								      \
	assert (written == 2);						      \
									      \
	/* See whether we have to emit an escape sequence.  */		      \
	if (set != used)						      \
	  {								      \
	    /* First see whether we announced that we use this		      \
	       character set.  */					      \
	    if ((ann & (16 << (used >> 3))) == 0)			      \
	      {								      \
		const char *escseq;					      \
									      \
		if (__glibc_unlikely (outptr + 4 > outend))		      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
									      \
		assert ((used >> 3) >= 1 && (used >> 3) <= 3);		      \
		escseq = ")A)G*H" + ((used >> 3) - 1) * 2;		      \
		*outptr++ = ESC;					      \
		*outptr++ = '$';					      \
		*outptr++ = *escseq++;					      \
		*outptr++ = *escseq++;					      \
									      \
		if (used == GB2312_set)					      \
		  ann = (ann & CNS11643_2_ann) | GB2312_ann;		      \
		else if (used == CNS11643_1_set)			      \
		  ann = (ann & CNS11643_2_ann) | CNS11643_1_ann;	      \
		else							      \
		  ann |= CNS11643_2_ann;				      \
	      }								      \
									      \
	    if (used == CNS11643_2_set)					      \
	      {								      \
		if (__glibc_unlikely (outptr + 2 > outend))		      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
		*outptr++ = SS2_0;					      \
		*outptr++ = SS2_1;					      \
	      }								      \
	    else							      \
	      {								      \
		/* We only have to emit something is currently ASCII is	      \
		   selected.  Otherwise we are switching within the	      \
		   SO charset.  */					      \
		if (set == ASCII_set)					      \
		  {							      \
		    if (__glibc_unlikely (outptr + 1 > outend))		      \
		      {							      \
			result = __GCONV_FULL_OUTPUT;			      \
			break;						      \
		      }							      \
		    *outptr++ = SO;					      \
		  }							      \
	      }								      \
									      \
	    /* Always test the length here since we have used up all the      \
	       guaranteed output buffer slots.  */			      \
	    if (__glibc_unlikely (outptr + 2 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	  }								      \
	else if (__glibc_unlikely (outptr + 2 > outend))		      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
									      \
	*outptr++ = buf[0];						      \
	*outptr++ = buf[1];						      \
	set = used;							      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *setp
#define INIT_PARAMS		int set = *setp & CURRENT_SEL_MASK; \
				int ann = *setp & CURRENT_ANN_MASK
#define REINIT_PARAMS		do					      \
				  {					      \
				    set = *setp & CURRENT_SEL_MASK;	      \
				    ann = *setp & CURRENT_ANN_MASK;	      \
				  }					      \
				while (0)
#define UPDATE_PARAMS		*setp = set | ann
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
