/* Conversion module for ISO-2022-CN-EXT.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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
#include <stdlib.h>
#include <string.h>
#include "gb2312.h"
#include "iso-ir-165.h"
#include "cns11643.h"
#include "cns11643l1.h"
#include "cns11643l2.h"
#include <libc-diag.h>

#include <assert.h>

/* This makes obvious what everybody knows: 0x1b is the Esc character.  */
#define ESC	0x1b

/* We have single-byte shift-in and shift-out sequences, and the single
   shift sequences SS2 and SS3 which replaces the SS2/SS3 designation for
   the next two bytes.  */
#define SI	0x0f
#define SO	0x0e
#define SS2_0	ESC
#define SS2_1	0x4e
#define SS3_0	ESC
#define SS3_1	0x4f

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"ISO-2022-CN-EXT//"
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define ONE_DIRECTION		0
#define FROM_LOOP		from_iso2022cn_ext_loop
#define TO_LOOP			to_iso2022cn_ext_loop
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


/* The charsets GB/T 12345-90, GB 7589-87, GB/T 13131-9X, GB 7590-87,
   and GB/T 13132-9X are not registered to the best of my knowledge and
   therefore have no escape sequence assigned.  We cannot handle them
   for this reason.  Tell the implementation about this.  */
#define X12345	'\0'
#define X7589	'\0'
#define X13131	'\0'
#define X7590	'\0'
#define X13132	'\0'


/* The COUNT element of the state keeps track of the currently selected
   character set.  The possible values are:  */
enum
{
  ASCII_set = 0,
  GB2312_set,
  GB12345_set,
  CNS11643_1_set,
  ISO_IR_165_set,
  SO_mask = 7,

  GB7589_set = 1 << 3,
  GB13131_set = 2 << 3,
  CNS11643_2_set = 3 << 3,
  SS2_mask = 3 << 3,

  GB7590_set = 1 << 5,
  GB13132_set = 2 << 5,
  CNS11643_3_set = 3 << 5,
  CNS11643_4_set = 4 << 5,
  CNS11643_5_set = 5 << 5,
  CNS11643_6_set = 6 << 5,
  CNS11643_7_set = 7 << 5,
  SS3_mask = 7 << 5,

#define CURRENT_MASK (SO_mask | SS2_mask | SS3_mask)

  GB2312_ann = 1 << 8,
  GB12345_ann = 2 << 8,
  CNS11643_1_ann = 3 << 8,
  ISO_IR_165_ann = 4 << 8,
  SO_ann = 7 << 8,

  GB7589_ann = 1 << 11,
  GB13131_ann = 2 << 11,
  CNS11643_2_ann = 3 << 11,
  SS2_ann = 3 << 11,

  GB7590_ann = 1 << 13,
  GB13132_ann = 2 << 13,
  CNS11643_3_ann = 3 << 13,
  CNS11643_4_ann = 4 << 13,
  CNS11643_5_ann = 5 << 13,
  CNS11643_6_ann = 6 << 13,
  CNS11643_7_ann = 7 << 13,
  SS3_ann = 7 << 13
};


/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  if (data->__statep->__count >> 3 != ASCII_set)			      \
    {									      \
      if (FROM_DIRECTION)						      \
	/* It's easy, we don't have to emit anything, we just reset the	      \
	   state for the input.  */					      \
	data->__statep->__count = ASCII_set << 3;			      \
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
	      if (data->__flags & __GCONV_IS_LAST)			      \
		*irreversible += 1;					      \
	      data->__statep->__count = ASCII_set << 3;			      \
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
    if (ch > 0x7f)							      \
      STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
									      \
    /* Recognize escape sequences.  */					      \
    if (ch == ESC)							      \
      {									      \
	/* There are three kinds of escape sequences we have to handle:	      \
	   - those announcing the use of GB and CNS characters on the	      \
	     line; we can simply ignore them				      \
	   - the initial byte of the SS2 sequence.			      \
	   - the initial byte of the SS3 sequence.			      \
	*/								      \
	if (inptr + 2 > inend						      \
	    || (inptr[1] == '$'						      \
		&& (inptr + 3 > inend					      \
		    || (inptr[2] == ')' && inptr + 4 > inend)		      \
		    || (inptr[2] == '*' && inptr + 4 > inend)		      \
		    || (inptr[2] == '+' && inptr + 4 > inend)))		      \
	    || (inptr[1] == SS2_1 && inptr + 4 > inend)			      \
	    || (inptr[1] == SS3_1 && inptr + 4 > inend))		      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	if (inptr[1] == '$'						      \
	    && ((inptr[2] == ')'					      \
		 && (inptr[3] == 'A'					      \
		     || (X12345 != '\0' && inptr[3] == X12345)		      \
		     || inptr[3] == 'E' || inptr[3] == 'G'))		      \
		|| (inptr[2] == '*'					      \
		    && ((X7589 != '\0' && inptr[3] == X7589)		      \
			|| (X13131 != '\0' && inptr[3] == X13131)	      \
			|| inptr[3] == 'H'))				      \
		|| (inptr[2] == '+'					      \
		    && ((X7590 != '\0' && inptr[3] == X7590)		      \
			|| (X13132 != '\0' && inptr[3] == X13132)	      \
			|| inptr[3] == 'I' || inptr[3] == 'J'		      \
			|| inptr[3] == 'K' || inptr[3] == 'L'		      \
			|| inptr[3] == 'M'))))				      \
	  {								      \
	    /* OK, we accept those character sets.  */			      \
	    if (inptr[3] == 'A')					      \
	      ann = (ann & ~SO_ann) | GB2312_ann;			      \
	    else if (inptr[3] == 'G')					      \
	      ann = (ann & ~SO_ann) | CNS11643_1_ann;			      \
	    else if (inptr[3] == 'E')					      \
	      ann = (ann & ~SO_ann) | ISO_IR_165_ann;			      \
	    else if (X12345 != '\0' && inptr[3] == X12345)		      \
	      ann = (ann & ~SO_ann) | GB12345_ann;			      \
	    else if (inptr[3] == 'H')					      \
	      ann = (ann & ~SS2_ann) | CNS11643_2_ann;			      \
	    else if (X7589 != '\0' && inptr[3] == X7589)		      \
	      ann = (ann & ~SS2_ann) | GB7589_ann;			      \
	    else if (X13131 != '\0' && inptr[3] == X13131)		      \
	      ann = (ann & ~SS2_ann) | GB13131_ann;			      \
	    else if (inptr[3] == 'I')					      \
	      ann = (ann & ~SS3_ann) | CNS11643_3_ann;			      \
	    else if (inptr[3] == 'J')					      \
	      ann = (ann & ~SS3_ann) | CNS11643_4_ann;			      \
	    else if (inptr[3] == 'K')					      \
	      ann = (ann & ~SS3_ann) | CNS11643_5_ann;			      \
	    else if (inptr[3] == 'L')					      \
	      ann = (ann & ~SS3_ann) | CNS11643_6_ann;			      \
	    else if (inptr[3] == 'M')					      \
	      ann = (ann & ~SS3_ann) | CNS11643_7_ann;			      \
	    else if (X7590 != '\0' && inptr[3] == X7590)		      \
	      ann = (ann & ~SS3_ann) | GB7590_ann;			      \
	    else if (X13132 != '\0' && inptr[3] == X13132)		      \
	      ann = (ann & ~SS3_ann) | GB13132_ann;			      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
      }									      \
    else if (ch == SO)							      \
      {									      \
	/* Switch to use GB2312, GB12345, CNS 11643 plane 1, or ISO-IR-165,   \
	   depending on which S0 designation came last.  The only problem     \
	   is what to do with faulty input files where no designator came.    \
	   XXX For now I'll default to use GB2312.  If this is not the	      \
	   best behavior (e.g., we should flag an error) let me know.  */     \
	++inptr;							      \
	if ((ann & SO_ann) != 0)					      \
	  switch (ann & SO_ann)						      \
	    {								      \
	    case GB2312_ann:						      \
	      set = GB2312_set;						      \
	      break;							      \
	    case GB12345_ann:						      \
	      set = GB12345_set;					      \
	      break;							      \
	    case CNS11643_1_ann:					      \
	      set = CNS11643_1_set;					      \
	      break;							      \
	    case ISO_IR_165_ann:					      \
	      set = ISO_IR_165_set;					      \
	      break;							      \
	    default:							      \
	      abort ();							      \
	    }								      \
	else								      \
	  {								      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
	continue;							      \
      }									      \
    else if (ch == SI)							      \
      {									      \
	/* Switch to use ASCII.  */					      \
	++inptr;							      \
	set = ASCII_set;						      \
	continue;							      \
      }									      \
									      \
    if (ch == ESC && inptr[1] == SS2_1)					      \
      {									      \
	/* This is a character from CNS 11643 plane 2.			      \
	   XXX We could test here whether the use of this character	      \
	   set was announced.						      \
	   XXX Currently GB7589 and GB13131 are not supported.  */	      \
	inptr += 2;							      \
	ch = cns11643l2_to_ucs4 (&inptr, 2, 0);				      \
	if (ch == __UNKNOWN_10646_CHAR)					      \
	  STANDARD_FROM_LOOP_ERR_HANDLER (2);				      \
      }									      \
    /* Note that we can assume here that at least 4 bytes are available if    \
       the first byte is ESC since otherwise the first if would have been     \
       true.  */							      \
    else if (ch == ESC && inptr[1] == SS3_1)				      \
      {									      \
	/* This is a character from CNS 11643 plane 3 or higher.	      \
	   XXX Currently GB7590 and GB13132 are not supported.  */	      \
	unsigned char buf[3];						      \
	const unsigned char *tmp = buf;					      \
									      \
	buf[1] = inptr[2];						      \
	buf[2] = inptr[3];						      \
	switch (ann & SS3_ann)						      \
	  {								      \
	  case CNS11643_3_ann:						      \
	    buf[0] = 0x23;						      \
	    ch = cns11643_to_ucs4 (&tmp, 3, 0);				      \
	    break;							      \
	  case CNS11643_4_ann:						      \
	    buf[0] = 0x24;						      \
	    ch = cns11643_to_ucs4 (&tmp, 3, 0);				      \
	    break;							      \
	  case CNS11643_5_ann:						      \
	    buf[0] = 0x25;						      \
	    ch = cns11643_to_ucs4 (&tmp, 3, 0);				      \
	    break;							      \
	  case CNS11643_6_ann:						      \
	    buf[0] = 0x26;						      \
	    ch = cns11643_to_ucs4 (&tmp, 3, 0);				      \
	    break;							      \
	  case CNS11643_7_ann:						      \
	    buf[0] = 0x27;						      \
	    ch = cns11643_to_ucs4 (&tmp, 3, 0);				      \
	    break;							      \
	  default:							      \
	    /* XXX Currently GB7590 and GB13132 are not supported.  */	      \
	    ch = __UNKNOWN_10646_CHAR;					      \
	    break;							      \
	  }								      \
	if (ch == __UNKNOWN_10646_CHAR)					      \
	  {								      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (4);				      \
	  }								      \
	assert (tmp == buf + 3);					      \
	inptr += 4;							      \
      }									      \
    else if (set == ASCII_set)						      \
      {									      \
	/* Almost done, just advance the input pointer.  */		      \
	++inptr;							      \
      }									      \
    else								      \
      {									      \
	/* That's pretty easy, we have a dedicated functions for this.  */    \
	if (inend - inptr < 2)						      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	if (set == GB2312_set)						      \
	  ch = gb2312_to_ucs4 (&inptr, inend - inptr, 0);		      \
	else if (set == ISO_IR_165_set)					      \
	  ch = isoir165_to_ucs4 (&inptr, inend - inptr);		      \
	else								      \
	  {								      \
	    assert (set == CNS11643_1_set);				      \
	    ch = cns11643l1_to_ucs4 (&inptr, inend - inptr, 0);		      \
	  }								      \
									      \
	if (ch == 0)							      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
	else if (ch == __UNKNOWN_10646_CHAR)				      \
	  {								      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (2);				      \
	  }								      \
      }									      \
									      \
    *((uint32_t *) outptr) = ch;					      \
    outptr += sizeof (uint32_t);					      \
  }
#define EXTRA_LOOP_DECLS	, int *setp
#define INIT_PARAMS		int set = (*setp >> 3) & CURRENT_MASK; \
				int ann = (*setp >> 3) & ~CURRENT_MASK
#define UPDATE_PARAMS		*setp = (set | ann) << 3
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	TO_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	TO_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	TO_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	TO_LOOP_MAX_NEEDED_TO
#define LOOPFCT			TO_LOOP
/* With GCC 5.3 when compiling with -Os the compiler emits a warning
   that buf[0] and buf[1] may be used uninitialized.  This can only
   happen in the case where tmpbuf[3] is used, and in that case the
   write to the tmpbuf[1] and tmpbuf[2] was assured because
   ucs4_to_cns11643 would have filled in those entries.  The difficulty
   is in getting the compiler to see this logic because tmpbuf[0] is
   involved in determining the code page and is the indicator that
   tmpbuf[2] is initialized.  */
DIAG_PUSH_NEEDS_COMMENT;
DIAG_IGNORE_Os_NEEDS_COMMENT (5, "-Wmaybe-uninitialized");
#define BODY \
  {									      \
    uint32_t ch;							      \
    size_t written = 0;							      \
									      \
    ch = *((const uint32_t *) inptr);					      \
									      \
    /* First see whether we can write the character using the currently	      \
       selected character set.  */					      \
    if (ch < 0x80)							      \
      {									      \
	if (set != ASCII_set)						      \
	  {								      \
	    *outptr++ = SI;						      \
	    set = ASCII_set;						      \
	    if (outptr == outend)					      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	  }								      \
									      \
	*outptr++ = ch;							      \
	written = 1;							      \
									      \
	/* At the end of the line we have to clear the `ann' flags since      \
	   every line must contain this information again.  */		      \
	if (ch == L'\n')						      \
	  ann = 0;							      \
      }									      \
    else								      \
      {									      \
	unsigned char buf[2] = { 0, 0 };				      \
	int used;							      \
									      \
	if (set == GB2312_set || ((ann & SO_ann) != CNS11643_1_ann	      \
				  && (ann & SO_ann) != ISO_IR_165_ann))	      \
	  {								      \
	    written = ucs4_to_gb2312 (ch, buf, 2);			      \
	    used = GB2312_set;						      \
	  }								      \
	else if (set == ISO_IR_165_set || (ann & SO_ann) == ISO_IR_165_set)   \
	  {								      \
	    written = ucs4_to_isoir165 (ch, buf, 2);			      \
	    used = ISO_IR_165_set;					      \
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
		unsigned char tmpbuf[3];				      \
									      \
		switch (0)						      \
		  {							      \
		  default:						      \
		    /* Well, see whether we have to change the SO set.  */    \
									      \
		    if (used != GB2312_set)				      \
		      {							      \
			written = ucs4_to_gb2312 (ch, buf, 2);		      \
			if (written != __UNKNOWN_10646_CHAR)		      \
			  {						      \
			    used = GB2312_set;				      \
			    break;					      \
			  }						      \
		      }							      \
									      \
		    if (used != ISO_IR_165_set)				      \
		      {							      \
			written = ucs4_to_isoir165 (ch, buf, 2);	      \
			if (written != __UNKNOWN_10646_CHAR)		      \
			  {						      \
			    used = ISO_IR_165_set;			      \
			    break;					      \
			  }						      \
		      }							      \
									      \
		    if (used != CNS11643_1_set)				      \
		      {							      \
			written = ucs4_to_cns11643l1 (ch, buf, 2);	      \
			if (written != __UNKNOWN_10646_CHAR)		      \
			  {						      \
			    used = CNS11643_1_set;			      \
			    break;					      \
			  }						      \
		      }							      \
									      \
		    written = ucs4_to_cns11643 (ch, tmpbuf, 3);		      \
		    if (written == 3 && tmpbuf[0] >= 3 && tmpbuf[0] <= 7)     \
		      {							      \
			buf[0] = tmpbuf[1];				      \
			buf[1] = tmpbuf[2];				      \
			switch (tmpbuf[0])				      \
			  {						      \
			  case 3:					      \
			    used = CNS11643_3_set;			      \
			    break;					      \
			  case 4:					      \
			    used = CNS11643_4_set;			      \
			    break;					      \
			  case 5:					      \
			    used = CNS11643_5_set;			      \
			    break;					      \
			  case 6:					      \
			    used = CNS11643_6_set;			      \
			    break;					      \
			  case 7:					      \
			    used = CNS11643_7_set;			      \
			    break;					      \
			  default:					      \
			    abort ();					      \
			  }						      \
			written = 2;					      \
			break;						      \
		      }							      \
									      \
		    /* XXX Currently GB7590 and GB13132 are not supported.  */\
									      \
		    /* Even this does not work.  Error.  */		      \
		    used = ASCII_set;					      \
		  }							      \
		if (used == ASCII_set)					      \
		  {							      \
		    UNICODE_TAG_HANDLER (ch, 4);			      \
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
	    if ((used & SO_mask) != 0 && (ann & SO_ann) != (used << 8))	      \
	      {								      \
		const char *escseq;					      \
									      \
		if (outptr + 4 > outend)				      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
									      \
		assert (used >= 1 && used <= 4);			      \
		escseq = ")A\0\0)G)E" + (used - 1) * 2;			      \
		*outptr++ = ESC;					      \
		*outptr++ = '$';					      \
		*outptr++ = *escseq++;					      \
		*outptr++ = *escseq++;					      \
									      \
		ann = (ann & ~SO_ann) | (used << 8);			      \
	      }								      \
	    else if ((used & SS2_mask) != 0 && (ann & SS2_ann) != (used << 8))\
	      {								      \
		const char *escseq;					      \
									      \
		assert (used == CNS11643_2_set); /* XXX */		      \
		escseq = "*H";						      \
		*outptr++ = ESC;					      \
		*outptr++ = '$';					      \
		*outptr++ = *escseq++;					      \
		*outptr++ = *escseq++;					      \
									      \
		ann = (ann & ~SS2_ann) | (used << 8);			      \
	      }								      \
	    else if ((used & SS3_mask) != 0 && (ann & SS3_ann) != (used << 8))\
	      {								      \
		const char *escseq;					      \
									      \
		assert ((used >> 5) >= 3 && (used >> 5) <= 7);		      \
		escseq = "+I+J+K+L+M" + ((used >> 5) - 3) * 2;		      \
		*outptr++ = ESC;					      \
		*outptr++ = '$';					      \
		*outptr++ = *escseq++;					      \
		*outptr++ = *escseq++;					      \
									      \
		ann = (ann & ~SS3_ann) | (used << 8);			      \
	      }								      \
									      \
	    if (used == CNS11643_2_set)					      \
	      {								      \
		if (outptr + 2 > outend)				      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
		*outptr++ = SS2_0;					      \
		*outptr++ = SS2_1;					      \
	      }								      \
	    else if (used >= CNS11643_3_set && used <= CNS11643_7_set)	      \
	      {								      \
		if (outptr + 2 > outend)				      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
		*outptr++ = SS3_0;					      \
		*outptr++ = SS3_1;					      \
	      }								      \
	    else							      \
	      {								      \
		/* We only have to emit something if currently ASCII is	      \
		   selected.  Otherwise we are switching within the	      \
		   SO charset.  */					      \
		if (set == ASCII_set)					      \
		  {							      \
		    if (outptr + 1 > outend)				      \
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
	    if (outptr + 2 > outend)					      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	  }								      \
	else if (outptr + 2 > outend)					      \
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
DIAG_POP_NEEDS_COMMENT;
#define EXTRA_LOOP_DECLS	, int *setp
#define INIT_PARAMS		int set = (*setp >> 3) & CURRENT_MASK; \
				int ann = (*setp >> 3) & ~CURRENT_MASK
#define REINIT_PARAMS		do					      \
				  {					      \
				    set = (*setp >> 3) & CURRENT_MASK;	      \
				    ann = (*setp >> 3) & ~CURRENT_MASK;	      \
				  }					      \
				while (0)
#define UPDATE_PARAMS		*setp = (set | ann) << 3
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
