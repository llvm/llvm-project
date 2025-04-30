/* Conversion module for ISO-2022-JP-3.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998,
   and Bruno Haible <bruno@clisp.org>, 2002.

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

#include <assert.h>
#include <dlfcn.h>
#include <gconv.h>
#include <stdint.h>
#include <string.h>

#include "jis0201.h"
#include "jis0208.h"
#include "jisx0213.h"

/* This makes obvious what everybody knows: 0x1b is the Esc character.  */
#define ESC 0x1b

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"ISO-2022-JP-3//"
#define FROM_LOOP		from_iso2022jp3_loop
#define TO_LOOP			to_iso2022jp3_loop
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define ONE_DIRECTION		0
#define FROM_LOOP_MIN_NEEDED_FROM	1
#define FROM_LOOP_MAX_NEEDED_FROM	4
#define FROM_LOOP_MIN_NEEDED_TO		4
#define FROM_LOOP_MAX_NEEDED_TO		8
#define TO_LOOP_MIN_NEEDED_FROM		4
#define TO_LOOP_MAX_NEEDED_FROM		4
#define TO_LOOP_MIN_NEEDED_TO		1
#define TO_LOOP_MAX_NEEDED_TO		6
#define PREPARE_LOOP \
  int saved_state;							      \
  int *statep = &data->__statep->__count;
#define EXTRA_LOOP_ARGS		, statep


/* The COUNT element of the state keeps track of the currently selected
   character set.  The possible values are:  */
enum
{
  ASCII_set = 0,		/* Esc ( B */
  JISX0208_1978_set = 1 << 3,	/* Esc $ @ */
  JISX0208_1983_set = 2 << 3,	/* Esc $ B */
  JISX0201_Roman_set = 3 << 3,	/* Esc ( J */
  JISX0201_Kana_set = 4 << 3,	/* Esc ( I */
  JISX0213_1_2000_set = 5 << 3,	/* Esc $ ( O */
  JISX0213_2_set = 6 << 3,	/* Esc $ ( P */
  JISX0213_1_2004_set = 7 << 3,	/* Esc $ ( Q */
  CURRENT_SEL_MASK = 7 << 3
};

/* During UCS-4 to ISO-2022-JP-3 conversion, the COUNT element of the
   state also contains the last two bytes to be output, shifted by 6
   bits, and a one-bit indicator whether they must be preceded by the
   shift sequence, in bit 22.  During ISO-2022-JP-3 to UCS-4
   conversion, COUNT may also contain a non-zero pending wide
   character, shifted by six bits.  This happens for certain inputs in
   JISX0213_1_2004_set and JISX0213_2_set if the second wide character
   in a combining sequence cannot be written because the buffer is
   full.  */

/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  if (data->__statep->__count != ASCII_set)			      \
    {									      \
      if (FROM_DIRECTION)						      \
	{								      \
	  if (__glibc_likely (outbuf + 4 <= outend))			      \
	    {								      \
	      /* Write out the last character.  */			      \
	      *((uint32_t *) outbuf) = data->__statep->__count >> 6;	      \
	      outbuf += sizeof (uint32_t);				      \
	      data->__statep->__count = ASCII_set;			\
	    }								      \
	  else								      \
	    /* We don't have enough room in the output buffer.  */	      \
	    status = __GCONV_FULL_OUTPUT;				      \
	}								      \
      else								      \
	{								      \
	  /* We are not in the initial state.  To switch back we have	      \
	     to write out the buffered character and/or emit the sequence     \
	     `Esc ( B'.  */						      \
	  size_t need =							      \
	    (data->__statep->__count >> 6				      \
	     ? (data->__statep->__count >> 22 ? 3 : 0) + 2		      \
	     : 0)							      \
	    + ((data->__statep->__count & CURRENT_SEL_MASK) != ASCII_set      \
	       ? 3 : 0);						      \
									      \
	  if (__glibc_unlikely (outbuf + need > outend))		      \
	    /* We don't have enough room in the output buffer.  */	      \
	    status = __GCONV_FULL_OUTPUT;				      \
	  else								      \
	    {								      \
	      if (data->__statep->__count >> 6)				      \
		{							      \
		  uint32_t lasttwo = data->__statep->__count >> 6;	      \
									      \
		  if (lasttwo >> 16)					      \
		    {							      \
		      /* Write out the shift sequence before the last	      \
			 character.  */					      \
		      assert ((data->__statep->__count & CURRENT_SEL_MASK)    \
			      == JISX0208_1983_set);			      \
		      *outbuf++ = ESC;					      \
		      *outbuf++ = '$';					      \
		      *outbuf++ = 'B';					      \
		    }							      \
		  /* Write out the last character.  */			      \
		  *outbuf++ = (lasttwo >> 8) & 0xff;			      \
		  *outbuf++ = lasttwo & 0xff;				      \
		}							      \
	      if ((data->__statep->__count & CURRENT_SEL_MASK) != ASCII_set)  \
		{							      \
		  /* Write out the shift sequence.  */			      \
		  *outbuf++ = ESC;					      \
		  *outbuf++ = '(';					      \
		  *outbuf++ = 'B';					      \
		}							      \
	      data->__statep->__count &= 7;				      \
	      data->__statep->__count |= ASCII_set;			      \
	    }								      \
	}								      \
    }


/* Since we might have to reset input pointer we must be able to save
   and retore the state.  */
#define SAVE_RESET_STATE(Save) \
  if (Save)								      \
    saved_state = *statep;						      \
  else									      \
    *statep = saved_state


/* First define the conversion function from ISO-2022-JP-3 to UCS-4.  */
#define MIN_NEEDED_INPUT	FROM_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	FROM_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	FROM_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	FROM_LOOP_MAX_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch;							      \
									      \
    /* Output any pending character.  */				      \
    ch = set >> 6;							      \
    if (__glibc_unlikely (ch != 0))					      \
      {									      \
	put32 (outptr, ch);						      \
	outptr += 4;							      \
	/* Remove the pending character, but preserve state bits.  */	      \
	set &= (1 << 6) - 1;						      \
	continue;							      \
      }									      \
									      \
    /* Otherwise read the next input byte.  */				      \
    ch = *inptr;							      \
									      \
    /* Recognize escape sequences.  */					      \
    if (__glibc_unlikely (ch == ESC))					      \
      {									      \
	/* We now must be prepared to read two to three more bytes.	      \
	   If we have a match in the first byte but then the input buffer     \
	   ends we terminate with an error since we must not risk missing     \
	   an escape sequence just because it is not entirely in the	      \
	   current input buffer.  */					      \
	if (__builtin_expect (inptr + 2 >= inend, 0)			      \
	    || (inptr[1] == '$' && inptr[2] == '('			      \
		&& __builtin_expect (inptr + 3 >= inend, 0)))		      \
	  {								      \
	    /* Not enough input available.  */				      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
									      \
	if (inptr[1] == '(')						      \
	  {								      \
	    if (inptr[2] == 'B')					      \
	      {								      \
		/* ASCII selected.  */					      \
		set = ASCII_set;					      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (inptr[2] == 'J')					      \
	      {								      \
		/* JIS X 0201 selected.  */				      \
		set = JISX0201_Roman_set;				      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (inptr[2] == 'I')					      \
	      {								      \
		/* JIS X 0201 selected.  */				      \
		set = JISX0201_Kana_set;				      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	  }								      \
	else if (inptr[1] == '$')					      \
	  {								      \
	    if (inptr[2] == '@')					      \
	      {								      \
		/* JIS X 0208-1978 selected.  */			      \
		set = JISX0208_1978_set;				      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (inptr[2] == 'B')					      \
	      {								      \
		/* JIS X 0208-1983 selected.  */			      \
		set = JISX0208_1983_set;				      \
		inptr += 3;						      \
		continue;						      \
	      }								      \
	    else if (inptr[2] == '(')					      \
	      {								      \
		if (inptr[3] == 'O' || inptr[3] == 'Q')			      \
		  {							      \
		    /* JIS X 0213 plane 1 selected.  */			      \
		    /* In this direction we don't need to distinguish the     \
		       versions from 2000 and 2004. */			      \
		    set = JISX0213_1_2004_set;				      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
		else if (inptr[3] == 'P')				      \
		  {							      \
		    /* JIS X 0213 plane 2 selected.  */			      \
		    set = JISX0213_2_set;				      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
	      }								      \
	  }								      \
      }									      \
									      \
    if (ch >= 0x80)							      \
      {									      \
	STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
      }									      \
    else if (set == ASCII_set || (ch < 0x21 || ch == 0x7f))		      \
      /* Almost done, just advance the input pointer.  */		      \
      ++inptr;								      \
    else if (set == JISX0201_Roman_set)					      \
      {									      \
	/* Use the JIS X 0201 table.  */				      \
	ch = jisx0201_to_ucs4 (ch);					      \
	if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	  {								      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
	++inptr;							      \
      }									      \
    else if (set == JISX0201_Kana_set)					      \
      {									      \
	/* Use the JIS X 0201 table.  */				      \
	ch = jisx0201_to_ucs4 (ch + 0x80);				      \
	if (__glibc_unlikely (ch == __UNKNOWN_10646_CHAR))		      \
	  {								      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
	++inptr;							      \
      }									      \
    else if (set == JISX0208_1978_set || set == JISX0208_1983_set)	      \
      {									      \
	/* XXX I don't have the tables for these two old variants of	      \
	   JIS X 0208.  Therefore I'm using the tables for JIS X	      \
	   0208-1990.  If somebody has problems with this please	      \
	   provide the appropriate tables.  */				      \
	ch = jisx0208_to_ucs4 (&inptr, inend - inptr, 0);		      \
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
    else /* (set == JISX0213_1_2004_set || set == JISX0213_2_set) */	      \
      {									      \
	if (__glibc_unlikely (inptr + 1 >= inend))			      \
	  {								      \
	    result = __GCONV_INCOMPLETE_INPUT;				      \
	    break;							      \
	  }								      \
									      \
	ch = jisx0213_to_ucs4 (						      \
	       ((JISX0213_1_2004_set - set + (1 << 3)) << 5) + ch,	      \
	       inptr[1]);						      \
	if (ch == 0)							      \
	  STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
									      \
	if (ch < 0x80)							      \
	  {								      \
	    /* It's a combining character.  */				      \
	    uint32_t u1 = __jisx0213_to_ucs_combining[ch - 1][0];	      \
	    uint32_t u2 = __jisx0213_to_ucs_combining[ch - 1][1];	      \
									      \
	    inptr += 2;							      \
									      \
	    put32 (outptr, u1);						      \
	    outptr += 4;						      \
									      \
	    /* See whether we have room for two characters.  */		      \
	    if (outptr + 4 <= outend)					      \
	      {								      \
		put32 (outptr, u2);					      \
		outptr += 4;						      \
		continue;						      \
	      }								      \
									      \
	    /* Otherwise store only the first character now, and	      \
	       put the second one into the queue.  */			      \
	    set |= u2 << 6;						      \
	    /* Tell the caller why we terminate the loop.  */		      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
									      \
	inptr += 2;							      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#define INIT_PARAMS		int set = *statep
#define UPDATE_PARAMS		*statep = set
#include <iconv/loop.c>


/* Next, define the other direction, from UCS-4 to ISO-2022-JP-3.  */

/* Composition tables for each of the relevant combining characters.  */
static const struct
{
  uint16_t base;
  uint16_t composed;
} comp_table_data[] =
{
#define COMP_TABLE_IDX_02E5 0
#define COMP_TABLE_LEN_02E5 1
  { 0x2b64, 0x2b65 }, /* 0x12B65 = 0x12B64 U+02E5 */
#define COMP_TABLE_IDX_02E9 (COMP_TABLE_IDX_02E5 + COMP_TABLE_LEN_02E5)
#define COMP_TABLE_LEN_02E9 1
  { 0x2b60, 0x2b66 }, /* 0x12B66 = 0x12B60 U+02E9 */
#define COMP_TABLE_IDX_0300 (COMP_TABLE_IDX_02E9 + COMP_TABLE_LEN_02E9)
#define COMP_TABLE_LEN_0300 5
  { 0x295c, 0x2b44 }, /* 0x12B44 = 0x1295C U+0300 */
  { 0x2b38, 0x2b48 }, /* 0x12B48 = 0x12B38 U+0300 */
  { 0x2b37, 0x2b4a }, /* 0x12B4A = 0x12B37 U+0300 */
  { 0x2b30, 0x2b4c }, /* 0x12B4C = 0x12B30 U+0300 */
  { 0x2b43, 0x2b4e }, /* 0x12B4E = 0x12B43 U+0300 */
#define COMP_TABLE_IDX_0301 (COMP_TABLE_IDX_0300 + COMP_TABLE_LEN_0300)
#define COMP_TABLE_LEN_0301 4
  { 0x2b38, 0x2b49 }, /* 0x12B49 = 0x12B38 U+0301 */
  { 0x2b37, 0x2b4b }, /* 0x12B4B = 0x12B37 U+0301 */
  { 0x2b30, 0x2b4d }, /* 0x12B4D = 0x12B30 U+0301 */
  { 0x2b43, 0x2b4f }, /* 0x12B4F = 0x12B43 U+0301 */
#define COMP_TABLE_IDX_309A (COMP_TABLE_IDX_0301 + COMP_TABLE_LEN_0301)
#define COMP_TABLE_LEN_309A 14
  { 0x242b, 0x2477 }, /* 0x12477 = 0x1242B U+309A */
  { 0x242d, 0x2478 }, /* 0x12478 = 0x1242D U+309A */
  { 0x242f, 0x2479 }, /* 0x12479 = 0x1242F U+309A */
  { 0x2431, 0x247a }, /* 0x1247A = 0x12431 U+309A */
  { 0x2433, 0x247b }, /* 0x1247B = 0x12433 U+309A */
  { 0x252b, 0x2577 }, /* 0x12577 = 0x1252B U+309A */
  { 0x252d, 0x2578 }, /* 0x12578 = 0x1252D U+309A */
  { 0x252f, 0x2579 }, /* 0x12579 = 0x1252F U+309A */
  { 0x2531, 0x257a }, /* 0x1257A = 0x12531 U+309A */
  { 0x2533, 0x257b }, /* 0x1257B = 0x12533 U+309A */
  { 0x253b, 0x257c }, /* 0x1257C = 0x1253B U+309A */
  { 0x2544, 0x257d }, /* 0x1257D = 0x12544 U+309A */
  { 0x2548, 0x257e }, /* 0x1257E = 0x12548 U+309A */
  { 0x2675, 0x2678 }, /* 0x12678 = 0x12675 U+309A */
};

#define MIN_NEEDED_INPUT	TO_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	TO_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	TO_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	TO_LOOP_MAX_NEEDED_TO
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch = get32 (inptr);					      \
									      \
    if (lasttwo != 0)							      \
      {									      \
	/* Attempt to combine the last character with this one.  */	      \
	unsigned int idx;						      \
	unsigned int len;						      \
									      \
	if (ch == 0x02e5)						      \
	  idx = COMP_TABLE_IDX_02E5, len = COMP_TABLE_LEN_02E5;		      \
	else if (ch == 0x02e9)						      \
	  idx = COMP_TABLE_IDX_02E9, len = COMP_TABLE_LEN_02E9;		      \
	else if (ch == 0x0300)						      \
	  idx = COMP_TABLE_IDX_0300, len = COMP_TABLE_LEN_0300;		      \
	else if (ch == 0x0301)						      \
	  idx = COMP_TABLE_IDX_0301, len = COMP_TABLE_LEN_0301;		      \
	else if (ch == 0x309a)						      \
	  idx = COMP_TABLE_IDX_309A, len = COMP_TABLE_LEN_309A;		      \
	else								      \
	  goto not_combining;						      \
									      \
	do								      \
	  if (comp_table_data[idx].base == (uint16_t) lasttwo)		      \
	    break;							      \
	while (++idx, --len > 0);					      \
									      \
	if (len > 0)							      \
	  {								      \
	    /* Output the combined character.  */			      \
	    /* We know the combined character is in JISX0213 plane 1,	      \
	       but the buffered character may have been in JISX0208 or in     \
	       JISX0213 plane 1.  */					      \
	    size_t need =						      \
	      (lasttwo >> 16						      \
	       || (set != JISX0213_1_2000_set && set != JISX0213_1_2004_set)  \
	       ? 4 : 0);						      \
									      \
	    if (__glibc_unlikely (outptr + need + 2 > outend))		      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    if (need)							      \
	      {								      \
		/* But first, output the escape sequence.  */		      \
		*outptr++ = ESC;					      \
		*outptr++ = '$';					      \
		*outptr++ = '(';					      \
		*outptr++ = 'O';					      \
		set = JISX0213_1_2000_set;				      \
	      }								      \
	    lasttwo = comp_table_data[idx].composed;			      \
	    *outptr++ = (lasttwo >> 8) & 0xff;				      \
	    *outptr++ = lasttwo & 0xff;					      \
	    lasttwo = 0;						      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
									      \
      not_combining:							      \
	/* Output the buffered character.  */				      \
	/* We know it is in JISX0208 or in JISX0213 plane 1.  */	      \
	{								      \
	  size_t need = (lasttwo >> 16 ? 3 : 0);			      \
									      \
	  if (__glibc_unlikely (outptr + need + 2 > outend))		      \
	    {								      \
	      result = __GCONV_FULL_OUTPUT;				      \
	      break;							      \
	    }								      \
	  if (need)							      \
	    {								      \
	      /* But first, output the escape sequence.  */		      \
	      assert (set == JISX0208_1983_set);			      \
	      *outptr++ = ESC;						      \
	      *outptr++ = '$';						      \
	      *outptr++ = 'B';						      \
	    }								      \
	  *outptr++ = (lasttwo >> 8) & 0xff;				      \
	  *outptr++ = lasttwo & 0xff;					      \
	  lasttwo = 0;							      \
	  continue;							      \
	}								      \
      }									      \
									      \
    /* First see whether we can write the character using the currently	      \
       selected character set.  */					      \
    if (set == ASCII_set)						      \
      {									      \
	/* Please note that the NUL byte is *not* matched if we are not	      \
	   currently using the ASCII charset.  This is because we must	      \
	   switch to the initial state whenever a NUL byte is written.  */    \
	if (ch <= 0x7f)							      \
	  {								      \
	    *outptr++ = ch;						      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
      }									      \
    /* ISO-2022-JP recommends to encode the newline character always in	      \
       ASCII since this allows a context-free interpretation of the	      \
       characters at the beginning of the next line.  Otherwise it would      \
       have to be known whether the last line ended using ASCII or	      \
       JIS X 0201.  */							      \
    else if (set == JISX0201_Roman_set)					      \
      {									      \
	unsigned char buf[1];						      \
	if (ucs4_to_jisx0201 (ch, buf) != __UNKNOWN_10646_CHAR		      \
	    && buf[0] > 0x20 && buf[0] < 0x80)				      \
	  {								      \
	    *outptr++ = buf[0];						      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
      }									      \
    else if (set == JISX0201_Kana_set)					      \
      {									      \
	unsigned char buf[1];						      \
	if (ucs4_to_jisx0201 (ch, buf) != __UNKNOWN_10646_CHAR		      \
	    && buf[0] >= 0x80)						      \
	  {								      \
	    *outptr++ = buf[0] - 0x80;					      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
      }									      \
    else if (/*set == JISX0208_1978_set || */ set == JISX0208_1983_set)	      \
      {									      \
	size_t written = ucs4_to_jisx0208 (ch, outptr, outend - outptr);      \
									      \
	if (written != __UNKNOWN_10646_CHAR)				      \
	  {								      \
	    uint32_t jch = ucs4_to_jisx0213 (ch);			      \
									      \
	    if (jch & 0x0080)						      \
	      {								      \
		/* A possible match in comp_table_data.  Buffer it.  */	      \
		lasttwo = jch & 0x7f7f;					      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	    if (__glibc_unlikely (written == 0))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    else							      \
	      {								      \
	 	outptr += written;					      \
		inptr += 4;						      \
		continue;						      \
	     }								      \
	  }								      \
      }									      \
    else								      \
      {									      \
	/* (set == JISX0213_1_2000_set || set == JISX0213_1_2004_set	      \
	    || set == JISX0213_2_set) */				      \
	uint32_t jch = ucs4_to_jisx0213 (ch);				      \
									      \
	if (jch != 0							      \
	    && (jch & 0x8000						      \
		? set == JISX0213_2_set					      \
		: (set == JISX0213_1_2004_set				      \
		   || (set == JISX0213_1_2000_set			      \
		       && !jisx0213_added_in_2004_p (jch)))))		      \
	  {								      \
	    if (jch & 0x0080)						      \
	      {								      \
		/* A possible match in comp_table_data.  Buffer it.  */	      \
									      \
		/* We know it's a JISX 0213 plane 1 character.  */	      \
		assert ((jch & 0x8000) == 0);				      \
									      \
		lasttwo = jch & 0x7f7f;					      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
									      \
	    if (__glibc_unlikely (outptr + 1 >= outend))		      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    *outptr++ = (jch >> 8) & 0x7f;				      \
	    *outptr++ = jch & 0x7f;					      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
      }									      \
									      \
    /* The attempts to use the currently selected character set failed,	      \
       either because the character requires a different character set,	      \
       or because the character is unknown.  */				      \
									      \
    if (ch <= 0x7f)							      \
      {									      \
	/* We must encode using ASCII.  First write out the escape	      \
	   sequence.  */						      \
	if (__glibc_unlikely (outptr + 3 > outend))			      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
									      \
	*outptr++ = ESC;						      \
	*outptr++ = '(';						      \
	*outptr++ = 'B';						      \
	set = ASCII_set;						      \
									      \
	if (__glibc_unlikely (outptr >= outend))			      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	*outptr++ = ch;							      \
      }									      \
    else								      \
      {									      \
	unsigned char buf[2];						      \
									      \
	/* Try JIS X 0201 Roman.  */					      \
	if (ucs4_to_jisx0201 (ch, buf) != __UNKNOWN_10646_CHAR		      \
	    && buf[0] > 0x20 && buf[0] < 0x80)				      \
	  {								      \
	    if (set != JISX0201_Roman_set)				      \
	      {								      \
		if (__glibc_unlikely (outptr + 3 > outend))		      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
		*outptr++ = ESC;					      \
		*outptr++ = '(';					      \
		*outptr++ = 'J';					      \
		set = JISX0201_Roman_set;				      \
	      }								      \
									      \
	    if (__glibc_unlikely (outptr >= outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    *outptr++ = buf[0];						      \
	  }								      \
	else								      \
	  {								      \
	    uint32_t jch = ucs4_to_jisx0213 (ch);			      \
									      \
	    /* Try JIS X 0208.  */					      \
	    size_t written = ucs4_to_jisx0208 (ch, buf, 2);		      \
	    if (written != __UNKNOWN_10646_CHAR)			      \
	      {								      \
		if (jch & 0x0080)					      \
		  {							      \
		    /* A possible match in comp_table_data.  Buffer it.  */   \
		    lasttwo = ((set != JISX0208_1983_set ? 1 : 0) << 16)      \
			      | (jch & 0x7f7f);				      \
		    set = JISX0208_1983_set;				      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
									      \
		if (set != JISX0208_1983_set)				      \
		  {							      \
		    if (__glibc_unlikely (outptr + 3 > outend))		      \
		      {							      \
			result = __GCONV_FULL_OUTPUT;			      \
			break;						      \
		      }							      \
		    *outptr++ = ESC;					      \
		    *outptr++ = '$';					      \
		    *outptr++ = 'B';					      \
		    set = JISX0208_1983_set;				      \
		  }							      \
									      \
		if (__glibc_unlikely (outptr + 2 > outend))		      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
		*outptr++ = buf[0];					      \
		*outptr++ = buf[1];					      \
	      }								      \
	    else							      \
	      {								      \
		/* Try JIS X 0213.  */					      \
		if (jch != 0)						      \
		  {							      \
		    int new_set =					      \
		      (jch & 0x8000					      \
		       ? JISX0213_2_set					      \
		       : jisx0213_added_in_2004_p (jch)			      \
			 ? JISX0213_1_2004_set				      \
			 : JISX0213_1_2000_set);			      \
									      \
		    if (set != new_set)					      \
		      {							      \
			if (__glibc_unlikely (outptr + 4 > outend))	      \
			  {						      \
			    result = __GCONV_FULL_OUTPUT;		      \
			    break;					      \
			  }						      \
			*outptr++ = ESC;				      \
			*outptr++ = '$';				      \
			*outptr++ = '(';				      \
			*outptr++ =					      \
			  ((new_set - JISX0213_1_2000_set) >> 3) + 'O';	      \
			set = new_set;					      \
		      }							      \
									      \
		    if (jch & 0x0080)					      \
		      {							      \
			/* A possible match in comp_table_data.		      \
			   Buffer it.  */				      \
									      \
			/* We know it's a JIS X 0213 plane 1 character.  */   \
			assert ((jch & 0x8000) == 0);			      \
									      \
			lasttwo = jch & 0x7f7f;				      \
			inptr += 4;					      \
			continue;					      \
		      }							      \
									      \
		    if (__glibc_unlikely (outptr + 1 >= outend))	      \
		      {							      \
			result = __GCONV_FULL_OUTPUT;			      \
			break;						      \
		      }							      \
		    *outptr++ = (jch >> 8) & 0x7f;			      \
		    *outptr++ = jch & 0x7f;				      \
		  }							      \
		else							      \
		  {							      \
		    /* Try JIS X 0201 Katakana.  This is officially not part  \
		       of ISO-2022-JP-3.  Therefore we try it after all other \
		       attempts.  */					      \
		    if (ucs4_to_jisx0201 (ch, buf) != __UNKNOWN_10646_CHAR    \
			&& buf[0] >= 0x80)				      \
		      {							      \
			if (set != JISX0201_Kana_set)			      \
			  {						      \
			    if (__builtin_expect (outptr + 3 > outend, 0))    \
			      {						      \
				result = __GCONV_FULL_OUTPUT;		      \
				break;					      \
			      }						      \
			    *outptr++ = ESC;				      \
			    *outptr++ = '(';				      \
			    *outptr++ = 'I';				      \
			    set = JISX0201_Kana_set;			      \
			  }						      \
									      \
			if (__glibc_unlikely (outptr >= outend))	      \
			  {						      \
			    result = __GCONV_FULL_OUTPUT;		      \
			    break;					      \
			  }						      \
			*outptr++ = buf[0] - 0x80;			      \
		      }							      \
		    else						      \
		      {							      \
			UNICODE_TAG_HANDLER (ch, 4);			      \
									      \
			/* Illegal character.  */			      \
			STANDARD_TO_LOOP_ERR_HANDLER (4);		      \
		      }							      \
		  }							      \
	      }								      \
	  }								      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#define INIT_PARAMS		int set = *statep & CURRENT_SEL_MASK;	      \
				uint32_t lasttwo = *statep >> 6
#define REINIT_PARAMS		do					      \
				  {					      \
				    set = *statep & CURRENT_SEL_MASK;	      \
				    lasttwo = *statep >> 6;		      \
				  }					      \
				while (0)
#define UPDATE_PARAMS		*statep = set | (lasttwo << 6)
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
