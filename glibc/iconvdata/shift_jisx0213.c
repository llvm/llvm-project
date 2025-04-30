/* Conversion from and to Shift_JISX0213.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <bruno@clisp.org>, 2002.

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

/* The structure of Shift_JISX0213 is as follows:

   0x00..0x7F: ISO646-JP, an ASCII variant

   0x{A1..DF}: JISX0201 Katakana.

   0x{81..9F,E0..EF}{40..7E,80..FC}: JISX0213 plane 1.

   0x{F0..FC}{40..7E,80..FC}: JISX0213 plane 2, with irregular row mapping.

   Note that some JISX0213 characters are not contained in Unicode 3.2
   and are therefore best represented as sequences of Unicode characters.
*/

#include "jisx0213.h"

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"SHIFT_JISX0213//"
#define FROM_LOOP		from_shift_jisx0213
#define TO_LOOP			to_shift_jisx0213
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define ONE_DIRECTION		0
#define FROM_LOOP_MIN_NEEDED_FROM	1
#define FROM_LOOP_MAX_NEEDED_FROM	2
#define FROM_LOOP_MIN_NEEDED_TO		4
#define FROM_LOOP_MAX_NEEDED_TO		8
#define TO_LOOP_MIN_NEEDED_FROM		4
#define TO_LOOP_MAX_NEEDED_FROM		4
#define TO_LOOP_MIN_NEEDED_TO		1
#define TO_LOOP_MAX_NEEDED_TO		2
#define PREPARE_LOOP \
  int saved_state;							      \
  int *statep = &data->__statep->__count;
#define EXTRA_LOOP_ARGS		, statep


/* Since we might have to reset input pointer we must be able to save
   and restore the state.  */
#define SAVE_RESET_STATE(Save) \
  if (Save)								      \
    saved_state = *statep;						      \
  else									      \
    *statep = saved_state


/* During Shift_JISX0213 to UCS-4 conversion, the COUNT element of the state
   contains the last UCS-4 character, shifted by 3 bits.
   During UCS-4 to Shift_JISX0213 conversion, the COUNT element of the state
   contains the last two bytes to be output, shifted by 3 bits.  */

/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  if (data->__statep->__count != 0)					      \
    {									      \
      if (FROM_DIRECTION)						      \
	{								      \
	  if (__glibc_likely (outbuf + 4 <= outend))			      \
	    {								      \
	      /* Write out the last character.  */			      \
	      *((uint32_t *) outbuf) = data->__statep->__count >> 3;	      \
	      outbuf += sizeof (uint32_t);				      \
	      data->__statep->__count = 0;				      \
	    }								      \
	  else								      \
	    /* We don't have enough room in the output buffer.  */	      \
	    status = __GCONV_FULL_OUTPUT;				      \
	}								      \
      else								      \
	{								      \
	  if (__glibc_likely (outbuf + 2 <= outend))			      \
	    {								      \
	      /* Write out the last character.  */			      \
	      uint32_t lasttwo = data->__statep->__count >> 3;		      \
	      *outbuf++ = (lasttwo >> 8) & 0xff;			      \
	      *outbuf++ = lasttwo & 0xff;				      \
	      data->__statep->__count = 0;				      \
	    }								      \
	  else								      \
	    /* We don't have enough room in the output buffer.  */	      \
	    status = __GCONV_FULL_OUTPUT;				      \
	}								      \
    }


/* First define the conversion function from Shift_JISX0213 to UCS-4.  */
#define MIN_NEEDED_INPUT	FROM_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	FROM_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	FROM_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	FROM_LOOP_MAX_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch;							      \
									      \
    /* Determine whether there is a buffered character pending.  */	      \
    ch = *statep >> 3;							      \
    if (__glibc_likely (ch == 0))					      \
      {									      \
	/* No - so look at the next input byte.  */			      \
	ch = *inptr;							      \
									      \
	if (ch < 0x80)							      \
	  {								      \
	    /* Plain ISO646-JP character.  */				      \
	    if (__glibc_unlikely (ch == 0x5c))				      \
	      ch = 0xa5;						      \
	    else if (__glibc_unlikely (ch == 0x7e))			      \
	      ch = 0x203e;						      \
	    ++inptr;							      \
	  }								      \
	else if (ch >= 0xa1 && ch <= 0xdf)				      \
	  {								      \
	    /* Half-width katakana.  */					      \
	    ch += 0xfec0;						      \
	    ++inptr;							      \
	  }								      \
	else if ((ch >= 0x81 && ch <= 0x9f) || (ch >= 0xe0 && ch <= 0xfc))    \
	  {								      \
	    /* Two byte character.  */					      \
	    uint32_t ch2;						      \
									      \
	    if (__glibc_unlikely (inptr + 1 >= inend))			      \
	      {								      \
		/* The second byte is not available.  */		      \
		result = __GCONV_INCOMPLETE_INPUT;			      \
		break;							      \
	      }								      \
									      \
	    ch2 = inptr[1];						      \
									      \
	    /* The second byte must be in the range 0x{40..7E,80..FC}.  */    \
	    if (__glibc_unlikely (ch2 < 0x40 || ch2 == 0x7f || ch2 > 0xfc))   \
	      {								      \
		/* This is an illegal character.  */			      \
		STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
	      }								      \
									      \
	    /* Convert to row and column.  */				      \
	    if (ch < 0xe0)						      \
	      ch -= 0x81;						      \
	    else							      \
	      ch -= 0xc1;						      \
	    if (ch2 < 0x80)						      \
	      ch2 -= 0x40;						      \
	    else							      \
	      ch2 -= 0x41;						      \
	    /* Now 0 <= ch <= 0x3b, 0 <= ch2 <= 0xbb.  */		      \
	    ch = 2 * ch;						      \
	    if (ch2 >= 0x5e)						      \
	      ch2 -= 0x5e, ch++;					      \
	    ch2 += 0x21;						      \
	    if (ch >= 0x5e)						      \
	      {								      \
		/* Handling of JISX 0213 plane 2 rows.  */		      \
		if (ch >= 0x67)						      \
		  ch += 230;						      \
		else if (ch >= 0x63 || ch == 0x5f)			      \
		  ch += 168;						      \
		else 							      \
		  ch += 162;						      \
	      }								      \
									      \
	    ch = jisx0213_to_ucs4 (0x121 + ch, ch2);			      \
									      \
	    if (ch == 0)						      \
	      {								      \
		/* This is an illegal character.  */			      \
		STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
	      }								      \
									      \
	    inptr += 2;							      \
									      \
	    if (ch < 0x80)						      \
	      {								      \
		/* It's a combining character.  */			      \
		uint32_t u1 = __jisx0213_to_ucs_combining[ch - 1][0];	      \
		uint32_t u2 = __jisx0213_to_ucs_combining[ch - 1][1];	      \
									      \
		put32 (outptr, u1);					      \
		outptr += 4;						      \
									      \
		/* See whether we have room for two characters.  */	      \
		if (outptr + 4 <= outend)				      \
		  {							      \
		    put32 (outptr, u2);					      \
		    outptr += 4;					      \
		    continue;						      \
		  }							      \
									      \
		/* Otherwise store only the first character now, and	      \
		   put the second one into the queue.  */		      \
		*statep = u2 << 3;					      \
		/* Tell the caller why we terminate the loop.  */	      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	  }								      \
	else								      \
	  {								      \
	    /* This is illegal.  */					      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#define ONEBYTE_BODY \
  {									      \
    if (c < 0x80)							      \
      {									      \
	if (c == 0x5c)							      \
	  return 0xa5;							      \
	if (c == 0x7e)							      \
	  return 0x203e;						      \
	return c;							      \
      }									      \
    if (c >= 0xa1 && c <= 0xdf)						      \
      return 0xfec0 + c;						      \
    return WEOF;							      \
  }
#include <iconv/loop.c>


/* Next, define the other direction, from UCS-4 to Shift_JISX0213.  */

/* Composition tables for each of the relevant combining characters.  */
static const struct
{
  uint16_t base;
  uint16_t composed;
} comp_table_data[] =
{
#define COMP_TABLE_IDX_02E5 0
#define COMP_TABLE_LEN_02E5 1
  { 0x8684, 0x8685 }, /* 0x12B65 = 0x12B64 U+02E5 */
#define COMP_TABLE_IDX_02E9 (COMP_TABLE_IDX_02E5 + COMP_TABLE_LEN_02E5)
#define COMP_TABLE_LEN_02E9 1
  { 0x8680, 0x8686 }, /* 0x12B66 = 0x12B60 U+02E9 */
#define COMP_TABLE_IDX_0300 (COMP_TABLE_IDX_02E9 + COMP_TABLE_LEN_02E9)
#define COMP_TABLE_LEN_0300 5
  { 0x857b, 0x8663 }, /* 0x12B44 = 0x1295C U+0300 */
  { 0x8657, 0x8667 }, /* 0x12B48 = 0x12B38 U+0300 */
  { 0x8656, 0x8669 }, /* 0x12B4A = 0x12B37 U+0300 */
  { 0x864f, 0x866b }, /* 0x12B4C = 0x12B30 U+0300 */
  { 0x8662, 0x866d }, /* 0x12B4E = 0x12B43 U+0300 */
#define COMP_TABLE_IDX_0301 (COMP_TABLE_IDX_0300 + COMP_TABLE_LEN_0300)
#define COMP_TABLE_LEN_0301 4
  { 0x8657, 0x8668 }, /* 0x12B49 = 0x12B38 U+0301 */
  { 0x8656, 0x866a }, /* 0x12B4B = 0x12B37 U+0301 */
  { 0x864f, 0x866c }, /* 0x12B4D = 0x12B30 U+0301 */
  { 0x8662, 0x866e }, /* 0x12B4F = 0x12B43 U+0301 */
#define COMP_TABLE_IDX_309A (COMP_TABLE_IDX_0301 + COMP_TABLE_LEN_0301)
#define COMP_TABLE_LEN_309A 14
  { 0x82a9, 0x82f5 }, /* 0x12477 = 0x1242B U+309A */
  { 0x82ab, 0x82f6 }, /* 0x12478 = 0x1242D U+309A */
  { 0x82ad, 0x82f7 }, /* 0x12479 = 0x1242F U+309A */
  { 0x82af, 0x82f8 }, /* 0x1247A = 0x12431 U+309A */
  { 0x82b1, 0x82f9 }, /* 0x1247B = 0x12433 U+309A */
  { 0x834a, 0x8397 }, /* 0x12577 = 0x1252B U+309A */
  { 0x834c, 0x8398 }, /* 0x12578 = 0x1252D U+309A */
  { 0x834e, 0x8399 }, /* 0x12579 = 0x1252F U+309A */
  { 0x8350, 0x839a }, /* 0x1257A = 0x12531 U+309A */
  { 0x8352, 0x839b }, /* 0x1257B = 0x12533 U+309A */
  { 0x835a, 0x839c }, /* 0x1257C = 0x1253B U+309A */
  { 0x8363, 0x839d }, /* 0x1257D = 0x12544 U+309A */
  { 0x8367, 0x839e }, /* 0x1257E = 0x12548 U+309A */
  { 0x83f3, 0x83f6 }, /* 0x12678 = 0x12675 U+309A */
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
    if ((*statep >> 3) != 0)						      \
      {									      \
	/* Attempt to combine the last character with this one.  */	      \
	uint16_t lasttwo = *statep >> 3;				      \
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
	  if (comp_table_data[idx].base == lasttwo)			      \
	    break;							      \
	while (++idx, --len > 0);					      \
									      \
	if (len > 0)							      \
	  {								      \
	    /* Output the combined character.  */			      \
	    if (__glibc_unlikely (outptr + 1 >= outend))		      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    lasttwo = comp_table_data[idx].composed;			      \
	    *outptr++ = (lasttwo >> 8) & 0xff;				      \
	    *outptr++ = lasttwo & 0xff;					      \
	    *statep = 0;						      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
									      \
      not_combining:							      \
	/* Output the buffered character.  */				      \
	if (__glibc_unlikely (outptr + 1 >= outend))			      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	*outptr++ = (lasttwo >> 8) & 0xff;				      \
	*outptr++ = lasttwo & 0xff;					      \
	*statep = 0;							      \
	continue;							      \
      }									      \
									      \
    if (ch < 0x80)							      \
      /* Plain ISO646-JP character.  */					      \
      *outptr++ = ch;							      \
    else if (ch == 0xa5)						      \
      *outptr++ = 0x5c;							      \
    else if (ch == 0x203e)						      \
      *outptr++ = 0x7e;							      \
    else if (ch >= 0xff61 && ch <= 0xff9f)				      \
      /* Half-width katakana.  */					      \
      *outptr++ = ch - 0xfec0;						      \
    else								      \
      {									      \
	unsigned int s1, s2;						      \
	uint32_t jch = ucs4_to_jisx0213 (ch);				      \
	if (jch == 0)							      \
	  {								      \
	    UNICODE_TAG_HANDLER (ch, 4);				      \
									      \
	    /* Illegal character.  */					      \
	    STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
	  }								      \
									      \
	/* Convert it to shifted representation.  */			      \
	s1 = jch >> 8;							      \
	s2 = jch & 0x7f;							      \
	s1 -= 0x21;							      \
	s2 -= 0x21;							      \
	if (s1 >= 0x5e)							      \
	  {								      \
	    /* Handling of JISX 0213 plane 2 rows.  */			      \
	    if (s1 >= 0xcd) /* rows 0x26E..0x27E */			      \
	      s1 -= 102;						      \
	    else if (s1 >= 0x8b || s1 == 0x87) /* rows 0x228, 0x22C..0x22F */ \
	      s1 -= 40;							      \
	    else /* rows 0x221, 0x223..0x225 */				      \
	      s1 -= 34;							      \
	    /* Now 0x5e <= s1 <= 0x77.  */				      \
	  }								      \
	if (s1 & 1)							      \
	  s2 += 0x5e;							      \
	s1 = s1 >> 1;							      \
	if (s1 < 0x1f)							      \
	  s1 += 0x81;							      \
	else								      \
	  s1 += 0xc1;							      \
	if (s2 < 0x3f)							      \
	  s2 += 0x40;							      \
	else								      \
	  s2 += 0x41;							      \
									      \
	if (jch & 0x0080)						      \
	  {								      \
	    /* A possible match in comp_table_data.  We have to buffer it.  */\
									      \
	    /* We know it's a JISX 0213 plane 1 character.  */		      \
	    assert ((jch & 0x8000) == 0);				      \
									      \
	    *statep = ((s1 << 8) | s2) << 3;				      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
									      \
	/* Output the shifted representation.  */			      \
	if (__glibc_unlikely (outptr + 1 >= outend))			      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	*outptr++ = s1;							      \
	*outptr++ = s2;							      \
      }									      \
									      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
