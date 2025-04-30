/* Conversion from and to EUC-JISX0213.
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

/* The structure of EUC-JISX0213 is as follows:

   0x00..0x7F: ASCII

   0x8E{A1..FE}: JISX0201 Katakana, with prefix 0x8E, offset by +0x80.

   0x8F{A1..FE}{A1..FE}: JISX0213 plane 2, with prefix 0x8F, offset by +0x8080.

   0x{A1..FE}{A1..FE}: JISX0213 plane 1, offset by +0x8080.

   Note that some JISX0213 characters are not contained in Unicode 3.2
   and are therefore best represented as sequences of Unicode characters.
*/

#include "jisx0213.h"

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"EUC-JISX0213//"
#define FROM_LOOP		from_euc_jisx0213
#define TO_LOOP			to_euc_jisx0213
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define ONE_DIRECTION		0
#define FROM_LOOP_MIN_NEEDED_FROM	1
#define FROM_LOOP_MAX_NEEDED_FROM	3
#define FROM_LOOP_MIN_NEEDED_TO		4
#define FROM_LOOP_MAX_NEEDED_TO		8
#define TO_LOOP_MIN_NEEDED_FROM		4
#define TO_LOOP_MAX_NEEDED_FROM		4
#define TO_LOOP_MIN_NEEDED_TO		1
#define TO_LOOP_MAX_NEEDED_TO		3
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


/* During EUC-JISX0213 to UCS-4 conversion, the COUNT element of the state
   contains the last UCS-4 character, shifted by 3 bits.
   During UCS-4 to EUC-JISX0213 conversion, the COUNT element of the state
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


/* First define the conversion function from EUC-JISX0213 to UCS-4.  */
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
	  /* Plain ASCII character.  */					      \
	  ++inptr;							      \
	else if ((ch >= 0xa1 && ch <= 0xfe) || ch == 0x8e || ch == 0x8f)      \
	  {								      \
	    /* Two or three byte character.  */				      \
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
	    /* The second byte must be >= 0xa1 and <= 0xfe.  */		      \
	    if (__glibc_unlikely (ch2 < 0xa1 || ch2 > 0xfe))		      \
	      {								      \
		/* This is an illegal character.  */			      \
		STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
	      }								      \
									      \
	    if (ch == 0x8e)						      \
	      {								      \
		/* Half-width katakana.  */				      \
		if (__glibc_unlikely (ch2 > 0xdf))			      \
		  STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
									      \
		ch = ch2 + 0xfec0;					      \
		inptr += 2;						      \
	      }								      \
	    else							      \
	      {								      \
		const unsigned char *endp;				      \
									      \
		if (ch == 0x8f)						      \
		  {							      \
		    /* JISX 0213 plane 2.  */				      \
		    uint32_t ch3;					      \
									      \
		    if (__glibc_unlikely (inptr + 2 >= inend))		      \
		      {							      \
			/* The third byte is not available.  */		      \
			result = __GCONV_INCOMPLETE_INPUT;		      \
			break;						      \
		      }							      \
									      \
		    ch3 = inptr[2];					      \
		    endp = inptr + 3;					      \
									      \
		    ch = jisx0213_to_ucs4 (0x200 - 0x80 + ch2, ch3 ^ 0x80);   \
		  }							      \
		else							      \
		  {							      \
		    /* JISX 0213 plane 1.  */				      \
		    endp = inptr + 2;					      \
									      \
		    ch = jisx0213_to_ucs4 (0x100 - 0x80 + ch, ch2 ^ 0x80);    \
		  }							      \
									      \
		if (ch == 0)						      \
		  /* This is an illegal character.  */			      \
		  STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
									      \
		inptr = endp;						      \
									      \
		if (ch < 0x80)						      \
		  {							      \
		    /* It's a combining character.  */			      \
		    uint32_t u1 = __jisx0213_to_ucs_combining[ch - 1][0];     \
		    uint32_t u2 = __jisx0213_to_ucs_combining[ch - 1][1];     \
									      \
		    put32 (outptr, u1);					      \
		    outptr += 4;					      \
									      \
		    /* See whether we have room for two characters.  */	      \
		    if (outptr + 4 <= outend)				      \
		      {							      \
			put32 (outptr, u2);				      \
			outptr += 4;					      \
			continue;					      \
		      }							      \
									      \
		    /* Otherwise store only the first character now, and      \
		       put the second one into the queue.  */		      \
		    *statep = u2 << 3;					      \
		    /* Tell the caller why we terminate the loop.  */	      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
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
      return c;								      \
    else								      \
      return WEOF;							      \
  }
#include <iconv/loop.c>


/* Next, define the other direction, from UCS-4 to EUC-JISX0213.  */

/* Composition tables for each of the relevant combining characters.  */
static const struct
{
  uint16_t base;
  uint16_t composed;
} comp_table_data[] =
{
#define COMP_TABLE_IDX_02E5 0
#define COMP_TABLE_LEN_02E5 1
  { 0xabe4, 0xabe5 }, /* 0x12B65 = 0x12B64 U+02E5 */
#define COMP_TABLE_IDX_02E9 (COMP_TABLE_IDX_02E5 + COMP_TABLE_LEN_02E5)
#define COMP_TABLE_LEN_02E9 1
  { 0xabe0, 0xabe6 }, /* 0x12B66 = 0x12B60 U+02E9 */
#define COMP_TABLE_IDX_0300 (COMP_TABLE_IDX_02E9 + COMP_TABLE_LEN_02E9)
#define COMP_TABLE_LEN_0300 5
  { 0xa9dc, 0xabc4 }, /* 0x12B44 = 0x1295C U+0300 */
  { 0xabb8, 0xabc8 }, /* 0x12B48 = 0x12B38 U+0300 */
  { 0xabb7, 0xabca }, /* 0x12B4A = 0x12B37 U+0300 */
  { 0xabb0, 0xabcc }, /* 0x12B4C = 0x12B30 U+0300 */
  { 0xabc3, 0xabce }, /* 0x12B4E = 0x12B43 U+0300 */
#define COMP_TABLE_IDX_0301 (COMP_TABLE_IDX_0300 + COMP_TABLE_LEN_0300)
#define COMP_TABLE_LEN_0301 4
  { 0xabb8, 0xabc9 }, /* 0x12B49 = 0x12B38 U+0301 */
  { 0xabb7, 0xabcb }, /* 0x12B4B = 0x12B37 U+0301 */
  { 0xabb0, 0xabcd }, /* 0x12B4D = 0x12B30 U+0301 */
  { 0xabc3, 0xabcf }, /* 0x12B4F = 0x12B43 U+0301 */
#define COMP_TABLE_IDX_309A (COMP_TABLE_IDX_0301 + COMP_TABLE_LEN_0301)
#define COMP_TABLE_LEN_309A 14
  { 0xa4ab, 0xa4f7 }, /* 0x12477 = 0x1242B U+309A */
  { 0xa4ad, 0xa4f8 }, /* 0x12478 = 0x1242D U+309A */
  { 0xa4af, 0xa4f9 }, /* 0x12479 = 0x1242F U+309A */
  { 0xa4b1, 0xa4fa }, /* 0x1247A = 0x12431 U+309A */
  { 0xa4b3, 0xa4fb }, /* 0x1247B = 0x12433 U+309A */
  { 0xa5ab, 0xa5f7 }, /* 0x12577 = 0x1252B U+309A */
  { 0xa5ad, 0xa5f8 }, /* 0x12578 = 0x1252D U+309A */
  { 0xa5af, 0xa5f9 }, /* 0x12579 = 0x1252F U+309A */
  { 0xa5b1, 0xa5fa }, /* 0x1257A = 0x12531 U+309A */
  { 0xa5b3, 0xa5fb }, /* 0x1257B = 0x12533 U+309A */
  { 0xa5bb, 0xa5fc }, /* 0x1257C = 0x1253B U+309A */
  { 0xa5c4, 0xa5fd }, /* 0x1257D = 0x12544 U+309A */
  { 0xa5c8, 0xa5fe }, /* 0x1257E = 0x12548 U+309A */
  { 0xa6f5, 0xa6f8 }, /* 0x12678 = 0x12675 U+309A */
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
      /* Plain ASCII character.  */					      \
      *outptr++ = ch;							      \
    else if (ch >= 0xff61 && ch <= 0xff9f)				      \
      {									      \
	/* Half-width katakana.  */					      \
	if (__glibc_unlikely (outptr + 1 >= outend))			      \
	  {								      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
	*outptr++ = 0x8e;						      \
	*outptr++ = ch - 0xfec0;					      \
      }									      \
    else								      \
      {									      \
	uint32_t jch = ucs4_to_jisx0213 (ch);				      \
	if (jch == 0)							      \
	  {								      \
	    UNICODE_TAG_HANDLER (ch, 4);				      \
									      \
	    /* Illegal character.  */					      \
	    STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
	  }								      \
									      \
	if (jch & 0x0080)						      \
	  {								      \
	    /* A possible match in comp_table_data.  We have to buffer it.  */\
									      \
	    /* We know it's a JISX 0213 plane 1 character.  */		      \
	    assert ((jch & 0x8000) == 0);				      \
									      \
	    *statep = (jch | 0x8080) << 3;				      \
	    inptr += 4;							      \
	    continue;							      \
	  }								      \
									      \
	if (jch & 0x8000)						      \
	  {								      \
	    /* JISX 0213 plane 2.  */					      \
	    if (__glibc_unlikely (outptr + 2 >= outend))		      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    *outptr++ = 0x8f;						      \
	  }								      \
	else								      \
	  {								      \
	    /* JISX 0213 plane 1.  */					      \
	    if (__glibc_unlikely (outptr + 1 >= outend))		      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	  }								      \
	*outptr++ = (jch >> 8) | 0x80;					      \
	*outptr++ = (jch & 0xff) | 0x80;				      \
      }									      \
									      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
