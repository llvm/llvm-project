/* Conversion from and to TSCII.
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
#include <assert.h>

/* TSCII is an 8-bit encoding consisting of:
   0x00..0x7F:       ASCII
   0x80..0x90, 0x95..0x9F, 0xAB..0xFE:
                     Tamil letters and glyphs
   0xA1..0xA5, 0xAA: Tamil combining letters (after the base character)
   0xA6..0xA8:       Tamil combining letters (before the base character)
   0x91..0x94:       Punctuation
   0xA9:             Symbols
*/

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"TSCII//"
#define FROM_LOOP		from_tscii
#define TO_LOOP			to_tscii
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define ONE_DIRECTION		0
#define FROM_LOOP_MIN_NEEDED_FROM	1
#define FROM_LOOP_MAX_NEEDED_FROM	2
#define FROM_LOOP_MIN_NEEDED_TO		4
#define FROM_LOOP_MAX_NEEDED_TO	       16
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


/* During TSCII to UCS-4 conversion, the COUNT element of the state contains
   the last UCS-4 character to be output, shifted by 8 bits, and an encoded
   representation of additional UCS-4 characters to be output (if any),
   shifted by 4 bits.  This character can be:
     0x0000                   Nothing pending.
     0x0BCD                   Pending VIRAMA sign. If bit 3 is set, it may be
                              omitted if followed by a vowel sign U or UU.
     0x0BC6, 0x0BC7, 0x0BC8   Pending vowel sign.  Bit 3 is set after the
                              consonant was seen.
     Other                    Bit 3 always cleared.  */

/* During UCS-4 to TSCII conversion, the COUNT element of the state contains
   the last byte (or sometimes the last two bytes) to be output, shifted by
   3 bits. This can be:
     0x00                     Nothing pending.
     0xB8..0xC9, 0x83..0x86   A consonant.
     0xEC, 0x8A               A consonant with VIRAMA sign (final or joining).
     0x87, 0xC38A             Two consonants combined through a VIRAMA sign. */

/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  if (data->__statep->__count != 0)					      \
    {									      \
      if (FROM_DIRECTION)						      \
	{								      \
	  do								      \
	    {								      \
	      if (__glibc_unlikely (outbuf + 4 > outend))		      \
		{							      \
		  /* We don't have enough room in the output buffer.  */      \
		  status = __GCONV_FULL_OUTPUT;				      \
		  break;						      \
		}							      \
	      /* Write out the pending character.  */			      \
	      *((uint32_t *) outbuf) = data->__statep->__count >> 8;	      \
	      outbuf += sizeof (uint32_t);				      \
	      /* Retrieve the successor state.  */			      \
	      data->__statep->__count =					      \
		tscii_next_state[(data->__statep->__count >> 4) & 0x0f];      \
	    }								      \
	  while (data->__statep->__count != 0);				      \
	}								      \
      else								      \
	{								      \
	  uint32_t last = data->__statep->__count >> 3;			      \
	  if (__glibc_unlikely (last >> 8))				      \
	    {								      \
	      /* Write out the last character, two bytes.  */		      \
	      if (__glibc_likely (outbuf + 2 <= outend))		      \
		{							      \
		  *outbuf++ = last & 0xff;				      \
		  *outbuf++ = (last >> 8) & 0xff;			      \
		  data->__statep->__count = 0;				      \
		}							      \
	      else							      \
		/* We don't have enough room in the output buffer.  */	      \
		status = __GCONV_FULL_OUTPUT;				      \
	    }								      \
	  else								      \
	    {								      \
	      /* Write out the last character, a single byte.  */	      \
	      if (__glibc_likely (outbuf < outend))			      \
		{							      \
		  *outbuf++ = last & 0xff;				      \
		  data->__statep->__count = 0;				      \
		}							      \
	      else							      \
		/* We don't have enough room in the output buffer.  */	      \
		status = __GCONV_FULL_OUTPUT;				      \
	    }								      \
	}								      \
    }


/* First define the conversion function from TSCII to UCS-4.  */

static const uint16_t tscii_to_ucs4[128][2] =
  {
    { 0x0BE6,      0 },
    { 0x0BE7,      0 },
    {      0,      0 },	/* 0x82 - maps to <U0BB8><U0BCD><U0BB0><U0BC0> */
    { 0x0B9C,      0 },
    { 0x0BB7,      0 },
    { 0x0BB8,      0 },
    { 0x0BB9,      0 },
    {      0,      0 },	/* 0x87 - maps to <U0B95><U0BCD><U0BB7> */
    { 0x0B9C, 0x0BCD },
    { 0x0BB7, 0x0BCD },
    {      0,      0 }, /* 0x8a - maps to <U0BB8> and buffers <U0BCD> */
    {      0,      0 }, /* 0x8b - maps to <U0BB9> and buffers <U0BCD> */
    {      0,      0 },	/* 0x8c - maps to <U0B95><U0BCD><U0BB7><U0BCD> */
    { 0x0BE8,      0 },
    { 0x0BE9,      0 },
    { 0x0BEA,      0 },
    { 0x0BEB,      0 },
    { 0x2018,      0 },
    { 0x2019,      0 },
    { 0x201C,      0 },
    { 0x201D,      0 },
    { 0x0BEC,      0 },
    { 0x0BED,      0 },
    { 0x0BEE,      0 },
    { 0x0BEF,      0 },
    { 0x0B99, 0x0BC1 },
    { 0x0B9E, 0x0BC1 },
    { 0x0B99, 0x0BC2 },
    { 0x0B9E, 0x0BC2 },
    { 0x0BF0,      0 },
    { 0x0BF1,      0 },
    { 0x0BF2,      0 },
    {      0,      0 },	/* 0xa0 - unmapped */
    { 0x0BBE,      0 },
    { 0x0BBF,      0 },
    { 0x0BC0,      0 },
    { 0x0BC1,      0 },
    { 0x0BC2,      0 },
    {      0,      0 }, /* 0xa6 - buffers <U0BC6> */
    {      0,      0 }, /* 0xa7 - buffers <U0BC7> */
    {      0,      0 }, /* 0xa8 - buffers <U0BC8> */
    { 0x00A9,      0 },
    { 0x0BD7,      0 },
    { 0x0B85,      0 },
    { 0x0B86,      0 },
    { 0x0B87,      0 },
    { 0x0B88,      0 },
    { 0x0B89,      0 },
    { 0x0B8A,      0 },
    { 0x0B8E,      0 },
    { 0x0B8F,      0 },
    { 0x0B90,      0 },
    { 0x0B92,      0 },
    { 0x0B93,      0 },
    { 0x0B94,      0 },
    { 0x0B83,      0 },
    { 0x0B95,      0 },
    { 0x0B99,      0 },
    { 0x0B9A,      0 },
    { 0x0B9E,      0 },
    { 0x0B9F,      0 },
    { 0x0BA3,      0 },
    { 0x0BA4,      0 },
    { 0x0BA8,      0 },
    { 0x0BAA,      0 },
    { 0x0BAE,      0 },
    { 0x0BAF,      0 },
    { 0x0BB0,      0 },
    { 0x0BB2,      0 },
    { 0x0BB5,      0 },
    { 0x0BB4,      0 },
    { 0x0BB3,      0 },
    { 0x0BB1,      0 },
    { 0x0BA9,      0 },
    { 0x0B9F, 0x0BBF },
    { 0x0B9F, 0x0BC0 },
    { 0x0B95, 0x0BC1 },
    { 0x0B9A, 0x0BC1 },
    { 0x0B9F, 0x0BC1 },
    { 0x0BA3, 0x0BC1 },
    { 0x0BA4, 0x0BC1 },
    { 0x0BA8, 0x0BC1 },
    { 0x0BAA, 0x0BC1 },
    { 0x0BAE, 0x0BC1 },
    { 0x0BAF, 0x0BC1 },
    { 0x0BB0, 0x0BC1 },
    { 0x0BB2, 0x0BC1 },
    { 0x0BB5, 0x0BC1 },
    { 0x0BB4, 0x0BC1 },
    { 0x0BB3, 0x0BC1 },
    { 0x0BB1, 0x0BC1 },
    { 0x0BA9, 0x0BC1 },
    { 0x0B95, 0x0BC2 },
    { 0x0B9A, 0x0BC2 },
    { 0x0B9F, 0x0BC2 },
    { 0x0BA3, 0x0BC2 },
    { 0x0BA4, 0x0BC2 },
    { 0x0BA8, 0x0BC2 },
    { 0x0BAA, 0x0BC2 },
    { 0x0BAE, 0x0BC2 },
    { 0x0BAF, 0x0BC2 },
    { 0x0BB0, 0x0BC2 },
    { 0x0BB2, 0x0BC2 },
    { 0x0BB5, 0x0BC2 },
    { 0x0BB4, 0x0BC2 },
    { 0x0BB3, 0x0BC2 },
    { 0x0BB1, 0x0BC2 },
    { 0x0BA9, 0x0BC2 },
    { 0x0B95, 0x0BCD },
    { 0x0B99, 0x0BCD },
    { 0x0B9A, 0x0BCD },
    { 0x0B9E, 0x0BCD },
    { 0x0B9F, 0x0BCD },
    { 0x0BA3, 0x0BCD },
    { 0x0BA4, 0x0BCD },
    { 0x0BA8, 0x0BCD },
    { 0x0BAA, 0x0BCD },
    { 0x0BAE, 0x0BCD },
    { 0x0BAF, 0x0BCD },
    { 0x0BB0, 0x0BCD },
    { 0x0BB2, 0x0BCD },
    { 0x0BB5, 0x0BCD },
    { 0x0BB4, 0x0BCD },
    { 0x0BB3, 0x0BCD },
    { 0x0BB1, 0x0BCD },
    { 0x0BA9, 0x0BCD },
    { 0x0B87,      0 },
    {      0,      0 }	/* 0xff - unmapped */
  };

static const uint32_t tscii_next_state[6] =
  {
    /* 0 means no more pending Unicode characters.  */
    0,
    /* 1 means <U0BB7>.  */
    (0x0BB7 << 8),
    /* 2 means <U0BC0>.  */
    (0x0BC0 << 8),
    /* 3 means <U0BCD>.  */
    (0x0BCD << 8),
    /* 4 means <U0BB0><U0BC0>.  */
    (0x0BB0 << 8) + (2 << 4),
    /* 5 means <U0BB7><U0BCD>.  */
    (0x0BB7 << 8) + (3 << 4)
  };

#define MIN_NEEDED_INPUT	FROM_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	FROM_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	FROM_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	FROM_LOOP_MAX_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
									      \
    if ((*statep >> 8) != 0)						      \
      {									      \
	/* Attempt to combine the last character with this one.  */	      \
	uint32_t last = *statep >> 8;					      \
									      \
	if (last == 0x0BCD && (*statep & (1 << 3)))			      \
	  {								      \
	    if (ch == 0xa4 || ch == 0xa5)				      \
	      {								      \
		ch += 0xb1d;						      \
		/* Now ch = 0x0BC1 or ch = 0x0BC2.  */			      \
		put32 (outptr, ch);					      \
		outptr += 4;						      \
		*statep = 0;						      \
		inptr++;						      \
		continue;						      \
	      }								      \
	  }								      \
	else if (last >= 0x0BC6 && last <= 0x0BC8)			      \
	  {								      \
	    if ((last == 0x0BC6 && ch == 0xa1)				      \
		|| (last == 0x0BC7 && (ch == 0xa1 || ch == 0xaa)))	      \
	      {								      \
		ch = last + 4 + (ch != 0xa1);				      \
		/* Now ch = 0x0BCA or ch = 0x0BCB or ch = 0x0BCC.  */	      \
		put32 (outptr, ch);					      \
		outptr += 4;						      \
		*statep = 0;						      \
		inptr++;						      \
		continue;						      \
	      }								      \
	    if ((ch >= 0xb8 && ch <= 0xc9) && (*statep & (1 << 3)) == 0)      \
	      {								      \
		ch = tscii_to_ucs4[ch - 0x80][0];			      \
		put32 (outptr, ch);					      \
		outptr += 4;						      \
		*statep |= 1 << 3;					      \
		inptr++;						      \
		continue;						      \
	      }								      \
	  }								      \
									      \
	do								      \
	  {								      \
	    /* Output the buffered character.  */			      \
	    put32 (outptr, last);					      \
	    outptr += 4;						      \
	    /* Retrieve the successor state.  */			      \
	    *statep = tscii_next_state[(*statep >> 4) & 0x0f];		      \
	  }								      \
	while (*statep != 0 && __builtin_expect (outptr + 4 <= outend, 1));   \
									      \
	if (*statep != 0)						      \
	  {								      \
	    /* We don't have enough room in the output buffer.		      \
	       Tell the caller why we terminate the loop.  */		      \
	    result = __GCONV_FULL_OUTPUT;				      \
	    break;							      \
	  }								      \
									      \
	continue;							      \
      }									      \
									      \
    if (ch < 0x80)							      \
      {									      \
	/* Plain ASCII character.  */					      \
	put32 (outptr, ch);						      \
	outptr += 4;							      \
      }									      \
    else								      \
      {									      \
	/* Tamil character.  */						      \
	uint32_t u1 = tscii_to_ucs4[ch - 0x80][0];			      \
									      \
	if (u1 != 0)							      \
	  {								      \
	    uint32_t u2 = tscii_to_ucs4[ch - 0x80][1];			      \
									      \
	    inptr++;							      \
									      \
	    put32 (outptr, u1);						      \
	    outptr += 4;						      \
									      \
	    if (u2 != 0)						      \
	      {								      \
		/* See whether we have room for two characters.  Otherwise    \
		   store only the first character now, and put the second     \
		   one into the queue.  */				      \
		if (__glibc_unlikely (outptr + 4 > outend))		      \
		  {							      \
		    *statep = u2 << 8;					      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
		put32 (outptr, u2);					      \
		outptr += 4;						      \
	      }								      \
	    continue;							      \
	  }								      \
	/* Special handling of a few Tamil characters.  */		      \
	else if (ch == 0xa6 || ch == 0xa7 || ch == 0xa8)		      \
	  {								      \
	    ch += 0x0b20;						      \
	    /* Now ch = 0x0BC6 or ch = 0x0BC7 or ch = 0x0BC8.  */	      \
	    *statep = ch << 8;						      \
	    inptr++;							      \
	    continue;							      \
	  }								      \
	else if (ch == 0x8a || ch == 0x8b)				      \
	  {								      \
	    ch += 0x0b2e;						      \
	    /* Now ch = 0x0BB8 or ch = 0x0BB9.  */			      \
	    put32 (outptr, ch);						      \
	    outptr += 4;						      \
	    *statep = (0x0BCD << 8) + (1 << 3);				      \
	    inptr++;							      \
	    continue;							      \
	  }								      \
	else if (ch == 0x82)						      \
	  {								      \
	    /* Output <U0BB8><U0BCD><U0BB0><U0BC0>, if we have room for	      \
	       four characters.  */					      \
	    inptr++;							      \
	    put32 (outptr, 0x0BB8);					      \
	    outptr += 4;						      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		*statep = (0x0BCD << 8) + (4 << 4);			      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    put32 (outptr, 0x0BCD);					      \
	    outptr += 4;						      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		*statep = (0x0BB0 << 8) + (2 << 4);			      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    put32 (outptr, 0x0BB0);					      \
	    outptr += 4;						      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		*statep = (0x0BC0 << 8);				      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    put32 (outptr, 0x0BC0);					      \
	    outptr += 4;						      \
	    continue;							      \
	  }								      \
	else if (ch == 0x87)						      \
	  {								      \
	    /* Output <U0B95><U0BCD><U0BB7>, if we have room for	      \
	       three characters.  */					      \
	    inptr++;							      \
	    put32 (outptr, 0x0B95);					      \
	    outptr += 4;						      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		*statep = (0x0BCD << 8) + (1 << 4);			      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    put32 (outptr, 0x0BCD);					      \
	    outptr += 4;						      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		*statep = (0x0BB7 << 8);				      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    put32 (outptr, 0x0BB7);					      \
	    outptr += 4;						      \
	    continue;							      \
	  }								      \
	else if (ch == 0x8c)						      \
	  {								      \
	    /* Output <U0B95><U0BCD><U0BB7><U0BCD>, if we have room for	      \
	       four characters.  */					      \
	    inptr++;							      \
	    put32 (outptr, 0x0B95);					      \
	    outptr += 4;						      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		*statep = (0x0BCD << 8) + (5 << 4);			      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    put32 (outptr, 0x0BCD);					      \
	    outptr += 4;						      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		*statep = (0x0BB7 << 8) + (3 << 4);			      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    put32 (outptr, 0x0BB7);					      \
	    outptr += 4;						      \
	    if (__glibc_unlikely (outptr + 4 > outend))			      \
	      {								      \
		*statep = (0x0BCD << 8);				      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    put32 (outptr, 0x0BCD);					      \
	    outptr += 4;						      \
	    continue;							      \
	  }								      \
	else								      \
	  {								      \
	    /* This is illegal.  */					      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr++;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#include <iconv/loop.c>


/* Next, define the other direction, from UCS-4 to TSCII.  */

static const uint8_t ucs4_to_tscii[128] =
  {
       0,    0,    0, 0xb7,    0, 0xab, 0xac, 0xfe, /* 0x0B80..0x0B87 */
    0xae, 0xaf, 0xb0,    0,    0,    0, 0xb1, 0xb2, /* 0x0B88..0x0B8F */
    0xb3,    0, 0xb4, 0xb5, 0xb6, 0xb8,    0,    0, /* 0x0B90..0x0B97 */
       0, 0xb9, 0xba,    0, 0x83,    0, 0xbb, 0xbc, /* 0x0B98..0x0B9F */
       0,    0,    0, 0xbd, 0xbe,    0,    0,    0, /* 0x0BA0..0x0BA7 */
    0xbf, 0xc9, 0xc0,    0,    0,    0, 0xc1, 0xc2, /* 0x0BA8..0x0BAF */
    0xc3, 0xc8, 0xc4, 0xc7, 0xc6, 0xc5,    0, 0x84, /* 0x0BB0..0x0BB7 */
    0x85, 0x86,    0,    0,    0,    0, 0xa1, 0xa2, /* 0x0BB8..0x0BBF */
    0xa3, 0xa4, 0xa5,    0,    0,    0, 0xa6, 0xa7, /* 0x0BC0..0x0BC7 */
    0xa8,    0,    0,    0,    0,    0,    0,    0, /* 0x0BC8..0x0BCF */
       0,    0,    0,    0,    0,    0,    0, 0xaa, /* 0x0BD0..0x0BD7 */
       0,    0,    0,    0,    0,    0,    0,    0, /* 0x0BD8..0x0BDF */
       0,    0,    0,    0,    0,    0, 0x80, 0x81, /* 0x0BE0..0x0BE7 */
    0x8d, 0x8e, 0x8f, 0x90, 0x95, 0x96, 0x97, 0x98, /* 0x0BE8..0x0BEF */
    0x9d, 0x9e, 0x9f,    0,    0,    0,    0,    0, /* 0x0BF0..0x0BF7 */
       0,    0,    0,    0,    0,    0,    0,    0  /* 0x0BF8..0x0BFF */
  };

static const uint8_t consonant_with_u[18] =
  {
    0xcc, 0x99, 0xcd, 0x9a, 0xce, 0xcf, 0xd0, 0xd1, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb
  };

static const uint8_t consonant_with_uu[18] =
  {
    0xdc, 0x9b, 0xdd, 0x9c, 0xde, 0xdf, 0xe0, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb
  };

static const uint8_t consonant_with_virama[18] =
  {
    0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf2, 0xf3, 0xf4,
    0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd
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
	uint32_t last = *statep >> 3;					      \
									      \
	if (last >= 0xb8 && last <= 0xc9)				      \
	  {								      \
	    if (ch == 0x0BC1)						      \
	      {								      \
		*outptr++ = consonant_with_u[last - 0xb8];		      \
		*statep = 0;						      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	    if (ch == 0x0BC2)						      \
	      {								      \
		*outptr++ = consonant_with_uu[last - 0xb8];		      \
		*statep = 0;						      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	    if (ch == 0x0BC6)						      \
	      {								      \
		if (__glibc_likely (outptr + 2 <= outend))		      \
		  {							      \
		    *outptr++ = 0xa6;					      \
		    *outptr++ = last;					      \
		    *statep = 0;					      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
		else							      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
	      }								      \
	    if (ch == 0x0BC7)						      \
	      {								      \
		if (__glibc_likely (outptr + 2 <= outend))		      \
		  {							      \
		    *outptr++ = 0xa7;					      \
		    *outptr++ = last;					      \
		    *statep = 0;					      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
		else							      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
	      }								      \
	    if (ch == 0x0BC8)						      \
	      {								      \
		if (__glibc_likely (outptr + 2 <= outend))		      \
		  {							      \
		    *outptr++ = 0xa8;					      \
		    *outptr++ = last;					      \
		    *statep = 0;					      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
		else							      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
	      }								      \
	    if (ch == 0x0BCA)						      \
	      {								      \
		if (__glibc_likely (outptr + 3 <= outend))		      \
		  {							      \
		    *outptr++ = 0xa6;					      \
		    *outptr++ = last;					      \
		    *outptr++ = 0xa1;					      \
		    *statep = 0;					      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
		else							      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
	      }								      \
	    if (ch == 0x0BCB)						      \
	      {								      \
		if (__glibc_likely (outptr + 3 <= outend))		      \
		  {							      \
		    *outptr++ = 0xa7;					      \
		    *outptr++ = last;					      \
		    *outptr++ = 0xa1;					      \
		    *statep = 0;					      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
		else							      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
	      }								      \
	    if (ch == 0x0BCC)						      \
	      {								      \
		if (__glibc_likely (outptr + 3 <= outend))		      \
		  {							      \
		    *outptr++ = 0xa7;					      \
		    *outptr++ = last;					      \
		    *outptr++ = 0xaa;					      \
		    *statep = 0;					      \
		    inptr += 4;						      \
		    continue;						      \
		  }							      \
		else							      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
	      }								      \
	    if (ch == 0x0BCD)						      \
	      {								      \
		if (last != 0xb8)					      \
		  {							      \
		    *outptr++ = consonant_with_virama[last - 0xb8];	      \
		    *statep = 0;					      \
		  }							      \
		else							      \
		  *statep = 0xec << 3;					      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	    if (last == 0xbc && (ch == 0x0BBF || ch == 0x0BC0))		      \
	      {								      \
		*outptr++ = ch - 0x0af5;				      \
		*statep = 0;						      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	  }								      \
	else if (last >= 0x83 && last <= 0x86)				      \
	  {								      \
	    if (last >= 0x85 && (ch == 0x0BC1 || ch == 0x0BC2))		      \
	      {								      \
		*outptr++ = last + 5;					      \
		*statep = 0;						      \
		continue;						      \
	      }								      \
	    if (ch == 0x0BCD)						      \
	      {								      \
		if (last != 0x85)					      \
		  {							      \
		    *outptr++ = last + 5;				      \
		    *statep = 0;					      \
		  }							      \
		else							      \
		  *statep = 0x8a << 3;					      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	  }								      \
	else if (last == 0xec)						      \
	  {								      \
	    if (ch == 0x0BB7)						      \
	      {								      \
		*statep = 0x87 << 3;					      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	  }								      \
	else if (last == 0x8a)						      \
	  {								      \
	    if (ch == 0x0BB0)						      \
	      {								      \
		*statep = 0xc38a << 3;					      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	  }								      \
	else if (last == 0x87)						      \
	  {								      \
	    if (ch == 0x0BCD)						      \
	      {								      \
		*outptr++ = 0x8c;					      \
		*statep = 0;						      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	  }								      \
	else								      \
	  {								      \
	    assert (last == 0xc38a);					      \
	    if (ch == 0x0BC0)						      \
	      {								      \
		*outptr++ = 0x82;					      \
		*statep = 0;						      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
	  }								      \
									      \
	/* Output the buffered character.  */				      \
	if (__glibc_unlikely (last >> 8))				      \
	  {								      \
	    if (__glibc_likely (outptr + 2 <= outend))			      \
	      {								      \
		*outptr++ = last & 0xff;				      \
		*outptr++ = (last >> 8) & 0xff;				      \
	      }								      \
	    else							      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	  }								      \
        else								      \
	  *outptr++ = last & 0xff;					      \
	*statep = 0;							      \
	continue;							      \
      }									      \
									      \
    if (ch < 0x80)							      \
      /* Plain ASCII character.  */					      \
      *outptr++ = ch;							      \
    else if (ch >= 0x0B80 && ch <= 0x0BFF)				      \
      {									      \
	/* Tamil character.  */						      \
	uint8_t t = ucs4_to_tscii[ch - 0x0B80];				      \
									      \
	if (t != 0)							      \
	  {								      \
	    if ((t >= 0xb8 && t <= 0xc9) || (t >= 0x83 && t <= 0x86))	      \
	      *statep = (uint32_t) t << 3;				      \
	    else							      \
	      *outptr++ = t;						      \
	  }								      \
	else if (ch >= 0x0BCA && ch <= 0x0BCC)				      \
	  {								      \
	    /* See whether we have room for two bytes.  */		      \
	    if (__glibc_likely (outptr + 2 <= outend))			      \
	      {								      \
		*outptr++ = (ch == 0x0BCA ? 0xa6 : 0xa7);		      \
		*outptr++ = (ch != 0x0BCC ? 0xa1 : 0xaa);		      \
	      }								      \
	    else							      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	  }								      \
	else								      \
	  {								      \
	    /* Illegal character.  */					      \
	    STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
	  }								      \
      }									      \
    else if (ch == 0x00A9)						      \
      *outptr++ = ch;							      \
    else if (ch == 0x2018 || ch == 0x2019)				      \
      *outptr++ = ch - 0x1f87;						      \
    else if (ch == 0x201C || ch == 0x201D)				      \
      *outptr++ = ch - 0x1f89;						      \
    else								      \
      {									      \
	UNICODE_TAG_HANDLER (ch, 4);					      \
									      \
	/* Illegal character.  */					      \
	STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
