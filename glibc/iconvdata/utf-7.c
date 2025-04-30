/* Conversion module for UTF-7.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bruno Haible <haible@clisp.cons.org>, 2000.

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

/* UTF-7 is a legacy encoding used for transmitting Unicode within the
   ASCII character set, used primarily by mail agents.  New programs
   are encouraged to use UTF-8 instead.

   UTF-7 is specified in RFC 2152 (and old RFC 1641, RFC 1642).  The
   original Base64 encoding is defined in RFC 2045.  */

#include <dlfcn.h>
#include <gconv.h>
#include <stdint.h>
#include <stdlib.h>


/* Define this to 1 if you want the so-called "optional direct" characters
      ! " # $ % & * ; < = > @ [ ] ^ _ ` { | }
   to be encoded. Define to 0 if you want them to be passed straight
   through, like the so-called "direct" characters.
   We set this to 1 because it's safer.
 */
#define UTF7_ENCODE_OPTIONAL_CHARS 1


/* The set of "direct characters":
   A-Z a-z 0-9 ' ( ) , - . / : ? space tab lf cr
*/

static const unsigned char direct_tab[128 / 8] =
  {
    0x00, 0x26, 0x00, 0x00, 0x81, 0xf3, 0xff, 0x87,
    0xfe, 0xff, 0xff, 0x07, 0xfe, 0xff, 0xff, 0x07
  };

static int
isdirect (uint32_t ch)
{
  return (ch < 128 && ((direct_tab[ch >> 3] >> (ch & 7)) & 1));
}


/* The set of "direct and optional direct characters":
   A-Z a-z 0-9 ' ( ) , - . / : ? space tab lf cr
   ! " # $ % & * ; < = > @ [ ] ^ _ ` { | }
*/

static const unsigned char xdirect_tab[128 / 8] =
  {
    0x00, 0x26, 0x00, 0x00, 0xff, 0xf7, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0x3f
  };

static int
isxdirect (uint32_t ch)
{
  return (ch < 128 && ((xdirect_tab[ch >> 3] >> (ch & 7)) & 1));
}


/* The set of "extended base64 characters":
   A-Z a-z 0-9 + / -
*/

static const unsigned char xbase64_tab[128 / 8] =
  {
    0x00, 0x00, 0x00, 0x00, 0x00, 0xa8, 0xff, 0x03,
    0xfe, 0xff, 0xff, 0x07, 0xfe, 0xff, 0xff, 0x07
  };

static int
isxbase64 (uint32_t ch)
{
  return (ch < 128 && ((xbase64_tab[ch >> 3] >> (ch & 7)) & 1));
}


/* Converts a value in the range 0..63 to a base64 encoded char.  */
static unsigned char
base64 (unsigned int i)
{
  if (i < 26)
    return i + 'A';
  else if (i < 52)
    return i - 26 + 'a';
  else if (i < 62)
    return i - 52 + '0';
  else if (i == 62)
    return '+';
  else if (i == 63)
    return '/';
  else
    abort ();
}


/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"UTF-7//"
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define FROM_LOOP		from_utf7_loop
#define TO_LOOP			to_utf7_loop
#define MIN_NEEDED_FROM		1
#define MAX_NEEDED_FROM		6
#define MIN_NEEDED_TO		4
#define MAX_NEEDED_TO		4
#define ONE_DIRECTION		0
#define PREPARE_LOOP \
  mbstate_t saved_state;						      \
  mbstate_t *statep = data->__statep;
#define EXTRA_LOOP_ARGS		, statep


/* Since we might have to reset input pointer we must be able to save
   and restore the state.  */
#define SAVE_RESET_STATE(Save) \
  if (Save)								      \
    saved_state = *statep;						      \
  else									      \
    *statep = saved_state


/* First define the conversion function from UTF-7 to UCS4.
   The state is structured as follows:
     __count bit 2..0: zero
     __count bit 8..3: shift
     __wch: data
   Precise meaning:
     shift      data
       0         --          not inside base64 encoding
     1..32  XX..XX00..00     inside base64, (32 - shift) bits pending
   This state layout is simpler than relying on STORE_REST/UNPACK_BYTES.

   When shift = 0, __wch needs to store at most one lookahead byte (see
   __GCONV_INCOMPLETE_INPUT below).
*/
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	MAX_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint_fast8_t ch = *inptr;						      \
									      \
    if ((statep->__count >> 3) == 0)					      \
      {									      \
	/* base64 encoding inactive.  */				      \
	if (isxdirect (ch))						      \
	  {								      \
	    inptr++;							      \
	    put32 (outptr, ch);						      \
	    outptr += 4;						      \
	  }								      \
	else if (__glibc_likely (ch == '+'))				      \
	  {								      \
	    if (__glibc_unlikely (inptr + 2 > inend))			      \
	      {								      \
		/* Not enough input available.  */			      \
		result = __GCONV_INCOMPLETE_INPUT;			      \
		break;							      \
	      }								      \
	    if (inptr[1] == '-')					      \
	      {								      \
		inptr += 2;						      \
		put32 (outptr, ch);					      \
		outptr += 4;						      \
	      }								      \
	    else							      \
	      {								      \
		/* Switch into base64 mode.  */				      \
		inptr++;						      \
		statep->__count = (32 << 3);				      \
		statep->__value.__wch = 0;				      \
	      }								      \
	  }								      \
	else								      \
	  {								      \
	    /* The input is invalid.  */				      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
      }									      \
    else								      \
      {									      \
	/* base64 encoding active.  */					      \
	uint32_t i;							      \
	int shift;							      \
									      \
	if (ch >= 'A' && ch <= 'Z')					      \
	  i = ch - 'A';							      \
	else if (ch >= 'a' && ch <= 'z')				      \
	  i = ch - 'a' + 26;						      \
	else if (ch >= '0' && ch <= '9')				      \
	  i = ch - '0' + 52;						      \
	else if (ch == '+')						      \
	  i = 62;							      \
	else if (ch == '/')						      \
	  i = 63;							      \
	else								      \
	  {								      \
	    /* Terminate base64 encoding.  */				      \
									      \
	    /* If accumulated data is nonzero, the input is invalid.  */      \
	    /* Also, partial UTF-16 characters are invalid.  */		      \
	    if (__builtin_expect (statep->__value.__wch != 0, 0)	      \
		|| __builtin_expect ((statep->__count >> 3) <= 26, 0))	      \
	      {								      \
		STANDARD_FROM_LOOP_ERR_HANDLER ((statep->__count = 0, 1));    \
	      }								      \
									      \
	    if (ch == '-')						      \
	      inptr++;							      \
									      \
	    statep->__count = 0;					      \
	    continue;							      \
	  }								      \
									      \
	/* Concatenate the base64 integer i to the accumulator.  */	      \
	shift = (statep->__count >> 3);					      \
	if (shift > 6)							      \
	  {								      \
	    uint32_t wch;						      \
									      \
	    shift -= 6;							      \
	    wch = statep->__value.__wch | (i << shift);			      \
									      \
	    if (shift <= 16 && shift > 10)				      \
	      {								      \
		/* An UTF-16 character has just been completed.  */	      \
		uint32_t wc1 = wch >> 16;				      \
									      \
		/* UTF-16: When we see a High Surrogate, we must also decode  \
		   the following Low Surrogate. */			      \
		if (!(wc1 >= 0xd800 && wc1 < 0xdc00))			      \
		  {							      \
		    wch = wch << 16;					      \
		    shift += 16;					      \
		    put32 (outptr, wc1);				      \
		    outptr += 4;					      \
		  }							      \
	      }								      \
	    else if (shift <= 10 && shift > 4)				      \
	      {								      \
		/* After a High Surrogate, verify that the next 16 bit	      \
		   indeed form a Low Surrogate.  */			      \
		uint32_t wc2 = wch & 0xffff;				      \
									      \
		if (! __builtin_expect (wc2 >= 0xdc00 && wc2 < 0xe000, 1))    \
		  {							      \
		    STANDARD_FROM_LOOP_ERR_HANDLER ((statep->__count = 0, 1));\
		  }							      \
	      }								      \
									      \
	    statep->__value.__wch = wch;				      \
	  }								      \
	else								      \
	  {								      \
	    /* An UTF-16 surrogate pair has just been completed.  */	      \
	    uint32_t wc1 = (uint32_t) statep->__value.__wch >> 16;	      \
	    uint32_t wc2 = ((uint32_t) statep->__value.__wch & 0xffff)	      \
			   | (i >> (6 - shift));			      \
									      \
	    statep->__value.__wch = (i << shift) << 26;			      \
	    shift += 26;						      \
									      \
	    assert (wc1 >= 0xd800 && wc1 < 0xdc00);			      \
	    assert (wc2 >= 0xdc00 && wc2 < 0xe000);			      \
	    put32 (outptr,						      \
		   0x10000 + ((wc1 - 0xd800) << 10) + (wc2 - 0xdc00));	      \
	    outptr += 4;						      \
	  }								      \
									      \
	statep->__count = shift << 3;					      \
									      \
	/* Now that we digested the input increment the input pointer.  */    \
	inptr++;							      \
      }									      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, mbstate_t *statep
#include <iconv/loop.c>


/* Next, define the conversion from UCS4 to UTF-7.
   The state is structured as follows:
     __count bit 2..0: zero
     __count bit 4..3: shift
     __count bit 8..5: data
   Precise meaning:
     shift      data
       0         0           not inside base64 encoding
       1         0           inside base64, no pending bits
       2       XX00          inside base64, 2 bits known for next byte
       3       XXXX          inside base64, 4 bits known for next byte

   __count bit 2..0 and __wch are always zero, because this direction
   never returns __GCONV_INCOMPLETE_INPUT.
*/
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MAX_NEEDED_INPUT	MAX_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch = get32 (inptr);					      \
									      \
    if ((statep->__count & 0x18) == 0)					      \
      {									      \
	/* base64 encoding inactive */					      \
	if (UTF7_ENCODE_OPTIONAL_CHARS ? isdirect (ch) : isxdirect (ch))      \
	  {								      \
	    *outptr++ = (unsigned char) ch;				      \
	  }								      \
	else								      \
	  {								      \
	    size_t count;						      \
									      \
	    if (ch == '+')						      \
	      count = 2;						      \
	    else if (ch < 0x10000)					      \
	      count = 3;						      \
	    else if (ch < 0x110000)					      \
	      count = 6;						      \
	    else							      \
	      STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
									      \
	    if (__glibc_unlikely (outptr + count > outend))		      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    *outptr++ = '+';						      \
	    if (ch == '+')						      \
	      *outptr++ = '-';						      \
	    else if (ch < 0x10000)					      \
	      {								      \
		*outptr++ = base64 (ch >> 10);				      \
		*outptr++ = base64 ((ch >> 4) & 0x3f);			      \
		statep->__count = ((ch & 15) << 5) | (3 << 3);		      \
	      }								      \
	    else if (ch < 0x110000)					      \
	      {								      \
		uint32_t ch1 = 0xd800 + ((ch - 0x10000) >> 10);		      \
		uint32_t ch2 = 0xdc00 + ((ch - 0x10000) & 0x3ff);	      \
									      \
		ch = (ch1 << 16) | ch2;					      \
		*outptr++ = base64 (ch >> 26);				      \
		*outptr++ = base64 ((ch >> 20) & 0x3f);			      \
		*outptr++ = base64 ((ch >> 14) & 0x3f);			      \
		*outptr++ = base64 ((ch >> 8) & 0x3f);			      \
		*outptr++ = base64 ((ch >> 2) & 0x3f);			      \
		statep->__count = ((ch & 3) << 7) | (2 << 3);		      \
	      }								      \
	    else							      \
	      abort ();							      \
	  }								      \
      }									      \
    else								      \
      {									      \
	/* base64 encoding active */					      \
	if (UTF7_ENCODE_OPTIONAL_CHARS ? isdirect (ch) : isxdirect (ch))      \
	  {								      \
	    /* deactivate base64 encoding */				      \
	    size_t count;						      \
									      \
	    count = ((statep->__count & 0x18) >= 0x10) + isxbase64 (ch) + 1;  \
	    if (__glibc_unlikely (outptr + count > outend))		      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    if ((statep->__count & 0x18) >= 0x10)			      \
	      *outptr++ = base64 ((statep->__count >> 3) & ~3);		      \
	    if (isxbase64 (ch))						      \
	      *outptr++ = '-';						      \
	    *outptr++ = (unsigned char) ch;				      \
	    statep->__count = 0;					      \
	  }								      \
	else								      \
	  {								      \
	    size_t count;						      \
									      \
	    if (ch < 0x10000)						      \
	      count = ((statep->__count & 0x18) >= 0x10 ? 3 : 2);	      \
	    else if (ch < 0x110000)					      \
	      count = ((statep->__count & 0x18) >= 0x18 ? 6 : 5);	      \
	    else							      \
	      STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
									      \
	    if (__glibc_unlikely (outptr + count > outend))		      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    if (ch < 0x10000)						      \
	      {								      \
		switch ((statep->__count >> 3) & 3)			      \
		  {							      \
		  case 1:						      \
		    *outptr++ = base64 (ch >> 10);			      \
		    *outptr++ = base64 ((ch >> 4) & 0x3f);		      \
		    statep->__count = ((ch & 15) << 5) | (3 << 3);	      \
		    break;						      \
		  case 2:						      \
		    *outptr++ =						      \
		      base64 (((statep->__count >> 3) & ~3) | (ch >> 12));    \
		    *outptr++ = base64 ((ch >> 6) & 0x3f);		      \
		    *outptr++ = base64 (ch & 0x3f);			      \
		    statep->__count = (1 << 3);				      \
		    break;						      \
		  case 3:						      \
		    *outptr++ =						      \
		      base64 (((statep->__count >> 3) & ~3) | (ch >> 14));    \
		    *outptr++ = base64 ((ch >> 8) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 2) & 0x3f);		      \
		    statep->__count = ((ch & 3) << 7) | (2 << 3);	      \
		    break;						      \
		  default:						      \
		    abort ();						      \
		  }							      \
	      }								      \
	    else if (ch < 0x110000)					      \
	      {								      \
		uint32_t ch1 = 0xd800 + ((ch - 0x10000) >> 10);		      \
		uint32_t ch2 = 0xdc00 + ((ch - 0x10000) & 0x3ff);	      \
									      \
		ch = (ch1 << 16) | ch2;					      \
		switch ((statep->__count >> 3) & 3)			      \
		  {							      \
		  case 1:						      \
		    *outptr++ = base64 (ch >> 26);			      \
		    *outptr++ = base64 ((ch >> 20) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 14) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 8) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 2) & 0x3f);		      \
		    statep->__count = ((ch & 3) << 7) | (2 << 3);	      \
		    break;						      \
		  case 2:						      \
		    *outptr++ =						      \
		      base64 (((statep->__count >> 3) & ~3) | (ch >> 28));    \
		    *outptr++ = base64 ((ch >> 22) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 16) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 10) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 4) & 0x3f);		      \
		    statep->__count = ((ch & 15) << 5) | (3 << 3);	      \
		    break;						      \
		  case 3:						      \
		    *outptr++ =						      \
		      base64 (((statep->__count >> 3) & ~3) | (ch >> 30));    \
		    *outptr++ = base64 ((ch >> 24) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 18) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 12) & 0x3f);		      \
		    *outptr++ = base64 ((ch >> 6) & 0x3f);		      \
		    *outptr++ = base64 (ch & 0x3f);			      \
		    statep->__count = (1 << 3);				      \
		    break;						      \
		  default:						      \
		    abort ();						      \
		  }							      \
	      }								      \
	    else							      \
	      abort ();							      \
	  }								      \
      }									      \
									      \
    /* Now that we wrote the output increment the input pointer.  */	      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, mbstate_t *statep
#include <iconv/loop.c>


/* Since this is a stateful encoding we have to provide code which resets
   the output state to the initial state.  This has to be done during the
   flushing.  */
#define EMIT_SHIFT_TO_INIT \
  if (FROM_DIRECTION)							      \
    /* Nothing to emit.  */						      \
    memset (data->__statep, '\0', sizeof (mbstate_t));			      \
  else									      \
    {									      \
      /* The "to UTF-7" direction.  Flush the remaining bits and terminate    \
	 with a '-' byte.  This will guarantee correct decoding if more	      \
	 UTF-7 encoded text is added afterwards.  */			      \
      int state = data->__statep->__count;				      \
									      \
      if (state & 0x18)							      \
	{								      \
	  /* Deactivate base64 encoding.  */				      \
	  size_t count = ((state & 0x18) >= 0x10) + 1;			      \
									      \
	  if (__glibc_unlikely (outbuf + count > outend))		      \
	    /* We don't have enough room in the output buffer.  */	      \
	    status = __GCONV_FULL_OUTPUT;				      \
	  else								      \
	    {								      \
	      /* Write out the shift sequence.  */			      \
	      if ((state & 0x18) >= 0x10)				      \
		*outbuf++ = base64 ((state >> 3) & ~3);			      \
	      *outbuf++ = '-';						      \
									      \
	      data->__statep->__count = 0;				      \
	    }								      \
	}								      \
      else								      \
	data->__statep->__count = 0;					      \
    }


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
