/* Mapping tables for JOHAB handling.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jungshik Shin <jshin@pantheon.yale.edu>
   and Ulrich Drepper <drepper@cygnus.com>, 1998.

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
#include <ksc5601.h>

/* The table for Bit pattern to Hangul Jamo
   5 bits each are used to encode
   leading consonants(19 + 1 filler), medial vowels(21 + 1 filler)
   and trailing consonants(27 + 1 filler).

   KS C 5601-1992 Annex 3 Table 2
   0 : Filler, -1: invalid, >= 1 : valid

 */
static const int init[32] =
{
  -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
  19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
};
static const int mid[32] =
{
  -1, -1, 0, 1, 2, 3, 4, 5,
  -1, -1, 6, 7, 8, 9, 10, 11,
  -1, -1, 12, 13, 14, 15, 16, 17,
  -1, -1, 18, 19, 20, 21, -1, -1
};
static const int final[32] =
{
  -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  -1, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, -1, -1
};

/*
   Hangul Jamo in Johab to Unicode 2.0 : Unicode 2.0
   defines 51 Hangul Compatibility Jamos in the block [0x3131,0x314e]

   It's to be considered later which Jamo block to use, Compatibility
   block [0x3131,0x314e] or Hangul Conjoining Jamo block, [0x1100,0x11ff]

 */
static const uint32_t init_to_ucs[19] =
{
  0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139, 0x3141, 0x3142,
  0x3143, 0x3145, 0x3146, 0x3147, 0x3148, 0x3149, 0x314a, 0x314b,
  0x314c, 0x314d, 0x314e
};

static const uint32_t final_to_ucs[31] =
{
  L'\0', L'\0', 0x3133, L'\0', 0x3135, 0x3136, L'\0', L'\0',
  0x313a, 0x313b, 0x313c, 0x313d, 0x313e, 0x313f,
  0x3140, L'\0', L'\0', 0x3144, L'\0', L'\0', L'\0', L'\0',
  L'\0', L'\0', L'\0', L'\0', L'\0', L'\0', L'\0', L'\0', L'\0'
};

/* The following three arrays are used to convert
   precomposed Hangul syllables in [0xac00,0xd???]
   to Jamo bit patterns for Johab encoding

   cf. : KS C 5601-1992, Annex3 Table 2

   Arrays are used to speed up things although it's possible
   to get the same result arithmetically.

 */
static const int init_to_bit[19] =
{
  0x8800, 0x8c00, 0x9000, 0x9400, 0x9800, 0x9c00,
  0xa000, 0xa400, 0xa800, 0xac00, 0xb000, 0xb400,
  0xb800, 0xbc00, 0xc000, 0xc400, 0xc800, 0xcc00,
  0xd000
};

static const int mid_to_bit[21] =
{
	  0x0060, 0x0080, 0x00a0, 0x00c0, 0x00e0,
  0x0140, 0x0160, 0x0180, 0x01a0, 0x01c0, 0x1e0,
  0x0240, 0x0260, 0x0280, 0x02a0, 0x02c0, 0x02e0,
  0x0340, 0x0360, 0x0380, 0x03a0
};

static const int final_to_bit[28] =
{
  1, 2, 3, 4, 5, 6, 7, 8, 9, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11,
  0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d
};

/* The conversion table from
   UCS4 Hangul Compatibility Jamo in [0x3131,0x3163]
   to Johab

   cf. 1. KS C 5601-1992 Annex 3 Table 2
   2. Unicode 2.0 manual

 */
static const uint16_t jamo_from_ucs_table[51] =
{
  0x8841, 0x8c41,
  0x8444,
  0x9041,
  0x8446, 0x8447,
  0x9441, 0x9841, 0x9c41,
  0x844a, 0x844b, 0x844c, 0x844d, 0x844e, 0x844f, 0x8450,
  0xa041, 0xa441, 0xa841,
  0x8454,
  0xac41, 0xb041, 0xb441, 0xb841, 0xbc41,
  0xc041, 0xc441, 0xc841, 0xcc41, 0xd041,
  0x8461, 0x8481, 0x84a1, 0x84c1, 0x84e1,
  0x8541, 0x8561, 0x8581, 0x85a1, 0x85c1, 0x85e1,
  0x8641, 0x8661, 0x8681, 0x86a1, 0x86c1, 0x86e1,
  0x8741, 0x8761, 0x8781, 0x87a1
};


static uint32_t
johab_sym_hanja_to_ucs (uint_fast32_t idx, uint_fast32_t c1, uint_fast32_t c2)
{
  if (idx <= 0xdefe)
    return (uint32_t) __ksc5601_sym_to_ucs[(c1 - 0xd9) * 188 + c2
					   - (c2 > 0x90 ? 0x43 : 0x31)];
  else
    return (uint32_t) __ksc5601_hanja_to_ucs[(c1 - 0xe0) * 188 + c2
					     - (c2 > 0x90 ? 0x43 : 0x31)];
}
/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"JOHAB//"
#define FROM_LOOP		from_johab
#define TO_LOOP			to_johab
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define MIN_NEEDED_FROM		1
#define MAX_NEEDED_FROM		2
#define MIN_NEEDED_TO		4
#define ONE_DIRECTION		0


/* First define the conversion function from JOHAB to UCS4.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
									      \
    if (ch <= 0x7f)							      \
      {									      \
	/* Plain ISO646-KR.  */						      \
	if (ch == 0x5c)							      \
	  ch = 0x20a9; /* half-width Korean Currency WON sign */	      \
	++inptr;							      \
      }									      \
    /* Johab : 1. Hangul						      \
       1st byte : 0x84-0xd3						      \
       2nd byte : 0x41-0x7e, 0x81-0xfe					      \
       2. Hanja & Symbol  :						      \
       1st byte : 0xd8-0xde, 0xe0-0xf9					      \
       2nd byte : 0x31-0x7e, 0x91-0xfe					      \
       0xd831-0xd87e and 0xd891-0xd8fe are user-defined area */		      \
    else								      \
      {									      \
	if (__builtin_expect (ch > 0xf9, 0)				      \
	    || __builtin_expect (ch == 0xdf, 0)				      \
	    || (__builtin_expect (ch > 0x7e, 0) && ch < 0x84)		      \
	    || (__builtin_expect (ch > 0xd3, 0) && ch < 0xd9))		      \
	  {								      \
	    /* These are illegal.  */					      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
	else								      \
	  {								      \
	    /* Two-byte character.  First test whether the next		      \
	       character is also available.  */				      \
	    uint32_t ch2;						      \
	    uint_fast32_t idx;						      \
									      \
	    if (__glibc_unlikely (inptr + 1 >= inend))			      \
	      {								      \
		/* The second character is not available.  Store the	      \
		   intermediate result.  */				      \
		result = __GCONV_INCOMPLETE_INPUT;			      \
		break;							      \
	      }								      \
									      \
	    ch2 = inptr[1];						      \
	    idx = ch * 256 + ch2;					      \
	    if (__glibc_likely (ch <= 0xd3))				      \
	      {								      \
		/* Hangul */						      \
		int_fast32_t i, m, f;					      \
									      \
		i = init[(idx & 0x7c00) >> 10];				      \
		m = mid[(idx & 0x03e0) >> 5];				      \
		f = final[idx & 0x001f];				      \
									      \
		if (__builtin_expect (i == -1, 0)			      \
		    || __builtin_expect (m == -1, 0)			      \
		    || __builtin_expect (f == -1, 0))			      \
		  {							      \
		    /* This is illegal.  */				      \
		    STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
		  }							      \
		else if (i > 0 && m > 0)				      \
		  ch = ((i - 1) * 21 + (m - 1)) * 28 + f + 0xac00;	      \
		else if (i > 0 && m == 0 && f == 0)			      \
		  ch = init_to_ucs[i - 1];				      \
		else if (i == 0 && m > 0 && f == 0)			      \
		  ch = 0x314e + m;	/* 0x314f + m - 1 */		      \
		else if (__builtin_expect ((i | m) == 0, 1)		      \
			 && __builtin_expect (f > 0, 1))		      \
		  ch = final_to_ucs[f - 1];	/* round trip?? */	      \
		else							      \
		  {							      \
		    /* This is illegal.  */				      \
		    STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
		  }							      \
	      }								      \
	    else							      \
	      {								      \
		if (__builtin_expect (ch2 < 0x31, 0)			      \
		    || (__builtin_expect (ch2 > 0x7e, 0) && ch2 < 0x91)	      \
		    || __builtin_expect (ch2, 0) == 0xff		      \
		    || (__builtin_expect (ch, 0) == 0xd9 && ch2 > 0xe8)	      \
		    || (__builtin_expect (ch, 0) == 0xda		      \
			&& ch2 > 0xa0 && ch2 < 0xd4)			      \
		    || (__builtin_expect (ch, 0) == 0xde && ch2 > 0xf1))      \
		  {							      \
		    /* This is illegal.  */				      \
		    STANDARD_FROM_LOOP_ERR_HANDLER (1);			      \
		  }							      \
		else							      \
		  {							      \
		    ch = johab_sym_hanja_to_ucs (idx, ch, ch2);		      \
		    /* if (idx <= 0xdefe)				      \
			 ch = __ksc5601_sym_to_ucs[(ch - 0xd9) * 192	      \
						   + ch2 - (ch2 > 0x90	      \
							    ? 0x43 : 0x31)];  \
		       else						      \
			 ch = __ksc5601_hanja_to_ucs[(ch - 0xe0) *192	      \
						     + ch2 -  (ch2 > 0x90     \
							       ?0x43 : 0x31)];\
		    */							      \
		  }							      \
	      }								      \
	  }								      \
									      \
	if (__glibc_unlikely (ch == 0))					      \
	  {								      \
	    /* This is an illegal character.  */			      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (2);				      \
	  }								      \
									      \
	inptr += 2;							      \
      }									      \
									      \
    put32 (outptr, ch);							      \
    outptr += 4;							      \
  }
#define LOOP_NEED_FLAGS
#define ONEBYTE_BODY \
  {									      \
    if (c <= 0x7f)							      \
      return (c == 0x5c ? 0x20a9 : c);					      \
    else								      \
      return WEOF;							      \
  }
#include <iconv/loop.c>


/* Next, define the other direction.  */
#define MIN_NEEDED_INPUT	MIN_NEEDED_TO
#define MIN_NEEDED_OUTPUT	MIN_NEEDED_FROM
#define MAX_NEEDED_OUTPUT	MAX_NEEDED_FROM
#define LOOPFCT			TO_LOOP
#define BODY \
  {									      \
    uint32_t ch = get32 (inptr);					      \
    /*									      \
       if (ch >= (sizeof (from_ucs4_lat1) / sizeof (from_ucs4_lat1[0])))      \
	 {								      \
	   if (ch >= 0x0391 && ch <= 0x0451)				      \
	     cp = from_ucs4_greek[ch - 0x391];				      \
	   else if (ch >= 0x2010 && ch <= 0x9fa0)			      \
	     cp = from_ucs4_cjk[ch - 0x02010];				      \
	   else								      \
	     break;							      \
	 }								      \
       else								      \
	 cp = from_ucs4_lat1[ch];					      \
    */									      \
									      \
    if (ch <= 0x7f && ch != 0x5c)					      \
      *outptr++ = ch;							      \
    else								      \
      {									      \
	if (ch >= 0xac00 && ch <= 0xd7a3)				      \
	  {								      \
	    if (__glibc_unlikely (outptr + 2 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    ch -= 0xac00;						      \
									      \
	    ch = (init_to_bit[ch / 588]	  /* 21 * 28 = 588 */		      \
		  + mid_to_bit[(ch / 28) % 21]/* (ch % (21 * 28)) / 28 */     \
		  + final_to_bit[ch %  28]);  /* (ch % (21 * 28)) % 28 */     \
									      \
	    *outptr++ = ch / 256;					      \
	    *outptr++ = ch % 256;					      \
	  }								      \
	/* KS C 5601-1992 Annex 3 regards  0xA4DA(Hangul Filler : U3164)      \
	   as symbol */							      \
	else if (ch >= 0x3131 && ch <= 0x3163)				      \
	  {								      \
	    ch = jamo_from_ucs_table[ch - 0x3131];			      \
									      \
	    if (__glibc_unlikely (outptr + 2 > outend))			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
									      \
	    *outptr++ = ch / 256;					      \
	    *outptr++ = ch % 256;					      \
	  }								      \
	else if ((ch >= 0x4e00 && ch <= 0x9fa5)				      \
		 || (ch >= 0xf900 && ch <= 0xfa0b))			      \
	  {								      \
	    size_t written;						      \
	    uint32_t temp;						      \
									      \
	    written = ucs4_to_ksc5601_hanja (ch, outptr, outend - outptr);    \
	    if (__builtin_expect (written, 1) == 0)			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    if (__glibc_unlikely (written == __UNKNOWN_10646_CHAR))	      \
	      {								      \
		STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
	      }								      \
									      \
	    outptr[0] -= 0x4a;						      \
	    outptr[1] -= 0x21;						      \
									      \
	    temp = outptr[0] * 94 + outptr[1];				      \
									      \
	    outptr[0] = 0xe0 + temp / 188;				      \
	    outptr[1] = temp % 188;					      \
	    outptr[1] += outptr[1] >= 78 ? 0x43 : 0x31;			      \
									      \
	    outptr += 2;						      \
	  }								      \
	else if (ch == 0x20a9)						      \
	  *outptr++ = 0x5c;						      \
	else								      \
	  {								      \
	    size_t written;						      \
	    uint32_t temp;						      \
									      \
	    written = ucs4_to_ksc5601_sym (ch, outptr, outend - outptr);      \
	    if (__builtin_expect (written, 1) == 0)			      \
	      {								      \
		result = __GCONV_FULL_OUTPUT;				      \
		break;							      \
	      }								      \
	    if (__builtin_expect (written == __UNKNOWN_10646_CHAR, 0)	      \
		|| (outptr[0] == 0x22 && outptr[1] > 0x68))		      \
	      {								      \
		UNICODE_TAG_HANDLER (ch, 4);				      \
		STANDARD_TO_LOOP_ERR_HANDLER (4);			      \
	      }								      \
									      \
	    temp = (outptr[0] < 0x4a ? outptr[0] + 0x191 : outptr[0] + 0x176);\
	    outptr[1] += (temp % 2 ? 0x5e : 0);				      \
	    outptr[1] += (outptr[1] < 0x6f ? 0x10 : 0x22);		      \
	    outptr[0] = temp / 2;					      \
									      \
	    outptr += 2;						      \
	  }								      \
      }									      \
									      \
    inptr += 4;								      \
  }
#define LOOP_NEED_FLAGS
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
