/* Conversion from and to CP1255.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998,
   and Bruno Haible <haible@clisp.cons.org>, 2001.

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
#include <assert.h>

#define NELEMS(arr) (sizeof (arr) / sizeof (arr[0]))

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"CP1255//"
#define FROM_LOOP		from_cp1255
#define TO_LOOP			to_cp1255
#define DEFINE_INIT		1
#define DEFINE_FINI		1
#define ONE_DIRECTION		0
#define FROM_LOOP_MIN_NEEDED_FROM	1
#define FROM_LOOP_MAX_NEEDED_FROM	1
#define FROM_LOOP_MIN_NEEDED_TO		4
#define FROM_LOOP_MAX_NEEDED_TO		4
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


/* During CP1255 to UCS4 conversion, the COUNT element of the state
   contains the last UCS4 character, shifted by 3 bits.  */


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
	/* We don't use shift states in the TO_DIRECTION.  */		      \
	data->__statep->__count = 0;					      \
    }


/* First define the conversion function from CP1255 to UCS4.  */

static const uint16_t to_ucs4[128] = {
  /* 0x80 */
  0x20AC,      0, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
  0x02C6, 0x2030,      0, 0x2039,      0,      0,      0,      0,
  /* 0x90 */
       0, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
  0x02DC, 0x2122,      0, 0x203A,      0,      0,      0,      0,
  /* 0xA0 */
  0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x20AA, 0x00A5, 0x00A6, 0x00A7,
  0x00A8, 0x00A9, 0x00D7, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
  /* 0xB0 */
  0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7,
  0x00B8, 0x00B9, 0x00F7, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF,
  /* 0xC0 */
  0x05B0, 0x05B1, 0x05B2, 0x05B3, 0x05B4, 0x05B5, 0x05B6, 0x05B7,
  0x05B8, 0x05B9,      0, 0x05BB, 0x05BC, 0x05BD, 0x05BE, 0x05BF,
  /* 0xD0 */
  0x05C0, 0x05C1, 0x05C2, 0x05C3, 0x05F0, 0x05F1, 0x05F2, 0x05F3,
  0x05F4,      0,      0,      0,      0,      0,      0,      0,
  /* 0xE0 */
  0x05D0, 0x05D1, 0x05D2, 0x05D3, 0x05D4, 0x05D5, 0x05D6, 0x05D7,
  0x05D8, 0x05D9, 0x05DA, 0x05DB, 0x05DC, 0x05DD, 0x05DE, 0x05DF,
  /* 0xF0 */
  0x05E0, 0x05E1, 0x05E2, 0x05E3, 0x05E4, 0x05E5, 0x05E6, 0x05E7,
  0x05E8, 0x05E9, 0x05EA,      0,      0, 0x200E, 0x200F,      0,
};

/* CP1255 contains eight combining characters:
   0x05b4, 0x05b7, 0x05b8, 0x05b9, 0x05bc, 0x05bf, 0x05c1, 0x05c2.  */

/* Composition tables for each of the relevant combining characters.  */
static const struct {
  uint16_t base;
  uint16_t composed;
} comp_table_data[] = {
#define COMP_TABLE_IDX_05B4 0
#define COMP_TABLE_LEN_05B4 1
  { 0x05D9, 0xFB1D },
#define COMP_TABLE_IDX_05B7 (COMP_TABLE_IDX_05B4 + COMP_TABLE_LEN_05B4)
#define COMP_TABLE_LEN_05B7 2
  { 0x05D0, 0xFB2E },
  { 0x05F2, 0xFB1F },
#define COMP_TABLE_IDX_05B8 (COMP_TABLE_IDX_05B7 + COMP_TABLE_LEN_05B7)
#define COMP_TABLE_LEN_05B8 1
  { 0x05D0, 0xFB2F },
#define COMP_TABLE_IDX_05B9 (COMP_TABLE_IDX_05B8 + COMP_TABLE_LEN_05B8)
#define COMP_TABLE_LEN_05B9 1
  { 0x05D5, 0xFB4B },
#define COMP_TABLE_IDX_05BC (COMP_TABLE_IDX_05B9 + COMP_TABLE_LEN_05B9)
#define COMP_TABLE_LEN_05BC 24
  { 0x05D0, 0xFB30 },
  { 0x05D1, 0xFB31 },
  { 0x05D2, 0xFB32 },
  { 0x05D3, 0xFB33 },
  { 0x05D4, 0xFB34 },
  { 0x05D5, 0xFB35 },
  { 0x05D6, 0xFB36 },
  { 0x05D8, 0xFB38 },
  { 0x05D9, 0xFB39 },
  { 0x05DA, 0xFB3A },
  { 0x05DB, 0xFB3B },
  { 0x05DC, 0xFB3C },
  { 0x05DE, 0xFB3E },
  { 0x05E0, 0xFB40 },
  { 0x05E1, 0xFB41 },
  { 0x05E3, 0xFB43 },
  { 0x05E4, 0xFB44 },
  { 0x05E6, 0xFB46 },
  { 0x05E7, 0xFB47 },
  { 0x05E8, 0xFB48 },
  { 0x05E9, 0xFB49 },
  { 0x05EA, 0xFB4A },
  { 0xFB2A, 0xFB2C },
  { 0xFB2B, 0xFB2D },
#define COMP_TABLE_IDX_05BF (COMP_TABLE_IDX_05BC + COMP_TABLE_LEN_05BC)
#define COMP_TABLE_LEN_05BF 3
  { 0x05D1, 0xFB4C },
  { 0x05DB, 0xFB4D },
  { 0x05E4, 0xFB4E },
#define COMP_TABLE_IDX_05C1 (COMP_TABLE_IDX_05BF + COMP_TABLE_LEN_05BF)
#define COMP_TABLE_LEN_05C1 2
  { 0x05E9, 0xFB2A },
  { 0xFB49, 0xFB2C },
#define COMP_TABLE_IDX_05C2 (COMP_TABLE_IDX_05C1 + COMP_TABLE_LEN_05C1)
#define COMP_TABLE_LEN_05C2 2
  { 0x05E9, 0xFB2B },
  { 0xFB49, 0xFB2D },
#define COMP_TABLE_IDX_END (COMP_TABLE_IDX_05C2 + COMP_TABLE_LEN_05C2)
};
/* Compile-time verification of table size.  */
typedef int verify1[(NELEMS (comp_table_data) == COMP_TABLE_IDX_END) - 1];

static const struct { unsigned int idx; unsigned int len; } comp_table[8] = {
  { COMP_TABLE_IDX_05B4, COMP_TABLE_LEN_05B4 },
  { COMP_TABLE_IDX_05B7, COMP_TABLE_LEN_05B7 },
  { COMP_TABLE_IDX_05B8, COMP_TABLE_LEN_05B8 },
  { COMP_TABLE_IDX_05B9, COMP_TABLE_LEN_05B9 },
  { COMP_TABLE_IDX_05BC, COMP_TABLE_LEN_05BC },
  { COMP_TABLE_IDX_05BF, COMP_TABLE_LEN_05BF },
  { COMP_TABLE_IDX_05C1, COMP_TABLE_LEN_05C1 },
  { COMP_TABLE_IDX_05C2, COMP_TABLE_LEN_05C2 },
};

#define MIN_NEEDED_INPUT	FROM_LOOP_MIN_NEEDED_FROM
#define MAX_NEEDED_INPUT	FROM_LOOP_MAX_NEEDED_FROM
#define MIN_NEEDED_OUTPUT	FROM_LOOP_MIN_NEEDED_TO
#define MAX_NEEDED_OUTPUT	FROM_LOOP_MAX_NEEDED_TO
#define LOOPFCT			FROM_LOOP
#define BODY \
  {									      \
    uint32_t ch = *inptr;						      \
    uint32_t last_ch;							      \
    int must_buffer_ch;							      \
									      \
    if (ch >= 0x80)							      \
      {									      \
	ch = to_ucs4[ch - 0x80];					      \
	if (__glibc_unlikely (ch == L'\0'))				      \
	  {								      \
	    /* This is an illegal character.  */			      \
	    STANDARD_FROM_LOOP_ERR_HANDLER (1);				      \
	  }								      \
      }									      \
									      \
    /* Determine whether there is a buffered character pending.  */	      \
    last_ch = *statep >> 3;						      \
									      \
    /* We have to buffer ch if it is a possible match in comp_table_data.  */ \
    must_buffer_ch = (ch >= 0x05d0 && ch <= 0x05f2);			      \
									      \
    if (last_ch)							      \
      {									      \
	if (ch >= 0x05b0 && ch < 0x05c5)				      \
	  {								      \
	    /* See whether last_ch and ch can be combined.  */		      \
	    unsigned int i, i1, i2;					      \
									      \
	    switch (ch)							      \
	      {								      \
	      case 0x05b4:						      \
		i = 0;							      \
		break;							      \
	      case 0x05b7:						      \
		i = 1;							      \
		break;							      \
	      case 0x05b8:						      \
		i = 2;							      \
		break;							      \
	      case 0x05b9:						      \
		i = 3;							      \
		break;							      \
	      case 0x05bc:						      \
		i = 4;							      \
		break;							      \
	      case 0x05bf:						      \
		i = 5;							      \
		break;							      \
	      case 0x05c1:						      \
		i = 6;							      \
		break;							      \
	      case 0x05c2:						      \
		i = 7;							      \
		break;							      \
	      default:							      \
		goto not_combining;					      \
	      }								      \
									      \
	    i1 = comp_table[i].idx;					      \
	    i2 = i1 + comp_table[i].len - 1;				      \
									      \
	    if (last_ch >= comp_table_data[i1].base			      \
		&& last_ch <= comp_table_data[i2].base)			      \
	      {								      \
		for (;;)						      \
		  {							      \
		    i = (i1 + i2) >> 1;					      \
		    if (last_ch == comp_table_data[i].base)		      \
		      break;						      \
		    if (last_ch < comp_table_data[i].base)		      \
		      {							      \
			if (i1 == i)					      \
			  goto not_combining;				      \
			i2 = i;						      \
		      }							      \
		    else						      \
		      {							      \
			if (i1 != i)					      \
			  i1 = i;					      \
			else						      \
			  {						      \
			    i = i2;					      \
			    if (last_ch == comp_table_data[i].base)	      \
			      break;					      \
			    goto not_combining;				      \
			  }						      \
		      }							      \
		  }							      \
		last_ch = comp_table_data[i].composed;			      \
		if (last_ch == 0xfb2a || last_ch == 0xfb2b		      \
		    || last_ch == 0xfb49)				      \
		  /* Buffer the combined character.  */			      \
		  *statep = last_ch << 3;				      \
		else							      \
		  {							      \
		    /* Output the combined character.  */		      \
		    put32 (outptr, last_ch);				      \
		    outptr += 4;					      \
		    *statep = 0;					      \
		  }							      \
		++inptr;						      \
		continue;						      \
	      }								      \
	  }								      \
									      \
      not_combining:							      \
	/* Output the buffered character.  */				      \
	put32 (outptr, last_ch);					      \
	outptr += 4;							      \
	*statep = 0;							      \
									      \
	/* If we don't have enough room to output ch as well, then deal	      \
	   with it in another round.  */				      \
	if (!must_buffer_ch && __builtin_expect (outptr + 4 > outend, 0))     \
	  continue;							      \
      }									      \
									      \
    if (must_buffer_ch)							      \
      *statep = ch << 3;						      \
    else								      \
      {									      \
	put32 (outptr, ch);						      \
	outptr += 4;							      \
      }									      \
    ++inptr;								      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#define ONEBYTE_BODY \
  {									      \
    if (c < 0x80)							      \
      return c;								      \
    uint32_t ch = to_ucs4[c - 0x80];					      \
    if (ch == L'\0' || (ch >= 0x05d0 && ch <= 0x05f2))			      \
      return WEOF;							      \
    return ch;								      \
  }
#include <iconv/loop.c>


/* Next, define the conversion function from UCS4 to CP1255.  */

static const unsigned char from_ucs4[] = {
#define FROM_IDX_00 0
  0xa0, 0xa1, 0xa2, 0xa3, 0x00, 0xa5, 0xa6, 0xa7, /* 0x00a0-0x00a7 */
  0xa8, 0xa9, 0x00, 0xab, 0xac, 0xad, 0xae, 0xaf, /* 0x00a8-0x00af */
  0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, /* 0x00b0-0x00b7 */
  0xb8, 0xb9, 0x00, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, /* 0x00b8-0x00bf */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x00c0-0x00c7 */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x00c8-0x00cf */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xaa, /* 0x00d0-0x00d7 */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x00d8-0x00df */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x00e0-0x00e7 */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x00e8-0x00ef */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xba, /* 0x00f0-0x00f7 */
#define FROM_IDX_02 (FROM_IDX_00 + 88)
                                      0x88, 0x00, /* 0x02c6-0x02c7 */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x02c8-0x02cf */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x02d0-0x02d7 */
  0x00, 0x00, 0x00, 0x00, 0x98,                   /* 0x02d8-0x02dc */
#define FROM_IDX_05 (FROM_IDX_02 + 23)
  0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, /* 0x05b0-0x05b7 */
  0xc8, 0xc9, 0x00, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, /* 0x05b8-0x05bf */
  0xd0, 0xd1, 0xd2, 0xd3, 0x00, 0x00, 0x00, 0x00, /* 0x05c0-0x05c7 */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x05c8-0x05cf */
  0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, /* 0x05d0-0x05d7 */
  0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef, /* 0x05d8-0x05df */
  0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, /* 0x05e0-0x05e7 */
  0xf8, 0xf9, 0xfa, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x05e8-0x05ef */
  0xd4, 0xd5, 0xd6, 0xd7, 0xd8,                   /* 0x05f0-0x05f4 */
#define FROM_IDX_20 (FROM_IDX_05 + 69)
                                      0xfd, 0xfe, /* 0x200e-0x200f */
  0x00, 0x00, 0x00, 0x96, 0x97, 0x00, 0x00, 0x00, /* 0x2010-0x2017 */
  0x91, 0x92, 0x82, 0x00, 0x93, 0x94, 0x84, 0x00, /* 0x2018-0x201f */
  0x86, 0x87, 0x95, 0x00, 0x00, 0x00, 0x85, 0x00, /* 0x2020-0x2027 */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x2028-0x202f */
  0x89, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x2030-0x2037 */
  0x00, 0x8b, 0x9b,                               /* 0x2038-0x203a */
#define FROM_IDX_FF (FROM_IDX_20 + 45)
};
/* Compile-time verification of table size.  */
typedef int verify2[(NELEMS (from_ucs4) == FROM_IDX_FF) - 1];

static const unsigned char comb_table[8] = {
  0xc4, 0xc7, 0xc8, 0xc9, 0xcc, 0xcf, 0xd1, 0xd2,
};

/* Decomposition table for the relevant Unicode characters. */
static const struct {
  uint16_t composed;
  uint16_t base;
  uint32_t comb1 : 8;
  int32_t comb2 : 8;
} decomp_table[] = {
  { 0xFB1D, 0x05D9, 0, -1 },
  { 0xFB1F, 0x05F2, 1, -1 },
  { 0xFB2A, 0x05E9, 6, -1 },
  { 0xFB2B, 0x05E9, 7, -1 },
  { 0xFB2C, 0x05E9, 4, 6 },
  { 0xFB2D, 0x05E9, 4, 7 },
  { 0xFB2E, 0x05D0, 1, -1 },
  { 0xFB2F, 0x05D0, 2, -1 },
  { 0xFB30, 0x05D0, 4, -1 },
  { 0xFB31, 0x05D1, 4, -1 },
  { 0xFB32, 0x05D2, 4, -1 },
  { 0xFB33, 0x05D3, 4, -1 },
  { 0xFB34, 0x05D4, 4, -1 },
  { 0xFB35, 0x05D5, 4, -1 },
  { 0xFB36, 0x05D6, 4, -1 },
  { 0xFB38, 0x05D8, 4, -1 },
  { 0xFB39, 0x05D9, 4, -1 },
  { 0xFB3A, 0x05DA, 4, -1 },
  { 0xFB3B, 0x05DB, 4, -1 },
  { 0xFB3C, 0x05DC, 4, -1 },
  { 0xFB3E, 0x05DE, 4, -1 },
  { 0xFB40, 0x05E0, 4, -1 },
  { 0xFB41, 0x05E1, 4, -1 },
  { 0xFB43, 0x05E3, 4, -1 },
  { 0xFB44, 0x05E4, 4, -1 },
  { 0xFB46, 0x05E6, 4, -1 },
  { 0xFB47, 0x05E7, 4, -1 },
  { 0xFB48, 0x05E8, 4, -1 },
  { 0xFB49, 0x05E9, 4, -1 },
  { 0xFB4A, 0x05EA, 4, -1 },
  { 0xFB4B, 0x05D5, 3, -1 },
  { 0xFB4C, 0x05D1, 5, -1 },
  { 0xFB4D, 0x05DB, 5, -1 },
  { 0xFB4E, 0x05E4, 5, -1 },
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
    if (ch < 0x0080)							      \
      {									      \
	*outptr++ = ch;							      \
	inptr += 4;							      \
      }									      \
    else								      \
      {									      \
	unsigned char res;						      \
									      \
	if (ch >= 0x00a0 && ch < 0x00f8)				      \
	  res = from_ucs4[ch - 0x00a0 + FROM_IDX_00];			      \
	else if (ch == 0x0192)						      \
	  res = 0x83;							      \
	else if (ch >= 0x02c6 && ch < 0x02dd)				      \
	  res = from_ucs4[ch - 0x02c6 + FROM_IDX_02];			      \
	else if (ch >= 0x05b0 && ch < 0x05f5)				      \
	  res = from_ucs4[ch - 0x05b0 + FROM_IDX_05];			      \
	else if (ch >= 0x200e && ch < 0x203b)				      \
	  res = from_ucs4[ch - 0x200e + FROM_IDX_20];			      \
	else if (ch == 0x20aa)						      \
	  res = 0xa4;							      \
	else if (ch == 0x20ac)						      \
	  res = 0x80;							      \
	else if (ch == 0x2122)						      \
	  res = 0x99;							      \
	else								      \
	  {								      \
	    UNICODE_TAG_HANDLER (ch, 4);				      \
	    res = 0;							      \
	  }								      \
									      \
	if (__glibc_likely (res != 0))					      \
	  {								      \
	    *outptr++ = res;						      \
	    inptr += 4;							      \
	  }								      \
	else								      \
	  {								      \
	    /* Try canonical decomposition.  */				      \
	    unsigned int i1, i2;					      \
									      \
	    i1 = 0;							      \
	    i2 = sizeof (decomp_table) / sizeof (decomp_table[0]) - 1;	      \
	    if (ch >= decomp_table[i1].composed				      \
		&& ch <= decomp_table[i2].composed)			      \
	      {								      \
		unsigned int i;						      \
									      \
		for (;;)						      \
		  {							      \
		    i = (i1 + i2) >> 1;					      \
		    if (ch == decomp_table[i].composed)			      \
		      break;						      \
		    if (ch < decomp_table[i].composed)			      \
		      {							      \
			if (i1 == i)					      \
			  goto failed;					      \
			i2 = i;						      \
		      }							      \
		    else						      \
		      {							      \
			if (i1 != i)					      \
			  i1 = i;					      \
			else						      \
			  {						      \
			    i = i2;					      \
			    if (ch == decomp_table[i].composed)		      \
			      break;					      \
			    goto failed;				      \
			  }						      \
		      }							      \
		  }							      \
									      \
		/* Found a canonical decomposition.  */			      \
		ch = decomp_table[i].base;				      \
		/* ch is one of 0x05d0..0x05d6, 0x05d8..0x05dc, 0x05de,	      \
		   0x05e0..0x05e1, 0x05e3..0x05e4, 0x05e6..0x05ea, 0x05f2. */ \
		ch = from_ucs4[ch - 0x05b0 + FROM_IDX_05];		      \
		assert (ch != 0);					      \
									      \
		if (decomp_table[i].comb2 < 0)				      \
		  {							      \
		    /* See whether we have room for two bytes.  */	      \
		    if (__glibc_unlikely (outptr + 1 >= outend))	      \
		      {							      \
			result = __GCONV_FULL_OUTPUT;			      \
			break;						      \
		      }							      \
									      \
		    *outptr++ = (unsigned char) ch;			      \
		    *outptr++ = comb_table[decomp_table[i].comb1];	      \
		  }							      \
		else							      \
		  {							      \
		    /* See whether we have room for three bytes.  */	      \
		    if (__glibc_unlikely (outptr + 2 >= outend))	      \
		      {							      \
			result = __GCONV_FULL_OUTPUT;			      \
			break;						      \
		      }							      \
									      \
		    *outptr++ = (unsigned char) ch;			      \
		    *outptr++ = comb_table[decomp_table[i].comb1];	      \
		    *outptr++ = comb_table[decomp_table[i].comb2];	      \
		  }							      \
									      \
		inptr += 4;						      \
		continue;						      \
	      }								      \
									      \
	  failed:							      \
	    /* This is an illegal character.  */			      \
	    STANDARD_TO_LOOP_ERR_HANDLER (4);				      \
	  }								      \
      }									      \
  }
#define LOOP_NEED_FLAGS
#define EXTRA_LOOP_DECLS	, int *statep
#include <iconv/loop.c>


/* Now define the toplevel functions.  */
#include <iconv/skeleton.c>
