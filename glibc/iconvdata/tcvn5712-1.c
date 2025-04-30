/* Conversion to and from TCVN5712-1.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2001.

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
#include <stdlib.h>

#define NELEMS(arr) (sizeof (arr) / sizeof (arr[0]))

/* Definitions used in the body of the `gconv' function.  */
#define CHARSET_NAME		"TCVN5712-1//"
#define FROM_LOOP		from_tcvn5712_1
#define TO_LOOP			to_tcvn5712_1
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


/* During TCVN5712-1 to UCS4 conversion, the COUNT element of the state
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


static const uint16_t map_from_tcvn_low[0x18] =
  {
    0x0000, 0x00da, 0x1ee4, 0x0003, 0x1eea, 0x1eec, 0x1eee, 0x0007,
    0x0008, 0x0009, 0x000a, 0x000b, 0x000c, 0x000d, 0x000e, 0x000f,
    0x0010, 0x1ee8, 0x1ef0, 0x1ef2, 0x1ef6, 0x1ef8, 0x00dd, 0x1ef4
  };

static const uint16_t map_from_tcvn_high[0x80] =
  {
    0x00c0, 0x1ea2, 0x00c3, 0x00c1, 0x1ea0, 0x1eb6, 0x1eac, 0x00c8,
    0x1eba, 0x1ebc, 0x00c9, 0x1eb8, 0x1ec6, 0x00cc, 0x1ec8, 0x0128,
    0x00cd, 0x1eca, 0x00d2, 0x1ece, 0x00d5, 0x00d3, 0x1ecc, 0x1ed8,
    0x1edc, 0x1ede, 0x1ee0, 0x1eda, 0x1ee2, 0x00d9, 0x1ee6, 0x0168,
    0x00a0, 0x0102, 0x00c2, 0x00ca, 0x00d4, 0x01a0, 0x01af, 0x0110,
    0x0103, 0x00e2, 0x00ea, 0x00f4, 0x01a1, 0x01b0, 0x0111, 0x1eb0,
    0x0300, 0x0309, 0x0303, 0x0301, 0x0323, 0x00e0, 0x1ea3, 0x00e3,
    0x00e1, 0x1ea1, 0x1eb2, 0x1eb1, 0x1eb3, 0x1eb5, 0x1eaf, 0x1eb4,
    0x1eae, 0x1ea6, 0x1ea8, 0x1eaa, 0x1ea4, 0x1ec0, 0x1eb7, 0x1ea7,
    0x1ea9, 0x1eab, 0x1ea5, 0x1ead, 0x00e8, 0x1ec2, 0x1ebb, 0x1ebd,
    0x00e9, 0x1eb9, 0x1ec1, 0x1ec3, 0x1ec5, 0x1ebf, 0x1ec7, 0x00ec,
    0x1ec9, 0x1ec4, 0x1ebe, 0x1ed2, 0x0129, 0x00ed, 0x1ecb, 0x00f2,
    0x1ed4, 0x1ecf, 0x00f5, 0x00f3, 0x1ecd, 0x1ed3, 0x1ed5, 0x1ed7,
    0x1ed1, 0x1ed9, 0x1edd, 0x1edf, 0x1ee1, 0x1edb, 0x1ee3, 0x00f9,
    0x1ed6, 0x1ee7, 0x0169, 0x00fa, 0x1ee5, 0x1eeb, 0x1eed, 0x1eef,
    0x1ee9, 0x1ef1, 0x1ef3, 0x1ef7, 0x1ef9, 0x00fd, 0x1ef5, 0x1ed0
  };


/* TCVN5712-1 contains five combining characters:
   0x0300, 0x0301, 0x0303, 0x0309, 0x0323.  */

/* Composition tables for each of the relevant combining characters.  */
static const struct
{
  uint16_t base;
  uint16_t composed;
} comp_table_data[] =
  {
#define COMP_TABLE_IDX_0300 0
#define COMP_TABLE_LEN_0300 28
    { 0x0041, 0x00C0 },
    { 0x0045, 0x00C8 },
    { 0x0049, 0x00CC },
    { 0x004E, 0x01F8 },
    { 0x004F, 0x00D2 },
    { 0x0055, 0x00D9 },
    { 0x0057, 0x1E80 },
    { 0x0059, 0x1EF2 },
    { 0x0061, 0x00E0 },
    { 0x0065, 0x00E8 },
    { 0x0069, 0x00EC },
    { 0x006E, 0x01F9 },
    { 0x006F, 0x00F2 },
    { 0x0075, 0x00F9 },
    { 0x0077, 0x1E81 },
    { 0x0079, 0x1EF3 },
  /*{ 0x00A8, 0x1FED },*/
    { 0x00C2, 0x1EA6 },
    { 0x00CA, 0x1EC0 },
    { 0x00D4, 0x1ED2 },
  /*{ 0x00DC, 0x01DB },*/
    { 0x00E2, 0x1EA7 },
    { 0x00EA, 0x1EC1 },
    { 0x00F4, 0x1ED3 },
  /*{ 0x00FC, 0x01DC },*/
    { 0x0102, 0x1EB0 },
    { 0x0103, 0x1EB1 },
  /*{ 0x0112, 0x1E14 },*/
  /*{ 0x0113, 0x1E15 },*/
  /*{ 0x014C, 0x1E50 },*/
  /*{ 0x014D, 0x1E51 },*/
    { 0x01A0, 0x1EDC },
    { 0x01A1, 0x1EDD },
    { 0x01AF, 0x1EEA },
    { 0x01B0, 0x1EEB },
#define COMP_TABLE_IDX_0301 (COMP_TABLE_IDX_0300 + COMP_TABLE_LEN_0300)
#define COMP_TABLE_LEN_0301 50
    { 0x0041, 0x00C1 },
    { 0x0043, 0x0106 },
    { 0x0045, 0x00C9 },
    { 0x0047, 0x01F4 },
    { 0x0049, 0x00CD },
    { 0x004B, 0x1E30 },
    { 0x004C, 0x0139 },
    { 0x004D, 0x1E3E },
    { 0x004E, 0x0143 },
    { 0x004F, 0x00D3 },
    { 0x0050, 0x1E54 },
    { 0x0052, 0x0154 },
    { 0x0053, 0x015A },
    { 0x0055, 0x00DA },
    { 0x0057, 0x1E82 },
    { 0x0059, 0x00DD },
    { 0x005A, 0x0179 },
    { 0x0061, 0x00E1 },
    { 0x0063, 0x0107 },
    { 0x0065, 0x00E9 },
    { 0x0067, 0x01F5 },
    { 0x0069, 0x00ED },
    { 0x006B, 0x1E31 },
    { 0x006C, 0x013A },
    { 0x006D, 0x1E3F },
    { 0x006E, 0x0144 },
    { 0x006F, 0x00F3 },
    { 0x0070, 0x1E55 },
    { 0x0072, 0x0155 },
    { 0x0073, 0x015B },
    { 0x0075, 0x00FA },
    { 0x0077, 0x1E83 },
    { 0x0079, 0x00FD },
    { 0x007A, 0x017A },
  /*{ 0x00A8, 0x0385 },*//* prefer U+0385 over U+1FEE */
    { 0x00C2, 0x1EA4 },
  /*{ 0x00C5, 0x01FA },*/
  /*{ 0x00C6, 0x01FC },*/
  /*{ 0x00C7, 0x1E08 },*/
    { 0x00CA, 0x1EBE },
  /*{ 0x00CF, 0x1E2E },*/
    { 0x00D4, 0x1ED0 },
    { 0x00D5, 0x1E4C },
  /*{ 0x00D8, 0x01FE },*/
  /*{ 0x00DC, 0x01D7 },*/
    { 0x00E2, 0x1EA5 },
  /*{ 0x00E5, 0x01FB },*/
  /*{ 0x00E6, 0x01FD },*/
  /*{ 0x00E7, 0x1E09 },*/
    { 0x00EA, 0x1EBF },
  /*{ 0x00EF, 0x1E2F },*/
    { 0x00F4, 0x1ED1 },
    { 0x00F5, 0x1E4D },
  /*{ 0x00F8, 0x01FF },*/
  /*{ 0x00FC, 0x01D8 },*/
    { 0x0102, 0x1EAE },
    { 0x0103, 0x1EAF },
  /*{ 0x0112, 0x1E16 },*/
  /*{ 0x0113, 0x1E17 },*/
  /*{ 0x014C, 0x1E52 },*/
  /*{ 0x014D, 0x1E53 },*/
    { 0x0168, 0x1E78 },
    { 0x0169, 0x1E79 },
    { 0x01A0, 0x1EDA },
    { 0x01A1, 0x1EDB },
    { 0x01AF, 0x1EE8 },
    { 0x01B0, 0x1EE9 },
#define COMP_TABLE_IDX_0303 (COMP_TABLE_IDX_0301 + COMP_TABLE_LEN_0301)
#define COMP_TABLE_LEN_0303 34
    { 0x0041, 0x00C3 },
    { 0x0045, 0x1EBC },
    { 0x0049, 0x0128 },
    { 0x004E, 0x00D1 },
    { 0x004F, 0x00D5 },
    { 0x0055, 0x0168 },
    { 0x0056, 0x1E7C },
    { 0x0059, 0x1EF8 },
    { 0x0061, 0x00E3 },
    { 0x0065, 0x1EBD },
    { 0x0069, 0x0129 },
    { 0x006E, 0x00F1 },
    { 0x006F, 0x00F5 },
    { 0x0075, 0x0169 },
    { 0x0076, 0x1E7D },
    { 0x0079, 0x1EF9 },
    { 0x00C2, 0x1EAA },
    { 0x00CA, 0x1EC4 },
    { 0x00D3, 0x1E4C },
    { 0x00D4, 0x1ED6 },
    { 0x00D6, 0x1E4E },
    { 0x00DA, 0x1E78 },
    { 0x00E2, 0x1EAB },
    { 0x00EA, 0x1EC5 },
    { 0x00F3, 0x1E4D },
    { 0x00F4, 0x1ED7 },
    { 0x00F6, 0x1E4F },
    { 0x00FA, 0x1E79 },
    { 0x0102, 0x1EB4 },
    { 0x0103, 0x1EB5 },
    { 0x01A0, 0x1EE0 },
    { 0x01A1, 0x1EE1 },
    { 0x01AF, 0x1EEE },
    { 0x01B0, 0x1EEF },
#define COMP_TABLE_IDX_0309 (COMP_TABLE_IDX_0303 + COMP_TABLE_LEN_0303)
#define COMP_TABLE_LEN_0309 24
    { 0x0041, 0x1EA2 },
    { 0x0045, 0x1EBA },
    { 0x0049, 0x1EC8 },
    { 0x004F, 0x1ECE },
    { 0x0055, 0x1EE6 },
    { 0x0059, 0x1EF6 },
    { 0x0061, 0x1EA3 },
    { 0x0065, 0x1EBB },
    { 0x0069, 0x1EC9 },
    { 0x006F, 0x1ECF },
    { 0x0075, 0x1EE7 },
    { 0x0079, 0x1EF7 },
    { 0x00C2, 0x1EA8 },
    { 0x00CA, 0x1EC2 },
    { 0x00D4, 0x1ED4 },
    { 0x00E2, 0x1EA9 },
    { 0x00EA, 0x1EC3 },
    { 0x00F4, 0x1ED5 },
    { 0x0102, 0x1EB2 },
    { 0x0103, 0x1EB3 },
    { 0x01A0, 0x1EDE },
    { 0x01A1, 0x1EDF },
    { 0x01AF, 0x1EEC },
    { 0x01B0, 0x1EED },
#define COMP_TABLE_IDX_0323 (COMP_TABLE_IDX_0309 + COMP_TABLE_LEN_0309)
#define COMP_TABLE_LEN_0323 50
    { 0x0041, 0x1EA0 },
    { 0x0042, 0x1E04 },
    { 0x0044, 0x1E0C },
    { 0x0045, 0x1EB8 },
    { 0x0048, 0x1E24 },
    { 0x0049, 0x1ECA },
    { 0x004B, 0x1E32 },
    { 0x004C, 0x1E36 },
    { 0x004D, 0x1E42 },
    { 0x004E, 0x1E46 },
    { 0x004F, 0x1ECC },
    { 0x0052, 0x1E5A },
    { 0x0053, 0x1E62 },
    { 0x0054, 0x1E6C },
    { 0x0055, 0x1EE4 },
    { 0x0056, 0x1E7E },
    { 0x0057, 0x1E88 },
    { 0x0059, 0x1EF4 },
    { 0x005A, 0x1E92 },
    { 0x0061, 0x1EA1 },
    { 0x0062, 0x1E05 },
    { 0x0064, 0x1E0D },
    { 0x0065, 0x1EB9 },
    { 0x0068, 0x1E25 },
    { 0x0069, 0x1ECB },
    { 0x006B, 0x1E33 },
    { 0x006C, 0x1E37 },
    { 0x006D, 0x1E43 },
    { 0x006E, 0x1E47 },
    { 0x006F, 0x1ECD },
    { 0x0072, 0x1E5B },
    { 0x0073, 0x1E63 },
    { 0x0074, 0x1E6D },
    { 0x0075, 0x1EE5 },
    { 0x0076, 0x1E7F },
    { 0x0077, 0x1E89 },
    { 0x0079, 0x1EF5 },
    { 0x007A, 0x1E93 },
    { 0x00C2, 0x1EAC },
    { 0x00CA, 0x1EC6 },
    { 0x00D4, 0x1ED8 },
    { 0x00E2, 0x1EAD },
    { 0x00EA, 0x1EC7 },
    { 0x00F4, 0x1ED9 },
    { 0x0102, 0x1EB6 },
    { 0x0103, 0x1EB7 },
    { 0x01A0, 0x1EE2 },
    { 0x01A1, 0x1EE3 },
    { 0x01AF, 0x1EF0 },
    { 0x01B0, 0x1EF1 },
#define COMP_TABLE_IDX_END (COMP_TABLE_IDX_0323 + COMP_TABLE_LEN_0323)
  };
/* Compile-time verification of table size.  */
typedef int verify1[(NELEMS (comp_table_data) == COMP_TABLE_IDX_END) - 1];

static const struct
{
  unsigned int idx;
  unsigned int len;
} comp_table[5] =
  {
    { COMP_TABLE_IDX_0300, COMP_TABLE_LEN_0300 },
    { COMP_TABLE_IDX_0301, COMP_TABLE_LEN_0301 },
    { COMP_TABLE_IDX_0303, COMP_TABLE_LEN_0303 },
    { COMP_TABLE_IDX_0309, COMP_TABLE_LEN_0309 },
    { COMP_TABLE_IDX_0323, COMP_TABLE_LEN_0323 },
  };


/* First define the conversion function from TCVN5712-1 to UCS4.  */
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
    if (ch < 0x18)							      \
      ch = map_from_tcvn_low[ch];					      \
    else if (ch >= 0x80)						      \
      ch = map_from_tcvn_high[ch - 0x80];				      \
									      \
    /* Determine whether there is a buffered character pending.  */	      \
    last_ch = *statep >> 3;						      \
									      \
    /* We have to buffer ch if it is a possible match in comp_table_data.  */ \
    must_buffer_ch = (ch >= 0x0041 && ch <= 0x01b0);                          \
									      \
    if (last_ch)							      \
      {									      \
	if (ch >= 0x0300 && ch < 0x0340)				      \
	  {								      \
	    /* See whether last_ch and ch can be combined.  */		      \
	    unsigned int i;						      \
	    unsigned int i1;						      \
	    unsigned int i2;						      \
									      \
	    switch (ch)							      \
	      {								      \
	      case 0x0300:						      \
		i = 0;							      \
		break;							      \
	      case 0x0301:						      \
		i = 1;							      \
		break;							      \
	      case 0x0303:						      \
		i = 2;							      \
		break;							      \
	      case 0x0309:						      \
		i = 3;							      \
		break;							      \
	      case 0x0323:						      \
		i = 4;							      \
		break;							      \
	      default:							      \
		abort ();						      \
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
		/* Output the combined character.  */			      \
		put32 (outptr, last_ch);				      \
		outptr += 4;						      \
		*statep = 0;						      \
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
#define EXTRA_LOOP_DECLS	, int *statep
#define ONEBYTE_BODY \
  {									      \
    uint32_t ch;							      \
									      \
    if (c < 0x18)							      \
      ch = map_from_tcvn_low[c];					      \
    else if (c >= 0x80)							      \
      ch = map_from_tcvn_high[c - 0x80];				      \
    else								      \
      ch = c;								      \
    if (ch >= 0x0041 && ch <= 0x01b0)					      \
      return WEOF;							      \
    return ch;								      \
  }
#include <iconv/loop.c>


/* Next, define the conversion function from UCS4 to TCVN5712-1.  */

static const unsigned char from_ucs4[] =
  {
#define FROM_IDX_00 0
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x07, /* 0x0001-0x0007 */
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, /* 0x0008-0x000f */
    0x10,                                           /* 0x0010-0x0010 */

#define FROM_IDX_01 (FROM_IDX_00 + 16)
    0x80, 0x83, 0xa2, 0x82, 0x00, 0x00, 0x00, 0x00, /* 0x00c0-0x00c7 */
    0x87, 0x8a, 0xa3, 0x00, 0x8d, 0x90, 0x00, 0x00, /* 0x00c8-0x00cf */
    0x00, 0x00, 0x92, 0x95, 0xa4, 0x94, 0x00, 0x00, /* 0x00d0-0x00d7 */
    0x00, 0x9d, 0x01, 0x00, 0x00, 0x16, 0x00, 0x00, /* 0x00d8-0x00df */
    0xb5, 0xb8, 0xa9, 0xb7, 0x00, 0x00, 0x00, 0x00, /* 0x00e0-0x00e7 */
    0xcc, 0xd0, 0xaa, 0x00, 0xd7, 0xdd, 0x00, 0x00, /* 0x00e8-0x00ef */
    0x00, 0x00, 0xdf, 0xe3, 0xab, 0xe2, 0x00, 0x00, /* 0x00f0-0x00f7 */
    0x00, 0xef, 0xf3, 0x00, 0x00, 0xfd, 0x00, 0x00, /* 0x00f8-0x00ff */
    0x00, 0x00, 0xa1, 0xa8, 0x00, 0x00, 0x00, 0x00, /* 0x0100-0x0107 */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x0108-0x010f */
    0xa7, 0xae, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x0110-0x0117 */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x0118-0x011f */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x0120-0x0127 */
    0x8f, 0xdc,                                     /* 0x0128-0x0129 */

#define FROM_IDX_02 (FROM_IDX_01 + 106)
    0x9f, 0xf2,                                     /* 0x0168-0x0169 */

#define FROM_IDX_03 (FROM_IDX_02 + 2)
    0xa5, 0xac, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x01a0-0x01a7 */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa6, /* 0x01a8-0x01af */
    0xad,                                           /* 0x01b0-0x01b0 */

#define FROM_IDX_04 (FROM_IDX_03 + 17)
    0xb0, 0xb3, 0x00, 0xb2, 0x00, 0x00, 0x00, 0x00, /* 0x0300-0x0307 */
    0x00, 0xb1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x0308-0x030f */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x0310-0x0317 */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, /* 0x0318-0x031f */
    0x00, 0x00, 0x00, 0xb4,                         /* 0x0320-0x0323 */

#define FROM_IDX_05 (FROM_IDX_04 + 36)
    0x84, 0xb9, 0x81, 0xb6, 0xc4, 0xca, 0xc1, 0xc7, /* 0x1ea0-0x1ea7 */
    0xc2, 0xc8, 0xc3, 0xc9, 0x86, 0xcb, 0xc0, 0xbe, /* 0x1ea8-0x1eaf */
    0xaf, 0xbb, 0xba, 0xbc, 0xbf, 0xbd, 0x85, 0xc6, /* 0x1eb0-0x1eb7 */
    0x8b, 0xd1, 0x88, 0xce, 0x89, 0xcf, 0xda, 0xd5, /* 0x1eb8-0x1ebf */
    0xc5, 0xd2, 0xcd, 0xd3, 0xd9, 0xd4, 0x8c, 0xd6, /* 0x1ec0-0x1ec7 */
    0x8e, 0xd8, 0x91, 0xde, 0x96, 0xe4, 0x93, 0xe1, /* 0x1ec8-0x1ecf */
    0xff, 0xe8, 0xdb, 0xe5, 0xe0, 0xe6, 0xf0, 0xe7, /* 0x1ed0-0x1ed7 */
    0x97, 0xe9, 0x9b, 0xed, 0x98, 0xea, 0x99, 0xeb, /* 0x1ed8-0x1edf */
    0x9a, 0xec, 0x9c, 0xee, 0x02, 0xf4, 0x9e, 0xf1, /* 0x1ee0-0x1ee7 */
    0x11, 0xf8, 0x04, 0xf5, 0x05, 0xf6, 0x06, 0xf7, /* 0x1ee8-0x1eef */
    0x12, 0xf9, 0x13, 0xfa, 0x17, 0xfe, 0x14, 0xfb, /* 0x1ef0-0x1ef7 */
    0x15, 0xfc                                      /* 0x1ef8-0x1ef9 */
#define FROM_IDX_END (FROM_IDX_05 + 90)
  };
/* Compile-time verification of table size.  */
typedef int verify2[(NELEMS (from_ucs4) == FROM_IDX_END) - 1];

/* Decomposition table for the relevant Unicode characters. */
static const struct
{
  uint16_t composed;
  uint32_t base:8;
  uint32_t comb1:8;
} decomp_table[] =
  {
    { 0x00d1, 0x4e, 0xb2 },
    { 0x00f1, 0x6e, 0xb2 },
    { 0x0106, 0x43, 0xb3 },
    { 0x0107, 0x63, 0xb3 },
    { 0x0139, 0x4c, 0xb3 },
    { 0x013a, 0x6c, 0xb3 },
    { 0x0143, 0x4e, 0xb3 },
    { 0x0144, 0x6e, 0xb3 },
    { 0x0154, 0x52, 0xb3 },
    { 0x0155, 0x72, 0xb3 },
    { 0x015a, 0x53, 0xb3 },
    { 0x015b, 0x73, 0xb3 },
    { 0x0179, 0x5a, 0xb3 },
    { 0x017a, 0x7a, 0xb3 },
    { 0x01f4, 0x47, 0xb3 },
    { 0x01f5, 0x67, 0xb3 },
    { 0x01f8, 0x4e, 0xb0 },
    { 0x01f9, 0x6e, 0xb0 },
    { 0x1e04, 0x42, 0xb4 },
    { 0x1e05, 0x62, 0xb4 },
    { 0x1e0c, 0x44, 0xb4 },
    { 0x1e0d, 0x64, 0xb4 },
    { 0x1e24, 0x48, 0xb4 },
    { 0x1e25, 0x68, 0xb4 },
    { 0x1e30, 0x4b, 0xb3 },
    { 0x1e31, 0x6b, 0xb3 },
    { 0x1e32, 0x4b, 0xb4 },
    { 0x1e33, 0x6b, 0xb4 },
    { 0x1e36, 0x4c, 0xb4 },
    { 0x1e37, 0x6c, 0xb4 },
    { 0x1e3e, 0x4d, 0xb3 },
    { 0x1e3f, 0x6d, 0xb3 },
    { 0x1e42, 0x4d, 0xb4 },
    { 0x1e43, 0x6d, 0xb4 },
    { 0x1e46, 0x4e, 0xb4 },
    { 0x1e47, 0x6e, 0xb4 },
    { 0x1e54, 0x50, 0xb3 },
    { 0x1e55, 0x70, 0xb3 },
    { 0x1e5a, 0x52, 0xb4 },
    { 0x1e5b, 0x72, 0xb4 },
    { 0x1e62, 0x53, 0xb4 },
    { 0x1e63, 0x73, 0xb4 },
    { 0x1e6c, 0x54, 0xb4 },
    { 0x1e6d, 0x74, 0xb4 },
    { 0x1e7c, 0x56, 0xb2 },
    { 0x1e7d, 0x76, 0xb2 },
    { 0x1e7e, 0x56, 0xb4 },
    { 0x1e7f, 0x76, 0xb4 },
    { 0x1e80, 0x57, 0xb0 },
    { 0x1e81, 0x77, 0xb0 },
    { 0x1e82, 0x57, 0xb3 },
    { 0x1e83, 0x77, 0xb3 },
    { 0x1e88, 0x57, 0xb4 },
    { 0x1e89, 0x77, 0xb4 },
    { 0x1e92, 0x5a, 0xb4 },
    { 0x1e93, 0x7a, 0xb4 },
  };


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
    if (ch == 0x00 || (ch >= 0x18 && ch < 0x80) || ch == 0xa0)		      \
      {									      \
	*outptr++ = ch;							      \
	inptr += 4;							      \
      }									      \
    else								      \
      {									      \
	unsigned char res;						      \
									      \
	if (ch <= 0x0010)						      \
	  res = from_ucs4[ch - 0x0001 + FROM_IDX_00];			      \
	else if (ch >= 0x00c0 && ch <= 0x0129)				      \
	  res = from_ucs4[ch - 0x00c0 + FROM_IDX_01];			      \
	else if (ch >= 0x0168 && ch <= 0x0169)				      \
	  res = from_ucs4[ch - 0x0168 + FROM_IDX_02];			      \
	else if (ch >= 0x01a0 && ch <= 0x01b0)				      \
	  res = from_ucs4[ch - 0x01a0 + FROM_IDX_03];			      \
	else if (ch >= 0x0300 && ch <= 0x0323)				      \
	  res = from_ucs4[ch - 0x0300 + FROM_IDX_04];			      \
	else if (ch >= 0x1ea0 && ch <= 0x1ef9)				      \
	  res = from_ucs4[ch - 0x1ea0 + FROM_IDX_05];			      \
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
	    unsigned int i1;						      \
	    unsigned int i2;						      \
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
		/* See whether we have room for two bytes.  */		      \
		if (__glibc_unlikely (outptr + 1 >= outend))		      \
		  {							      \
		    result = __GCONV_FULL_OUTPUT;			      \
		    break;						      \
		  }							      \
									      \
		/* Found a canonical decomposition.  */			      \
		*outptr++ = decomp_table[i].base;			      \
		*outptr++ = decomp_table[i].comb1;			      \
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
