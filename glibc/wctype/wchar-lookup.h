/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <stdint.h>

/* Tables indexed by a wide character are compressed through the use
   of a multi-level lookup.  The compression effect comes from blocks
   that don't need particular data and from blocks that can share their
   data.  */

/* Bit tables are accessed by cutting wc in four blocks of bits:
   - the high 32-q-p bits,
   - the next q bits,
   - the next p bits,
   - the next 5 bits.

	    +------------------+-----+-----+-----+
     wc  =  +     32-q-p-5     |  q  |  p  |  5  |
	    +------------------+-----+-----+-----+

   p and q are variable.  For 16-bit Unicode it is sufficient to
   choose p and q such that q+p+5 <= 16.

   The table contains the following uint32_t words:
   - q+p+5,
   - s = upper exclusive bound for wc >> (q+p+5),
   - p+5,
   - 2^q-1,
   - 2^p-1,
   - 1st-level table: s offsets, pointing into the 2nd-level table,
   - 2nd-level table: k*2^q offsets, pointing into the 3rd-level table,
   - 3rd-level table: j*2^p words, each containing 32 bits of data.
*/

static __inline int
__attribute ((always_inline))
wctype_table_lookup (const char *table, uint32_t wc)
{
  uint32_t shift1 = ((const uint32_t *) table)[0];
  uint32_t index1 = wc >> shift1;
  uint32_t bound = ((const uint32_t *) table)[1];
  if (index1 < bound)
    {
      uint32_t lookup1 = ((const uint32_t *) table)[5 + index1];
      if (lookup1 != 0)
	{
	  uint32_t shift2 = ((const uint32_t *) table)[2];
	  uint32_t mask2 = ((const uint32_t *) table)[3];
	  uint32_t index2 = (wc >> shift2) & mask2;
	  uint32_t lookup2 = ((const uint32_t *)(table + lookup1))[index2];
	  if (lookup2 != 0)
	    {
	      uint32_t mask3 = ((const uint32_t *) table)[4];
	      uint32_t index3 = (wc >> 5) & mask3;
	      uint32_t lookup3 = ((const uint32_t *)(table + lookup2))[index3];

	      return (lookup3 >> (wc & 0x1f)) & 1;
	    }
	}
    }
  return 0;
}

/* Byte tables are similar to bit tables, except that the addressing
   unit is a single byte, and no 5 bits are used as a word index.  */

static __inline int
__attribute ((always_inline))
wcwidth_table_lookup (const char *table, uint32_t wc)
{
  uint32_t shift1 = ((const uint32_t *) table)[0];
  uint32_t index1 = wc >> shift1;
  uint32_t bound = ((const uint32_t *) table)[1];
  if (index1 < bound)
    {
      uint32_t lookup1 = ((const uint32_t *) table)[5 + index1];
      if (lookup1 != 0)
	{
	  uint32_t shift2 = ((const uint32_t *) table)[2];
	  uint32_t mask2 = ((const uint32_t *) table)[3];
	  uint32_t index2 = (wc >> shift2) & mask2;
	  uint32_t lookup2 = ((const uint32_t *)(table + lookup1))[index2];
	  if (lookup2 != 0)
	    {
	      uint32_t mask3 = ((const uint32_t *) table)[4];
	      uint32_t index3 = wc & mask3;
	      uint8_t lookup3 = ((const uint8_t *)(table + lookup2))[index3];

	      return lookup3;
	    }
	}
    }
  return 0xff;
}

/* Mapping tables are similar to bit tables, except that the
   addressing unit is a single signed 32-bit word, containing the
   difference between the desired result and the argument, and no 5
   bits are used as a word index.  */

static __inline uint32_t
__attribute ((always_inline))
wctrans_table_lookup (const char *table, uint32_t wc)
{
  uint32_t shift1 = ((const uint32_t *) table)[0];
  uint32_t index1 = wc >> shift1;
  uint32_t bound = ((const uint32_t *) table)[1];
  if (index1 < bound)
    {
      uint32_t lookup1 = ((const uint32_t *) table)[5 + index1];
      if (lookup1 != 0)
	{
	  uint32_t shift2 = ((const uint32_t *) table)[2];
	  uint32_t mask2 = ((const uint32_t *) table)[3];
	  uint32_t index2 = (wc >> shift2) & mask2;
	  uint32_t lookup2 = ((const uint32_t *)(table + lookup1))[index2];
	  if (lookup2 != 0)
	    {
	      uint32_t mask3 = ((const uint32_t *) table)[4];
	      uint32_t index3 = wc & mask3;
	      int32_t lookup3 = ((const int32_t *)(table + lookup2))[index3];

	      return wc + lookup3;
	    }
	}
    }
  return wc;
}
