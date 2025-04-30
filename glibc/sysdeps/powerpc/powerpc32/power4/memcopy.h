/* memcopy.h -- definitions for memory copy functions.  Generic C version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Torbjorn Granlund (tege@sics.se).

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

/* The strategy of the memory functions is:

     1. Copy bytes until the destination pointer is aligned.

     2. Copy words in unrolled loops.  If the source and destination
     are not aligned in the same way, use word memory operations,
     but shift and merge two read words before writing.

     3. Copy the few remaining bytes.

   This is fast on processors that have at least 10 registers for
   allocation by GCC, and that can access memory at reg+const in one
   instruction.

   I made an "exhaustive" test of this memmove when I wrote it,
   exhaustive in the sense that I tried all alignment and length
   combinations, with and without overlap.  */

#include <sysdeps/generic/memcopy.h>

/* The macros defined in this file are:

   BYTE_COPY_FWD(dst_beg_ptr, src_beg_ptr, nbytes_to_copy)

   BYTE_COPY_BWD(dst_end_ptr, src_end_ptr, nbytes_to_copy)

   WORD_COPY_FWD(dst_beg_ptr, src_beg_ptr, nbytes_remaining, nbytes_to_copy)

   WORD_COPY_BWD(dst_end_ptr, src_end_ptr, nbytes_remaining, nbytes_to_copy)

   MERGE(old_word, sh_1, new_word, sh_2)
     [I fail to understand.  I feel stupid.  --roland]
*/


/* Threshold value for when to enter the unrolled loops.  */
#undef	OP_T_THRES
#define OP_T_THRES 16

/* Copy exactly NBYTES bytes from SRC_BP to DST_BP,
   without any assumptions about alignment of the pointers.  */
#undef BYTE_COPY_FWD
#define BYTE_COPY_FWD(dst_bp, src_bp, nbytes)				      \
  do									      \
    {									      \
      size_t __nbytes = (nbytes);					      \
      if (__nbytes & 1)							      \
        {								      \
	  ((byte *) dst_bp)[0] =  ((byte *) src_bp)[0];			      \
	  src_bp += 1;							      \
	  dst_bp += 1;							      \
	  __nbytes -= 1;						      \
        }								      \
      while (__nbytes > 0)						      \
	{								      \
	  byte __x = ((byte *) src_bp)[0];				      \
	  byte __y = ((byte *) src_bp)[1];				      \
	  src_bp += 2;							      \
	  __nbytes -= 2;						      \
	  ((byte *) dst_bp)[0] = __x;					      \
	  ((byte *) dst_bp)[1] = __y;					      \
	  dst_bp += 2;							      \
	}								      \
    } while (0)

/* Copy exactly NBYTES_TO_COPY bytes from SRC_END_PTR to DST_END_PTR,
   beginning at the bytes right before the pointers and continuing towards
   smaller addresses.  Don't assume anything about alignment of the
   pointers.  */
#undef BYTE_COPY_BWD
#define BYTE_COPY_BWD(dst_ep, src_ep, nbytes)				      \
  do									      \
    {									      \
      size_t __nbytes = (nbytes);					      \
      if (__nbytes & 1)							      \
        {								      \
	  src_ep -= 1;							      \
	  dst_ep -= 1;							      \
	  ((byte *) dst_ep)[0] =  ((byte *) src_ep)[0];			      \
	  __nbytes -= 1;						      \
        }								      \
      while (__nbytes > 0)						      \
	{								      \
	  byte __x, __y;						      \
	  src_ep -= 2;							      \
	  __y = ((byte *) src_ep)[1];					      \
	  __x = ((byte *) src_ep)[0];					      \
	  dst_ep -= 2;							      \
	  __nbytes -= 2;						      \
	  ((byte *) dst_ep)[1] = __y;					      \
	  ((byte *) dst_ep)[0] = __x;					      \
	}								      \
    } while (0)

/* The powerpc memcpy implementation is safe to use for memmove.  */
#undef MEMCPY_OK_FOR_FWD_MEMMOVE
#define MEMCPY_OK_FOR_FWD_MEMMOVE 1
