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

#ifndef _MEMCOPY_H
#define _MEMCOPY_H	1

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

#include <sys/cdefs.h>
#include <endian.h>
#include <pagecopy.h>

/* The macros defined in this file are:

   BYTE_COPY_FWD(dst_beg_ptr, src_beg_ptr, nbytes_to_copy)

   BYTE_COPY_BWD(dst_end_ptr, src_end_ptr, nbytes_to_copy)

   WORD_COPY_FWD(dst_beg_ptr, src_beg_ptr, nbytes_remaining, nbytes_to_copy)

   WORD_COPY_BWD(dst_end_ptr, src_end_ptr, nbytes_remaining, nbytes_to_copy)

   MERGE(old_word, sh_1, new_word, sh_2)
     [I fail to understand.  I feel stupid.  --roland]
*/

/* Type to use for aligned memory operations.
   This should normally be the biggest type supported by a single load
   and store.  */
#define	op_t	unsigned long int
#define OPSIZ	(sizeof (op_t))

/* Type to use for unaligned operations.  */
typedef unsigned char byte;

#if __BYTE_ORDER == __LITTLE_ENDIAN
#define MERGE(w0, sh_1, w1, sh_2) (((w0) >> (sh_1)) | ((w1) << (sh_2)))
#endif
#if __BYTE_ORDER == __BIG_ENDIAN
#define MERGE(w0, sh_1, w1, sh_2) (((w0) << (sh_1)) | ((w1) >> (sh_2)))
#endif

/* Copy exactly NBYTES bytes from SRC_BP to DST_BP,
   without any assumptions about alignment of the pointers.  */
#define BYTE_COPY_FWD(dst_bp, src_bp, nbytes)				      \
  do									      \
    {									      \
      size_t __nbytes = (nbytes);					      \
      while (__nbytes > 0)						      \
	{								      \
	  byte __x = ((byte *) src_bp)[0];				      \
	  src_bp += 1;							      \
	  __nbytes -= 1;						      \
	  ((byte *) dst_bp)[0] = __x;					      \
	  dst_bp += 1;							      \
	}								      \
    } while (0)

/* Copy exactly NBYTES_TO_COPY bytes from SRC_END_PTR to DST_END_PTR,
   beginning at the bytes right before the pointers and continuing towards
   smaller addresses.  Don't assume anything about alignment of the
   pointers.  */
#define BYTE_COPY_BWD(dst_ep, src_ep, nbytes)				      \
  do									      \
    {									      \
      size_t __nbytes = (nbytes);					      \
      while (__nbytes > 0)						      \
	{								      \
	  byte __x;							      \
	  src_ep -= 1;							      \
	  __x = ((byte *) src_ep)[0];					      \
	  dst_ep -= 1;							      \
	  __nbytes -= 1;						      \
	  ((byte *) dst_ep)[0] = __x;					      \
	}								      \
    } while (0)

/* Copy *up to* NBYTES bytes from SRC_BP to DST_BP, with
   the assumption that DST_BP is aligned on an OPSIZ multiple.  If
   not all bytes could be easily copied, store remaining number of bytes
   in NBYTES_LEFT, otherwise store 0.  */
extern void _wordcopy_fwd_aligned (long int, long int, size_t)
  attribute_hidden __THROW;
extern void _wordcopy_fwd_dest_aligned (long int, long int, size_t)
  attribute_hidden __THROW;
#define WORD_COPY_FWD(dst_bp, src_bp, nbytes_left, nbytes)		      \
  do									      \
    {									      \
      if (src_bp % OPSIZ == 0)						      \
	_wordcopy_fwd_aligned (dst_bp, src_bp, (nbytes) / OPSIZ);	      \
      else								      \
	_wordcopy_fwd_dest_aligned (dst_bp, src_bp, (nbytes) / OPSIZ);	      \
      src_bp += (nbytes) & -OPSIZ;					      \
      dst_bp += (nbytes) & -OPSIZ;					      \
      (nbytes_left) = (nbytes) % OPSIZ;					      \
    } while (0)

/* Copy *up to* NBYTES_TO_COPY bytes from SRC_END_PTR to DST_END_PTR,
   beginning at the words (of type op_t) right before the pointers and
   continuing towards smaller addresses.  May take advantage of that
   DST_END_PTR is aligned on an OPSIZ multiple.  If not all bytes could be
   easily copied, store remaining number of bytes in NBYTES_REMAINING,
   otherwise store 0.  */
extern void _wordcopy_bwd_aligned (long int, long int, size_t)
  attribute_hidden __THROW;
extern void _wordcopy_bwd_dest_aligned (long int, long int, size_t)
  attribute_hidden __THROW;
#define WORD_COPY_BWD(dst_ep, src_ep, nbytes_left, nbytes)		      \
  do									      \
    {									      \
      if (src_ep % OPSIZ == 0)						      \
	_wordcopy_bwd_aligned (dst_ep, src_ep, (nbytes) / OPSIZ);	      \
      else								      \
	_wordcopy_bwd_dest_aligned (dst_ep, src_ep, (nbytes) / OPSIZ);	      \
      src_ep -= (nbytes) & -OPSIZ;					      \
      dst_ep -= (nbytes) & -OPSIZ;					      \
      (nbytes_left) = (nbytes) % OPSIZ;					      \
    } while (0)

/* The macro PAGE_COPY_FWD_MAYBE (dstp, srcp, nbytes_left, nbytes) is invoked
   like WORD_COPY_FWD et al.  The pointers should be at least word aligned.
   This will check if virtual copying by pages can and should be done and do it
   if so.  The pointers will be aligned to PAGE_SIZE bytes.  The macro requires
   that pagecopy.h defines at least PAGE_COPY_THRESHOLD to 0.  If
   PAGE_COPY_THRESHOLD is non-zero, the header must also define PAGE_COPY_FWD
   and PAGE_SIZE.
*/
#if PAGE_COPY_THRESHOLD

# include <assert.h>

# define PAGE_COPY_FWD_MAYBE(dstp, srcp, nbytes_left, nbytes)		      \
  do									      \
    {									      \
      if ((nbytes) >= PAGE_COPY_THRESHOLD				      \
	  && PAGE_OFFSET ((dstp) - (srcp)) == 0)			      \
	{								      \
	  /* The amount to copy is past the threshold for copying	      \
	     pages virtually with kernel VM operations, and the		      \
	     source and destination addresses have the same alignment.  */    \
	  size_t nbytes_before = PAGE_OFFSET (-(dstp));			      \
	  if (nbytes_before != 0)					      \
	    {								      \
	      /* First copy the words before the first page boundary.  */     \
	      WORD_COPY_FWD (dstp, srcp, nbytes_left, nbytes_before);	      \
	      assert (nbytes_left == 0);				      \
	      nbytes -= nbytes_before;					      \
	    }								      \
	  PAGE_COPY_FWD (dstp, srcp, nbytes_left, nbytes);		      \
	}								      \
    } while (0)

/* The page size is always a power of two, so we can avoid modulo division.  */
# define PAGE_OFFSET(n)	((n) & (PAGE_SIZE - 1))

#else

# define PAGE_COPY_FWD_MAYBE(dstp, srcp, nbytes_left, nbytes) /* nada */

#endif

/* Threshold value for when to enter the unrolled loops.  */
#define	OP_T_THRES	16

/* Set to 1 if memcpy is safe to use for forward-copying memmove with
   overlapping addresses.  This is 0 by default because memcpy implementations
   are generally not safe for overlapping addresses.  */
#define MEMCPY_OK_FOR_FWD_MEMMOVE 0

#endif /* memcopy.h */
