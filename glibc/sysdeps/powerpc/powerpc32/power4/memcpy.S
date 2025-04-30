/* Optimized memcpy implementation for PowerPC32 on PowerPC64.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#include <sysdep.h>

/* void * [r3] memcpy (void *dst [r3], void *src [r4], size_t len [r5]);
   Returns 'dst'.

   Memcpy handles short copies (< 32-bytes) using a binary move blocks
   (no loops) of lwz/stw.  The tail (remaining 1-3) bytes is handled
   with the appropriate combination of byte and halfword load/stores.
   There is minimal effort to optimize the alignment of short moves.

   Longer moves (>= 32-bytes) justify the effort to get at least the
   destination word (4-byte) aligned.  Further optimization is
   possible when both source and destination are word aligned.
   Each case has an optimized unrolled loop.   */

	.machine power4
EALIGN (memcpy, 5, 0)
	CALL_MCOUNT

    stwu  1,-32(1)
    cfi_adjust_cfa_offset(32)
    stw   30,20(1)
    cfi_offset(30,(20-32))
    mr    30,3
    cmplwi cr1,5,31
    stw   31,24(1)
    cfi_offset(31,(24-32))
    neg   0,3
    andi. 11,3,3	/* check alignment of dst.  */
    clrlwi 0,0,30	/* Number of bytes until the 1st word of dst.  */
    clrlwi 10,4,30	/* check alignment of src.  */
    cmplwi cr6,5,8
    ble-  cr1,.L2	/* If move < 32 bytes use short move code.  */
    cmplw cr6,10,11
    mr    12,4
    srwi  9,5,2		/* Number of full words remaining.  */
    mtcrf 0x01,0
    mr    31,5
    beq   .L0

    subf  31,0,5
  /* Move 0-3 bytes as needed to get the destination word aligned.  */
1:  bf    31,2f
    lbz   6,0(12)
    addi  12,12,1
    stb   6,0(3)
    addi  3,3,1
2:  bf    30,0f
    lhz   6,0(12)
    addi  12,12,2
    sth   6,0(3)
    addi  3,3,2
0:
    clrlwi 10,12,30	/* check alignment of src again.  */
    srwi  9,31,2	/* Number of full words remaining.  */

  /* Copy words from source to destination, assuming the destination is
     aligned on a word boundary.

     At this point we know there are at least 25 bytes left (32-7) to copy.
     The next step is to determine if the source is also word aligned.
     If not branch to the unaligned move code at .L6. which uses
     a load, shift, store strategy.

     Otherwise source and destination are word aligned, and we can use
     the optimized word copy loop.  */
.L0:
    clrlwi	11,31,30  /* calculate the number of tail bytes */
    mtcrf 0x01,9
    bne-  cr6,.L6   /* If source is not word aligned.  */

  /* Move words where destination and source are word aligned.
     Use an unrolled loop to copy 4 words (16-bytes) per iteration.
     If the copy is not an exact multiple of 16 bytes, 1-3
     words are copied as needed to set up the main loop.  After
     the main loop exits there may be a tail of 1-3 bytes. These bytes are
     copied a halfword/byte at a time as needed to preserve alignment.  */

    srwi  8,31,4    /* calculate the 16 byte loop count */
    cmplwi	cr1,9,4
    cmplwi	cr6,11,0
    mr    11,12

    bf    30,1f
    lwz   6,0(12)
    lwz   7,4(12)
    addi  11,12,8
    mtctr 8
    stw   6,0(3)
    stw   7,4(3)
    addi  10,3,8
    bf    31,4f
    lwz   0,8(12)
    stw   0,8(3)
    blt   cr1,3f
    addi  11,12,12
    addi  10,3,12
    b     4f
    .align  4
1:
    mr    10,3
    mtctr 8
    bf    31,4f
    lwz   6,0(12)
    addi  11,12,4
    stw   6,0(3)
    addi  10,3,4

    .align  4
4:
    lwz   6,0(11)
    lwz   7,4(11)
    lwz   8,8(11)
    lwz   0,12(11)
    stw   6,0(10)
    stw   7,4(10)
    stw   8,8(10)
    stw   0,12(10)
    addi  11,11,16
    addi  10,10,16
    bdnz  4b
3:
    clrrwi 0,31,2
    mtcrf 0x01,31
    beq   cr6,0f
.L9:
    add   3,3,0
    add   12,12,0

/*  At this point we have a tail of 0-3 bytes and we know that the
    destination is word aligned.  */
2:  bf    30,1f
    lhz   6,0(12)
    addi  12,12,2
    sth   6,0(3)
    addi  3,3,2
1:  bf    31,0f
    lbz   6,0(12)
    stb   6,0(3)
0:
  /* Return original dst pointer.  */
    mr  3,30
    lwz 30,20(1)
    lwz 31,24(1)
    addi 1,1,32
    blr

/* Copy up to 31 bytes.  This is divided into two cases 0-8 bytes and
   9-31 bytes.  Each case is handled without loops, using binary
   (1,2,4,8) tests.

   In the short (0-8 byte) case no attempt is made to force alignment
   of either source or destination.  The hardware will handle the
   unaligned load/stores with small delays for crossing 32- 64-byte, and
   4096-byte boundaries. Since these short moves are unlikely to be
   unaligned or cross these boundaries, the overhead to force
   alignment is not justified.

   The longer (9-31 byte) move is more likely to cross 32- or 64-byte
   boundaries.  Since only loads are sensitive to the 32-/64-byte
   boundaries it is more important to align the source than the
   destination.  If the source is not already word aligned, we first
   move 1-3 bytes as needed.  While the destination and stores may
   still be unaligned, this is only an issue for page (4096 byte
   boundary) crossing, which should be rare for these short moves.
   The hardware handles this case automatically with a small delay.  */

    .align  4
.L2:
    mtcrf 0x01,5
    neg   8,4
    clrrwi 11,4,2
    andi. 0,8,3
    ble   cr6,.LE8	/* Handle moves of 0-8 bytes.  */
/* At least 9 bytes left.  Get the source word aligned.  */
    cmplwi	cr1,5,16
    mr    10,5
    mr    12,4
    cmplwi	cr6,0,2
    beq   .L3	/* If the source is already word aligned skip this.  */
/* Copy 1-3 bytes to get source address word aligned.  */
    lwz   6,0(11)
    subf  10,0,5
    add   12,4,0
    blt   cr6,5f
    srwi  7,6,16
    bgt	  cr6,3f
#ifdef __LITTLE_ENDIAN__
    sth   7,0(3)
#else
    sth   6,0(3)
#endif
    b     7f
    .align  4
3:
#ifdef __LITTLE_ENDIAN__
    rotlwi 6,6,24
    stb   6,0(3)
    sth   7,1(3)
#else
    stb   7,0(3)
    sth   6,1(3)
#endif
    b     7f
    .align  4
5:
#ifdef __LITTLE_ENDIAN__
    rotlwi 6,6,8
#endif
    stb   6,0(3)
7:
    cmplwi	cr1,10,16
    add   3,3,0
    mtcrf 0x01,10
    .align  4
.L3:
/* At least 6 bytes left and the source is word aligned.  */
    blt   cr1,8f
16: /* Move 16 bytes.  */
    lwz   6,0(12)
    lwz   7,4(12)
    stw   6,0(3)
    lwz   6,8(12)
    stw   7,4(3)
    lwz   7,12(12)
    addi  12,12,16
    stw   6,8(3)
    stw   7,12(3)
    addi  3,3,16
8:  /* Move 8 bytes.  */
    bf    28,4f
    lwz   6,0(12)
    lwz   7,4(12)
    addi  12,12,8
    stw   6,0(3)
    stw   7,4(3)
    addi  3,3,8
4:  /* Move 4 bytes.  */
    bf    29,2f
    lwz   6,0(12)
    addi  12,12,4
    stw   6,0(3)
    addi  3,3,4
2:  /* Move 2-3 bytes.  */
    bf    30,1f
    lhz   6,0(12)
    sth   6,0(3)
    bf    31,0f
    lbz   7,2(12)
    stb   7,2(3)
    mr    3,30
    lwz   30,20(1)
    addi  1,1,32
    blr
1:  /* Move 1 byte.  */
    bf    31,0f
    lbz   6,0(12)
    stb   6,0(3)
0:
  /* Return original dst pointer.  */
    mr   3,30
    lwz  30,20(1)
    addi 1,1,32
    blr

/* Special case to copy 0-8 bytes.  */
    .align  4
.LE8:
    mr    12,4
    bne   cr6,4f
    lwz   6,0(4)
    lwz   7,4(4)
    stw   6,0(3)
    stw   7,4(3)
  /* Return original dst pointer.  */
    mr    3,30
    lwz   30,20(1)
    addi  1,1,32
    blr
    .align  4
4:  bf    29,2b
    lwz   6,0(4)
    stw   6,0(3)
6:
    bf    30,5f
    lhz   7,4(4)
    sth   7,4(3)
    bf    31,0f
    lbz   8,6(4)
    stb   8,6(3)
    mr    3,30
    lwz   30,20(1)
    addi  1,1,32
    blr
    .align  4
5:
    bf    31,0f
    lbz   6,4(4)
    stb   6,4(3)
    .align  4
0:
  /* Return original dst pointer.  */
    mr   3,30
    lwz  30,20(1)
    addi 1,1,32
    blr

    .align  4
.L6:

  /* Copy words where the destination is aligned but the source is
     not.  Use aligned word loads from the source, shifted to realign
     the data, to allow aligned destination stores.
     Use an unrolled loop to copy 4 words (16-bytes) per iteration.
     A single word is retained for storing at loop exit to avoid walking
     off the end of a page within the loop.
     If the copy is not an exact multiple of 16 bytes, 1-3
     words are copied as needed to set up the main loop.  After
     the main loop exits there may be a tail of 1-3 bytes. These bytes are
     copied a halfword/byte at a time as needed to preserve alignment.  */


    cmplwi  cr6,11,0  /* are there tail bytes left ? */
    subf    5,10,12   /* back up src pointer to prev word alignment */
    slwi    10,10,3   /* calculate number of bits to shift 1st word left */
    addi    11,9,-1   /* we move one word after the loop */
    srwi    8,11,2    /* calculate the 16 byte loop count */
    lwz     6,0(5)    /* load 1st src word into R6 */
    mr      4,3
    lwz     7,4(5)    /* load 2nd src word into R7 */
    mtcrf   0x01,11
    subfic  9,10,32   /* number of bits to shift 2nd word right */
    mtctr   8
    bf      30,1f

    /* there are at least two words to copy, so copy them */
#ifdef __LITTLE_ENDIAN__
    srw   0,6,10
    slw   8,7,9
#else
    slw   0,6,10  /* shift 1st src word to left align it in R0 */
    srw   8,7,9   /* shift 2nd src word to right align it in R8 */
#endif
    or    0,0,8   /* or them to get word to store */
    lwz   6,8(5)  /* load the 3rd src word */
    stw   0,0(4)  /* store the 1st dst word */
#ifdef __LITTLE_ENDIAN__
    srw   0,7,10
    slw   8,6,9
#else
    slw   0,7,10  /* now left align 2nd src word into R0 */
    srw   8,6,9   /* shift 3rd src word to right align it in R8 */
#endif
    or    0,0,8   /* or them to get word to store */
    lwz   7,12(5)
    stw   0,4(4)  /* store the 2nd dst word */
    addi  4,4,8
    addi  5,5,16
    bf    31,4f
    /* there is a third word to copy, so copy it */
#ifdef __LITTLE_ENDIAN__
    srw   0,6,10
    slw   8,7,9
#else
    slw   0,6,10  /* shift 3rd src word to left align it in R0 */
    srw   8,7,9   /* shift 4th src word to right align it in R8 */
#endif
    or    0,0,8   /* or them to get word to store */
    stw   0,0(4)  /* store 3rd dst word */
    mr    6,7
    lwz   7,0(5)
    addi  5,5,4
    addi  4,4,4
    b     4f
    .align 4
1:
#ifdef __LITTLE_ENDIAN__
    srw     0,6,10
    slw     8,7,9
#else
    slw     0,6,10  /* shift 1st src word to left align it in R0 */
    srw     8,7,9   /* shift 2nd src word to right align it in R8 */
#endif
    addi  5,5,8
    or    0,0,8   /* or them to get word to store */
    bf    31,4f
    mr    6,7
    lwz   7,0(5)
    addi  5,5,4
    stw   0,0(4)  /* store the 1st dst word */
    addi  4,4,4

    .align  4
4:
    /* copy 16 bytes at a time */
#ifdef __LITTLE_ENDIAN__
    srw   0,6,10
    slw   8,7,9
#else
    slw   0,6,10
    srw   8,7,9
#endif
    or    0,0,8
    lwz   6,0(5)
    stw   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srw   0,7,10
    slw   8,6,9
#else
    slw   0,7,10
    srw   8,6,9
#endif
    or    0,0,8
    lwz   7,4(5)
    stw   0,4(4)
#ifdef __LITTLE_ENDIAN__
    srw   0,6,10
    slw   8,7,9
#else
    slw   0,6,10
    srw   8,7,9
#endif
    or    0,0,8
    lwz   6,8(5)
    stw   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srw   0,7,10
    slw   8,6,9
#else
    slw   0,7,10
    srw   8,6,9
#endif
    or    0,0,8
    lwz   7,12(5)
    stw   0,12(4)
    addi  5,5,16
    addi  4,4,16
    bdnz+ 4b
8:
    /* calculate and store the final word */
#ifdef __LITTLE_ENDIAN__
    srw   0,6,10
    slw   8,7,9
#else
    slw   0,6,10
    srw   8,7,9
#endif
    or    0,0,8
    stw   0,0(4)
3:
    clrrwi 0,31,2
    mtcrf 0x01,31
    bne   cr6,.L9	/* If the tail is 0 bytes we are done!  */

  /* Return original dst pointer.  */
    mr   3,30
    lwz  30,20(1)
    lwz  31,24(1)
    addi 1,1,32
    blr
END (memcpy)

libc_hidden_builtin_def (memcpy)
