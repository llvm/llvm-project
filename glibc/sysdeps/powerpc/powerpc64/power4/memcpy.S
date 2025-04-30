/* Optimized memcpy implementation for PowerPC64.
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
   The 64-bit implementations of POWER3 and POWER4 do a reasonable job
   of handling unaligned load/stores that do not cross 32-byte boundaries.

   Longer moves (>= 32-bytes) justify the effort to get at least the
   destination doubleword (8-byte) aligned.  Further optimization is
   possible when both source and destination are doubleword aligned.
   Each case has a optimized unrolled loop.   */

#ifndef MEMCPY
# define MEMCPY memcpy
#endif
	.machine power4
ENTRY_TOCLESS (MEMCPY, 5)
	CALL_MCOUNT 3

    cmpldi cr1,5,31
    neg   0,3
    std   3,-16(1)
    std   31,-8(1)
    cfi_offset(31,-8)
    andi. 11,3,7	/* check alignment of dst.  */
    clrldi 0,0,61	/* Number of bytes until the 1st doubleword of dst.  */
    clrldi 10,4,61	/* check alignment of src.  */
    cmpldi cr6,5,8
    ble-  cr1,.L2	/* If move < 32 bytes use short move code.  */
    cmpld cr6,10,11
    mr    12,4
    srdi  9,5,3		/* Number of full double words remaining.  */
    mtcrf 0x01,0
    mr    31,5
    beq   .L0

    subf  31,0,5
  /* Move 0-7 bytes as needed to get the destination doubleword aligned.  */
1:  bf    31,2f
    lbz   6,0(12)
    addi  12,12,1
    stb   6,0(3)
    addi  3,3,1
2:  bf    30,4f
    lhz   6,0(12)
    addi  12,12,2
    sth   6,0(3)
    addi  3,3,2
4:  bf    29,0f
    lwz   6,0(12)
    addi  12,12,4
    stw   6,0(3)
    addi  3,3,4
0:
    clrldi 10,12,61	/* check alignment of src again.  */
    srdi  9,31,3	/* Number of full double words remaining.  */

  /* Copy doublewords from source to destination, assuming the
     destination is aligned on a doubleword boundary.

     At this point we know there are at least 25 bytes left (32-7) to copy.
     The next step is to determine if the source is also doubleword aligned.
     If not branch to the unaligned move code at .L6. which uses
     a load, shift, store strategy.

     Otherwise source and destination are doubleword aligned, and we can
     the optimized doubleword copy loop.  */
.L0:
    clrldi  11,31,61
    mtcrf   0x01,9
    cmpldi  cr1,11,0
    bne-    cr6,.L6   /* If source is not DW aligned.  */

  /* Move doublewords where destination and source are DW aligned.
     Use a unrolled loop to copy 4 doubleword (32-bytes) per iteration.
     If the copy is not an exact multiple of 32 bytes, 1-3
     doublewords are copied as needed to set up the main loop.  After
     the main loop exits there may be a tail of 1-7 bytes. These byte are
     copied a word/halfword/byte at a time as needed to preserve alignment.  */

    srdi  8,31,5
    cmpldi	cr1,9,4
    cmpldi	cr6,11,0
    mr    11,12

    bf    30,1f
    ld    6,0(12)
    ld    7,8(12)
    addi  11,12,16
    mtctr 8
    std   6,0(3)
    std   7,8(3)
    addi  10,3,16
    bf    31,4f
    ld    0,16(12)
    std   0,16(3)
    blt   cr1,3f
    addi  11,12,24
    addi  10,3,24
    b     4f
    .align  4
1:
    mr    10,3
    mtctr 8
    bf    31,4f
    ld    6,0(12)
    addi  11,12,8
    std   6,0(3)
    addi  10,3,8

    .align  4
4:
    ld    6,0(11)
    ld    7,8(11)
    ld    8,16(11)
    ld    0,24(11)
    addi  11,11,32
2:
    std   6,0(10)
    std   7,8(10)
    std   8,16(10)
    std   0,24(10)
    addi  10,10,32
    bdnz  4b
3:

    rldicr 0,31,0,60
    mtcrf 0x01,31
    beq   cr6,0f
.L9:
    add   3,3,0
    add   12,12,0

/*  At this point we have a tail of 0-7 bytes and we know that the
    destination is double word aligned.  */
4:  bf    29,2f
    lwz   6,0(12)
    addi  12,12,4
    stw   6,0(3)
    addi  3,3,4
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
    ld 31,-8(1)
    ld 3,-16(1)
    blr

/* Copy up to 31 bytes.  This divided into two cases 0-8 bytes and 9-31
   bytes.  Each case is handled without loops, using binary (1,2,4,8)
   tests.

   In the short (0-8 byte) case no attempt is made to force alignment
   of either source or destination.  The hardware will handle the
   unaligned load/stores with small delays for crossing 32- 64-byte, and
   4096-byte boundaries. Since these short moves are unlikely to be
   unaligned or cross these boundaries, the overhead to force
   alignment is not justified.

   The longer (9-31 byte) move is more likely to cross 32- or 64-byte
   boundaries.  Since only loads are sensitive to the 32-/64-byte
   boundaries it is more important to align the source then the
   destination.  If the source is not already word aligned, we first
   move 1-3 bytes as needed.  Since we are only word aligned we don't
   use double word load/stores to insure that all loads are aligned.
   While the destination and stores may still be unaligned, this
   is only an issue for page (4096 byte boundary) crossing, which
   should be rare for these short moves.  The hardware handles this
   case automatically with a small delay.  */

    .align  4
.L2:
    mtcrf 0x01,5
    neg   8,4
    clrrdi	11,4,2
    andi. 0,8,3
    ble   cr6,.LE8	/* Handle moves of 0-8 bytes.  */
/* At least 9 bytes left.  Get the source word aligned.  */
    cmpldi	cr1,5,16
    mr    10,5
    mr    12,4
    cmpldi	cr6,0,2
    beq   .L3	/* If the source is already word aligned skip this.  */
/* Copy 1-3 bytes to get source address word aligned.  */
    lwz   6,0(11)
    subf  10,0,5
    add   12,4,0
    blt   cr6,5f
    srdi  7,6,16
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
    cmpldi	cr1,10,16
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
    ld 3,-16(1)
    blr
1:  /* Move 1 byte.  */
    bf    31,0f
    lbz   6,0(12)
    stb   6,0(3)
0:
  /* Return original dst pointer.  */
    ld    3,-16(1)
    blr

/* Special case to copy 0-8 bytes.  */
    .align  4
.LE8:
    mr    12,4
    bne   cr6,4f
/* Would have liked to use use ld/std here but the 630 processors are
   slow for load/store doubles that are not at least word aligned.
   Unaligned Load/Store word execute with only a 1 cycle penalty.  */
    lwz   6,0(4)
    lwz   7,4(4)
    stw   6,0(3)
    stw   7,4(3)
  /* Return original dst pointer.  */
    ld    3,-16(1)
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
    ld 3,-16(1)
    blr
    .align  4
5:
    bf    31,0f
    lbz   6,4(4)
    stb   6,4(3)
    .align  4
0:
  /* Return original dst pointer.  */
    ld    3,-16(1)
    blr

    .align  4
.L6:

  /* Copy doublewords where the destination is aligned but the source is
     not.  Use aligned doubleword loads from the source, shifted to realign
     the data, to allow aligned destination stores.  */
    addi    11,9,-1  /* loop DW count is one less than total */
    subf    5,10,12
    sldi    10,10,3
    mr      4,3
    srdi    8,11,2   /* calculate the 32 byte loop count */
    ld      6,0(5)
    mtcrf   0x01,11
    cmpldi  cr6,9,4
    mtctr   8
    ld      7,8(5)
    subfic  9,10,64
    bf      30,1f

    /* there are at least two DWs to copy */
#ifdef __LITTLE_ENDIAN__
    srd     0,6,10
    sld     8,7,9
#else
    sld     0,6,10
    srd     8,7,9
#endif
    or      0,0,8
    ld      6,16(5)
    std     0,0(4)
#ifdef __LITTLE_ENDIAN__
    srd     0,7,10
    sld     8,6,9
#else
    sld     0,7,10
    srd     8,6,9
#endif
    or      0,0,8
    ld      7,24(5)
    std     0,8(4)
    addi    4,4,16
    addi    5,5,32
    blt     cr6,8f  /* if total DWs = 3, then bypass loop */
    bf      31,4f
    /* there is a third DW to copy */
#ifdef __LITTLE_ENDIAN__
    srd     0,6,10
    sld     8,7,9
#else
    sld     0,6,10
    srd     8,7,9
#endif
    or      0,0,8
    std     0,0(4)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    addi    4,4,8
    beq     cr6,8f  /* if total DWs = 4, then bypass loop */
    b       4f
    .align 4
1:
#ifdef __LITTLE_ENDIAN__
    srd     0,6,10
    sld     8,7,9
#else
    sld     0,6,10
    srd     8,7,9
#endif
    addi    5,5,16
    or      0,0,8
    bf      31,4f
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    std     0,0(4)
    addi    4,4,8
    .align 4
/* copy 32 bytes at a time */
4:
#ifdef __LITTLE_ENDIAN__
    srd   0,6,10
    sld   8,7,9
#else
    sld   0,6,10
    srd   8,7,9
#endif
    or    0,0,8
    ld    6,0(5)
    std   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srd   0,7,10
    sld   8,6,9
#else
    sld   0,7,10
    srd   8,6,9
#endif
    or    0,0,8
    ld    7,8(5)
    std   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srd   0,6,10
    sld   8,7,9
#else
    sld   0,6,10
    srd   8,7,9
#endif
    or    0,0,8
    ld    6,16(5)
    std   0,16(4)
#ifdef __LITTLE_ENDIAN__
    srd   0,7,10
    sld   8,6,9
#else
    sld   0,7,10
    srd   8,6,9
#endif
    or    0,0,8
    ld    7,24(5)
    std   0,24(4)
    addi  5,5,32
    addi  4,4,32
    bdnz+ 4b
    .align 4
8:
    /* calculate and store the final DW */
#ifdef __LITTLE_ENDIAN__
    srd   0,6,10
    sld   8,7,9
#else
    sld   0,6,10
    srd   8,7,9
#endif
    or    0,0,8
    std   0,0(4)
3:
    rldicr 0,31,0,60
    mtcrf 0x01,31
    bne   cr1,.L9	/* If the tail is 0 bytes we are done!  */
  /* Return original dst pointer.  */
    ld 31,-8(1)
    ld 3,-16(1)
    blr
END_GEN_TB (MEMCPY,TB_TOCLESS)
libc_hidden_builtin_def (memcpy)
