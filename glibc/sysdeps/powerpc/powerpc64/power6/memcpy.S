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
   Each case has a optimized unrolled loop.

   For POWER6 unaligned loads will take a 20+ cycle hiccup for any
   L1 cache miss that crosses a 32- or 128-byte boundary.  Store
   is more forgiving and does not take a hiccup until page or
   segment boundaries.  So we require doubleword alignment for
   the source but may take a risk and only require word alignment
   for the destination.  */

#ifndef MEMCPY
# define MEMCPY memcpy
#endif
	.machine	"power6"
ENTRY_TOCLESS (MEMCPY, 7)
	CALL_MCOUNT 3

    cmpldi cr1,5,31
    neg   0,3
    std   3,-16(1)
    std   31,-8(1)
    andi. 11,3,7	/* check alignment of dst.  */
    clrldi 0,0,61	/* Number of bytes until the 1st doubleword of dst.  */
    clrldi 10,4,61	/* check alignment of src.  */
    cmpldi cr6,5,8
    ble-  cr1,.L2	/* If move < 32 bytes use short move code.  */
    mtcrf 0x01,0
    cmpld cr6,10,11
    srdi  9,5,3		/* Number of full double words remaining.  */
    beq   .L0

    subf  5,0,5
  /* Move 0-7 bytes as needed to get the destination doubleword aligned.
     Duplicate some code to maximize fall-through and minimize agen delays.  */
1:  bf    31,2f
    lbz   6,0(4)
    stb   6,0(3)
    bf    30,5f
    lhz   6,1(4)
    sth   6,1(3)
    bf    29,0f
    lwz   6,3(4)
    stw   6,3(3)
    b     0f
5:
    bf    29,0f
    lwz   6,1(4)
    stw   6,1(3)
    b     0f

2:  bf    30,4f
    lhz   6,0(4)
    sth   6,0(3)
    bf    29,0f
    lwz   6,2(4)
    stw   6,2(3)
    b     0f

4:  bf    29,0f
    lwz   6,0(4)
    stw   6,0(3)
0:
/* Add the number of bytes until the 1st doubleword of dst to src and dst.  */
    add   4,4,0
    add   3,3,0

    clrldi 10,4,61	/* check alignment of src again.  */
    srdi  9,5,3	/* Number of full double words remaining.  */

  /* Copy doublewords from source to destination, assuming the
     destination is aligned on a doubleword boundary.

     At this point we know there are at least 25 bytes left (32-7) to copy.
     The next step is to determine if the source is also doubleword aligned.
     If not branch to the unaligned move code at .L6. which uses
     a load, shift, store strategy.

     Otherwise source and destination are doubleword aligned, and we can
     the optimized doubleword copy loop.  */
    .align  4
.L0:
    clrldi  11,5,61
    andi.   0,5,0x78
    srdi    12,5,7	/* Number of 128-byte blocks to move.  */
    cmpldi  cr1,11,0	/* If the tail is 0 bytes  */
    bne-    cr6,.L6     /* If source is not DW aligned.  */

  /* Move doublewords where destination and source are DW aligned.
     Use a unrolled loop to copy 16 doublewords (128-bytes) per iteration.
     If the copy is not an exact multiple of 128 bytes, 1-15
     doublewords are copied as needed to set up the main loop.  After
     the main loop exits there may be a tail of 1-7 bytes. These byte
     are copied a word/halfword/byte at a time as needed to preserve
     alignment.

     For POWER6 the L1 is store-through and the L2 is store-in.  The
     L2 is clocked at half CPU clock so we can store 16 bytes every
     other cycle.  POWER6 also has a load/store bypass so we can do
     load, load, store, store every 2 cycles.

     The following code is sensitive to cache line alignment.  Do not
     make any change with out first making sure they don't result in
     splitting ld/std pairs across a cache line.  */

    mtcrf 0x02,5
    mtcrf 0x01,5
    cmpldi  cr5,12,1
    beq   L(das_loop)

    bf    25,4f
    .align  3
    ld    6,0(4)
    ld    7,8(4)
    mr    11,4
    mr    10,3
    std   6,0(3)
    std   7,8(3)
    ld    6,16(4)
    ld    7,24(4)
    std   6,16(3)
    std   7,24(3)
    ld    6,0+32(4)
    ld    7,8+32(4)
    addi  4,4,64
    addi  3,3,64
    std   6,0+32(10)
    std   7,8+32(10)
    ld    6,16+32(11)
    ld    7,24+32(11)
    std   6,16+32(10)
    std   7,24+32(10)
4:
    mr    10,3
    bf    26,2f
    ld    6,0(4)
    ld    7,8(4)
    mr    11,4
    nop
    std   6,0(3)
    std   7,8(3)
    ld    6,16(4)
    ld    7,24(4)
    addi  4,4,32
    std   6,16(3)
    std   7,24(3)
    addi  3,3,32
6:
    nop
    bf    27,5f
    ld    6,0+32(11)
    ld    7,8+32(11)
    addi  4,4,16
    addi  3,3,16
    std   6,0+32(10)
    std   7,8+32(10)
    bf    28,L(das_loop_s)
    ld    0,16+32(11)
    addi  4,4,8
    addi  3,3,8
    std   0,16+32(10)
    blt   cr5,L(das_tail)
    b     L(das_loop)
    .align  3
5:
    nop
    bf    28,L(das_loop_s)
    ld    6,32(11)
    addi  4,4,8
    addi  3,3,8
    std   6,32(10)
    blt   cr5,L(das_tail)
    b     L(das_loop)
    .align  3
2:
    mr    11,4
    bf    27,1f
    ld    6,0(4)
    ld    7,8(4)
    addi  4,4,16
    addi  3,3,16
    std   6,0(10)
    std   7,8(10)
    bf    28,L(das_loop_s)
    ld    0,16(11)
    addi  4,11,24
    addi  3,10,24
    std   0,16(10)
    blt   cr5,L(das_tail)
    b     L(das_loop)
    .align  3
1:
    nop
    bf    28,L(das_loop_s)
    ld    6,0(4)
    addi  4,4,8
    addi  3,3,8
    std   6,0(10)
L(das_loop_s):
    nop
    blt   cr5,L(das_tail)
    .align  4
L(das_loop):
    ld    6,0(4)
    ld    7,8(4)
    mr    10,3
    mr    11,4
    std   6,0(3)
    std   7,8(3)
    addi  12,12,-1
    nop
    ld    8,16(4)
    ld    0,24(4)
    std   8,16(3)
    std   0,24(3)

    ld    6,0+32(4)
    ld    7,8+32(4)
    std   6,0+32(3)
    std   7,8+32(3)
    ld    8,16+32(4)
    ld    0,24+32(4)
    std   8,16+32(3)
    std   0,24+32(3)

    ld    6,0+64(11)
    ld    7,8+64(11)
    std   6,0+64(10)
    std   7,8+64(10)
    ld    8,16+64(11)
    ld    0,24+64(11)
    std   8,16+64(10)
    std   0,24+64(10)

    ld    6,0+96(11)
    ld    7,8+96(11)
    addi  4,4,128
    addi  3,3,128
    std   6,0+96(10)
    std   7,8+96(10)
    ld    8,16+96(11)
    ld    0,24+96(11)
    std   8,16+96(10)
    std   0,24+96(10)
    ble   cr5,L(das_loop_e)

    mtctr   12
    .align  4
L(das_loop2):
    ld    6,0(4)
    ld    7,8(4)
    mr    10,3
    mr    11,4
    std   6,0(3)
    std   7,8(3)
    ld    8,16(4)
    ld    0,24(4)
    std   8,16(3)
    std   0,24(3)

    ld    6,0+32(4)
    ld    7,8+32(4)
    std   6,0+32(3)
    std   7,8+32(3)
    ld    8,16+32(4)
    ld    0,24+32(4)
    std   8,16+32(3)
    std   0,24+32(3)

    ld    6,0+64(11)
    ld    7,8+64(11)
    std   6,0+64(10)
    std   7,8+64(10)
    ld    8,16+64(11)
    ld    0,24+64(11)
    std   8,16+64(10)
    std   0,24+64(10)

    ld    6,0+96(11)
    ld    7,8+96(11)
    addi  4,4,128
    addi  3,3,128
    std   6,0+96(10)
    std   7,8+96(10)
    ld    8,16+96(11)
    ld    0,24+96(11)
    std   8,16+96(10)
    std   0,24+96(10)
    bdnz  L(das_loop2)
L(das_loop_e):
/* Check of a 1-7 byte tail, return if none.  */
    bne   cr1,L(das_tail2)
/* Return original dst pointer.  */
    ld 3,-16(1)
    blr
    .align  4
L(das_tail):
    beq   cr1,0f

L(das_tail2):
/*  At this point we have a tail of 0-7 bytes and we know that the
    destination is double word aligned.  */
4:  bf    29,2f
    lwz   6,0(4)
    stw   6,0(3)
    bf    30,5f
    lhz   6,4(4)
    sth   6,4(3)
    bf    31,0f
    lbz   6,6(4)
    stb   6,6(3)
    b     0f
5:  bf    31,0f
    lbz   6,4(4)
    stb   6,4(3)
    b     0f

2:  bf    30,1f
    lhz   6,0(4)
    sth   6,0(3)
    bf    31,0f
    lbz   6,2(4)
    stb   6,2(3)
    b     0f

1:  bf    31,0f
    lbz   6,0(4)
    stb   6,0(3)
0:
  /* Return original dst pointer.  */
    ld 3,-16(1)
    blr

/* Copy up to 31 bytes.  This divided into two cases 0-8 bytes and 9-31
   bytes.  Each case is handled without loops, using binary (1,2,4,8)
   tests.

   In the short (0-8 byte) case no attempt is made to force alignment
   of either source or destination.  The hardware will handle the
   unaligned load/stores with small delays for crossing 32- 128-byte,
   and 4096-byte boundaries. Since these short moves are unlikely to be
   unaligned or cross these boundaries, the overhead to force
   alignment is not justified.

   The longer (9-31 byte) move is more likely to cross 32- or 128-byte
   boundaries.  Since only loads are sensitive to the 32-/128-byte
   boundaries it is more important to align the source then the
   destination.  If the source is not already word aligned, we first
   move 1-3 bytes as needed.  Since we are only word aligned we don't
   use double word load/stores to insure that all loads are aligned.
   While the destination and stores may still be unaligned, this
   is only an issue for page (4096 byte boundary) crossing, which
   should be rare for these short moves.  The hardware handles this
   case automatically with a small (~20 cycle) delay.  */
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
    beq   L(dus_tail)	/* If the source is already word aligned skip this.  */
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
L(dus_tail):
/* At least 6 bytes left and the source is word aligned.  This allows
   some speculative loads up front.  */
/* We need to special case the fall-through because the biggest delays
   are due to address computation not being ready in time for the
   AGEN.  */
    lwz   6,0(12)
    lwz   7,4(12)
    blt   cr1,L(dus_tail8)
    cmpldi	cr0,10,24
L(dus_tail16): /* Move 16 bytes.  */
    stw   6,0(3)
    stw   7,4(3)
    lwz   6,8(12)
    lwz   7,12(12)
    stw   6,8(3)
    stw   7,12(3)
/* Move 8 bytes more.  */
    bf    28,L(dus_tail16p8)
    cmpldi	cr1,10,28
    lwz   6,16(12)
    lwz   7,20(12)
    stw   6,16(3)
    stw   7,20(3)
/* Move 4 bytes more.  */
    bf    29,L(dus_tail16p4)
    lwz   6,24(12)
    stw   6,24(3)
    addi  12,12,28
    addi  3,3,28
    bgt   cr1,L(dus_tail2)
 /* exactly 28 bytes.  Return original dst pointer and exit.  */
    ld    3,-16(1)
    blr
    .align  4
L(dus_tail16p8):  /* less than 8 bytes left.  */
    beq   cr1,L(dus_tailX) /* exactly 16 bytes, early exit.  */
    cmpldi	cr1,10,20
    bf    29,L(dus_tail16p2)
/* Move 4 bytes more.  */
    lwz   6,16(12)
    stw   6,16(3)
    addi  12,12,20
    addi  3,3,20
    bgt   cr1,L(dus_tail2)
 /* exactly 20 bytes.  Return original dst pointer and exit.  */
    ld    3,-16(1)
    blr
    .align  4
L(dus_tail16p4):  /* less than 4 bytes left.  */
    addi  12,12,24
    addi  3,3,24
    bgt   cr0,L(dus_tail2)
 /* exactly 24 bytes.  Return original dst pointer and exit.  */
    ld    3,-16(1)
    blr
    .align  4
L(dus_tail16p2):  /* 16 bytes moved, less than 4 bytes left.  */
    addi  12,12,16
    addi  3,3,16
    b     L(dus_tail2)

    .align  4
L(dus_tail8):  /* Move 8 bytes.  */
/*  r6, r7 already loaded speculatively.  */
    cmpldi	cr1,10,8
    cmpldi	cr0,10,12
    bf    28,L(dus_tail4)
    .align  2
    stw   6,0(3)
    stw   7,4(3)
/* Move 4 bytes more.  */
    bf    29,L(dus_tail8p4)
    lwz   6,8(12)
    stw   6,8(3)
    addi  12,12,12
    addi  3,3,12
    bgt   cr0,L(dus_tail2)
 /* exactly 12 bytes.  Return original dst pointer and exit.  */
    ld    3,-16(1)
    blr
    .align  4
L(dus_tail8p4):  /* less than 4 bytes left.  */
    addi  12,12,8
    addi  3,3,8
    bgt   cr1,L(dus_tail2)
 /* exactly 8 bytes.  Return original dst pointer and exit.  */
    ld    3,-16(1)
    blr

    .align  4
L(dus_tail4):  /* Move 4 bytes.  */
/*  r6 already loaded speculatively.  If we are here we know there is
    more than 4 bytes left.  So there is no need to test.  */
    addi  12,12,4
    stw   6,0(3)
    addi  3,3,4
L(dus_tail2):  /* Move 2-3 bytes.  */
    bf    30,L(dus_tail1)
    lhz   6,0(12)
    sth   6,0(3)
    bf    31,L(dus_tailX)
    lbz   7,2(12)
    stb   7,2(3)
    ld 3,-16(1)
    blr
L(dus_tail1):  /* Move 1 byte.  */
    bf    31,L(dus_tailX)
    lbz   6,0(12)
    stb   6,0(3)
L(dus_tailX):
  /* Return original dst pointer.  */
    ld    3,-16(1)
    blr

/* Special case to copy 0-8 bytes.  */
    .align  4
.LE8:
    mr    12,4
    bne   cr6,L(dus_4)
/* Exactly 8 bytes.  We may cross a 32-/128-byte boundary and take a ~20
   cycle delay.  This case should be rare and any attempt to avoid this
   would take most of 20 cycles any way.  */
    ld   6,0(4)
    std   6,0(3)
  /* Return original dst pointer.  */
    ld    3,-16(1)
    blr
    .align  4
L(dus_4):
    bf    29,L(dus_tail2)
    lwz   6,0(4)
    stw   6,0(3)
    bf    30,L(dus_5)
    lhz   7,4(4)
    sth   7,4(3)
    bf    31,L(dus_0)
    lbz   8,6(4)
    stb   8,6(3)
    ld 3,-16(1)
    blr
    .align  4
L(dus_5):
    bf    31,L(dus_0)
    lbz   6,4(4)
    stb   6,4(3)
L(dus_0):
  /* Return original dst pointer.  */
    ld    3,-16(1)
    blr

    .align  4
.L6:
    cfi_offset(31,-8)
    mr    12,4
    mr    31,5
  /* Copy doublewords where the destination is aligned but the source is
     not.  Use aligned doubleword loads from the source, shifted to realign
     the data, to allow aligned destination stores.  */
    addi    11,9,-1  /* loop DW count is one less than total */
    subf    5,10,12  /* Move source addr to previous full double word.  */
    cmpldi  cr5, 10, 2
    cmpldi  cr0, 10, 4
    mr      4,3
    srdi    8,11,2   /* calculate the 32 byte loop count */
    ld      6,0(5)   /* pre load 1st full doubleword.  */
    mtcrf   0x01,11
    cmpldi  cr6,9,4
    mtctr   8
    ld      7,8(5)   /* pre load 2nd full doubleword.  */
    bge     cr0, L(du4_do)
    blt     cr5, L(du1_do)
    beq     cr5, L(du2_do)
    b       L(du3_do)

    .align 4
L(du1_do):
    bf      30,L(du1_1dw)

    /* there are at least two DWs to copy */
    /* FIXME: can combine last shift and "or" into "rldimi" */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 8
    sldi     8,7, 64-8
#else
    sldi     0,6, 8
    srdi     8,7, 64-8
#endif
    or      0,0,8
    ld      6,16(5)
    std     0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi     0,7, 8
    sldi     8,6, 64-8
#else
    sldi     0,7, 8
    srdi     8,6, 64-8
#endif
    or      0,0,8
    ld      7,24(5)
    std     0,8(4)
    addi    4,4,16
    addi    5,5,32
    blt     cr6,L(du1_fini)  /* if total DWs = 3, then bypass loop */
    bf      31,L(du1_loop)
    /* there is a third DW to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 8
    sldi     8,7, 64-8
#else
    sldi     0,6, 8
    srdi     8,7, 64-8
#endif
    or      0,0,8
    std     0,0(4)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    addi    4,4,8
    beq     cr6,L(du1_fini)  /* if total DWs = 4, then bypass loop */
    b       L(du1_loop)
    .align 4
L(du1_1dw):
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 8
    sldi     8,7, 64-8
#else
    sldi     0,6, 8
    srdi     8,7, 64-8
#endif
    addi    5,5,16
    or      0,0,8
    bf      31,L(du1_loop)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    std     0,0(4)
    addi    4,4,8
    .align 4
/* copy 32 bytes at a time */
L(du1_loop):
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 8
    sldi   8,7, 64-8
#else
    sldi   0,6, 8
    srdi   8,7, 64-8
#endif
    or    0,0,8
    ld    6,0(5)
    std   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 8
    sldi   8,6, 64-8
#else
    sldi   0,7, 8
    srdi   8,6, 64-8
#endif
    or    0,0,8
    ld    7,8(5)
    std   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 8
    sldi   8,7, 64-8
#else
    sldi   0,6, 8
    srdi   8,7, 64-8
#endif
    or    0,0,8
    ld    6,16(5)
    std   0,16(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 8
    sldi   8,6, 64-8
#else
    sldi   0,7, 8
    srdi   8,6, 64-8
#endif
    or    0,0,8
    ld    7,24(5)
    std   0,24(4)
    addi  5,5,32
    addi  4,4,32
    bdnz+ L(du1_loop)
    .align 4
L(du1_fini):
    /* calculate and store the final DW */
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 8
    sldi   8,7, 64-8
#else
    sldi   0,6, 8
    srdi   8,7, 64-8
#endif
    or    0,0,8
    std   0,0(4)
    b     L(du_done)

    .align 4
L(du2_do):
    bf      30,L(du2_1dw)

    /* there are at least two DWs to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 16
    sldi     8,7, 64-16
#else
    sldi     0,6, 16
    srdi     8,7, 64-16
#endif
    or      0,0,8
    ld      6,16(5)
    std     0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi     0,7, 16
    sldi     8,6, 64-16
#else
    sldi     0,7, 16
    srdi     8,6, 64-16
#endif
    or      0,0,8
    ld      7,24(5)
    std     0,8(4)
    addi    4,4,16
    addi    5,5,32
    blt     cr6,L(du2_fini)  /* if total DWs = 3, then bypass loop */
    bf      31,L(du2_loop)
    /* there is a third DW to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 16
    sldi     8,7, 64-16
#else
    sldi     0,6, 16
    srdi     8,7, 64-16
#endif
    or      0,0,8
    std     0,0(4)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    addi    4,4,8
    beq     cr6,L(du2_fini)  /* if total DWs = 4, then bypass loop */
    b       L(du2_loop)
    .align 4
L(du2_1dw):
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 16
    sldi     8,7, 64-16
#else
    sldi     0,6, 16
    srdi     8,7, 64-16
#endif
    addi    5,5,16
    or      0,0,8
    bf      31,L(du2_loop)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    std     0,0(4)
    addi    4,4,8
    .align 4
/* copy 32 bytes at a time */
L(du2_loop):
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 16
    sldi   8,7, 64-16
#else
    sldi   0,6, 16
    srdi   8,7, 64-16
#endif
    or    0,0,8
    ld    6,0(5)
    std   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 16
    sldi   8,6, 64-16
#else
    sldi   0,7, 16
    srdi   8,6, 64-16
#endif
    or    0,0,8
    ld    7,8(5)
    std   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 16
    sldi   8,7, 64-16
#else
    sldi   0,6, 16
    srdi   8,7, 64-16
#endif
    or    0,0,8
    ld    6,16(5)
    std   0,16(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 16
    sldi   8,6, 64-16
#else
    sldi   0,7, 16
    srdi   8,6, 64-16
#endif
    or    0,0,8
    ld    7,24(5)
    std   0,24(4)
    addi  5,5,32
    addi  4,4,32
    bdnz+ L(du2_loop)
    .align 4
L(du2_fini):
    /* calculate and store the final DW */
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 16
    sldi   8,7, 64-16
#else
    sldi   0,6, 16
    srdi   8,7, 64-16
#endif
    or    0,0,8
    std   0,0(4)
    b     L(du_done)

    .align 4
L(du3_do):
    bf      30,L(du3_1dw)

    /* there are at least two DWs to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 24
    sldi     8,7, 64-24
#else
    sldi     0,6, 24
    srdi     8,7, 64-24
#endif
    or      0,0,8
    ld      6,16(5)
    std     0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi     0,7, 24
    sldi     8,6, 64-24
#else
    sldi     0,7, 24
    srdi     8,6, 64-24
#endif
    or      0,0,8
    ld      7,24(5)
    std     0,8(4)
    addi    4,4,16
    addi    5,5,32
    blt     cr6,L(du3_fini)  /* if total DWs = 3, then bypass loop */
    bf      31,L(du3_loop)
    /* there is a third DW to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 24
    sldi     8,7, 64-24
#else
    sldi     0,6, 24
    srdi     8,7, 64-24
#endif
    or      0,0,8
    std     0,0(4)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    addi    4,4,8
    beq     cr6,L(du3_fini)  /* if total DWs = 4, then bypass loop */
    b       L(du3_loop)
    .align 4
L(du3_1dw):
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 24
    sldi     8,7, 64-24
#else
    sldi     0,6, 24
    srdi     8,7, 64-24
#endif
    addi    5,5,16
    or      0,0,8
    bf      31,L(du3_loop)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    std     0,0(4)
    addi    4,4,8
    .align 4
/* copy 32 bytes at a time */
L(du3_loop):
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 24
    sldi   8,7, 64-24
#else
    sldi   0,6, 24
    srdi   8,7, 64-24
#endif
    or    0,0,8
    ld    6,0(5)
    std   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 24
    sldi   8,6, 64-24
#else
    sldi   0,7, 24
    srdi   8,6, 64-24
#endif
    or    0,0,8
    ld    7,8(5)
    std   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 24
    sldi   8,7, 64-24
#else
    sldi   0,6, 24
    srdi   8,7, 64-24
#endif
    or    0,0,8
    ld    6,16(5)
    std   0,16(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 24
    sldi   8,6, 64-24
#else
    sldi   0,7, 24
    srdi   8,6, 64-24
#endif
    or    0,0,8
    ld    7,24(5)
    std   0,24(4)
    addi  5,5,32
    addi  4,4,32
    bdnz+ L(du3_loop)
    .align 4
L(du3_fini):
    /* calculate and store the final DW */
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 24
    sldi   8,7, 64-24
#else
    sldi   0,6, 24
    srdi   8,7, 64-24
#endif
    or    0,0,8
    std   0,0(4)
    b     L(du_done)

    .align 4
L(du4_do):
    cmpldi  cr5, 10, 6
    beq     cr0, L(du4_dox)
    blt     cr5, L(du5_do)
    beq     cr5, L(du6_do)
    b       L(du7_do)
L(du4_dox):
    bf      30,L(du4_1dw)

    /* there are at least two DWs to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 32
    sldi     8,7, 64-32
#else
    sldi     0,6, 32
    srdi     8,7, 64-32
#endif
    or      0,0,8
    ld      6,16(5)
    std     0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi     0,7, 32
    sldi     8,6, 64-32
#else
    sldi     0,7, 32
    srdi     8,6, 64-32
#endif
    or      0,0,8
    ld      7,24(5)
    std     0,8(4)
    addi    4,4,16
    addi    5,5,32
    blt     cr6,L(du4_fini)  /* if total DWs = 3, then bypass loop */
    bf      31,L(du4_loop)
    /* there is a third DW to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 32
    sldi     8,7, 64-32
#else
    sldi     0,6, 32
    srdi     8,7, 64-32
#endif
    or      0,0,8
    std     0,0(4)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    addi    4,4,8
    beq     cr6,L(du4_fini)  /* if total DWs = 4, then bypass loop */
    b       L(du4_loop)
    .align 4
L(du4_1dw):
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 32
    sldi     8,7, 64-32
#else
    sldi     0,6, 32
    srdi     8,7, 64-32
#endif
    addi    5,5,16
    or      0,0,8
    bf      31,L(du4_loop)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    std     0,0(4)
    addi    4,4,8
    .align 4
/* copy 32 bytes at a time */
L(du4_loop):
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 32
    sldi   8,7, 64-32
#else
    sldi   0,6, 32
    srdi   8,7, 64-32
#endif
    or    0,0,8
    ld    6,0(5)
    std   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 32
    sldi   8,6, 64-32
#else
    sldi   0,7, 32
    srdi   8,6, 64-32
#endif
    or    0,0,8
    ld    7,8(5)
    std   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 32
    sldi   8,7, 64-32
#else
    sldi   0,6, 32
    srdi   8,7, 64-32
#endif
    or    0,0,8
    ld    6,16(5)
    std   0,16(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 32
    sldi   8,6, 64-32
#else
    sldi   0,7, 32
    srdi   8,6, 64-32
#endif
    or    0,0,8
    ld    7,24(5)
    std   0,24(4)
    addi  5,5,32
    addi  4,4,32
    bdnz+ L(du4_loop)
    .align 4
L(du4_fini):
    /* calculate and store the final DW */
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 32
    sldi   8,7, 64-32
#else
    sldi   0,6, 32
    srdi   8,7, 64-32
#endif
    or    0,0,8
    std   0,0(4)
    b     L(du_done)

    .align 4
L(du5_do):
    bf      30,L(du5_1dw)

    /* there are at least two DWs to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 40
    sldi     8,7, 64-40
#else
    sldi     0,6, 40
    srdi     8,7, 64-40
#endif
    or      0,0,8
    ld      6,16(5)
    std     0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi     0,7, 40
    sldi     8,6, 64-40
#else
    sldi     0,7, 40
    srdi     8,6, 64-40
#endif
    or      0,0,8
    ld      7,24(5)
    std     0,8(4)
    addi    4,4,16
    addi    5,5,32
    blt     cr6,L(du5_fini)  /* if total DWs = 3, then bypass loop */
    bf      31,L(du5_loop)
    /* there is a third DW to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 40
    sldi     8,7, 64-40
#else
    sldi     0,6, 40
    srdi     8,7, 64-40
#endif
    or      0,0,8
    std     0,0(4)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    addi    4,4,8
    beq     cr6,L(du5_fini)  /* if total DWs = 4, then bypass loop */
    b       L(du5_loop)
    .align 4
L(du5_1dw):
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 40
    sldi     8,7, 64-40
#else
    sldi     0,6, 40
    srdi     8,7, 64-40
#endif
    addi    5,5,16
    or      0,0,8
    bf      31,L(du5_loop)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    std     0,0(4)
    addi    4,4,8
    .align 4
/* copy 32 bytes at a time */
L(du5_loop):
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 40
    sldi   8,7, 64-40
#else
    sldi   0,6, 40
    srdi   8,7, 64-40
#endif
    or    0,0,8
    ld    6,0(5)
    std   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 40
    sldi   8,6, 64-40
#else
    sldi   0,7, 40
    srdi   8,6, 64-40
#endif
    or    0,0,8
    ld    7,8(5)
    std   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 40
    sldi   8,7, 64-40
#else
    sldi   0,6, 40
    srdi   8,7, 64-40
#endif
    or    0,0,8
    ld    6,16(5)
    std   0,16(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 40
    sldi   8,6, 64-40
#else
    sldi   0,7, 40
    srdi   8,6, 64-40
#endif
    or    0,0,8
    ld    7,24(5)
    std   0,24(4)
    addi  5,5,32
    addi  4,4,32
    bdnz+ L(du5_loop)
    .align 4
L(du5_fini):
    /* calculate and store the final DW */
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 40
    sldi   8,7, 64-40
#else
    sldi   0,6, 40
    srdi   8,7, 64-40
#endif
    or    0,0,8
    std   0,0(4)
    b     L(du_done)

    .align 4
L(du6_do):
    bf      30,L(du6_1dw)

    /* there are at least two DWs to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 48
    sldi     8,7, 64-48
#else
    sldi     0,6, 48
    srdi     8,7, 64-48
#endif
    or      0,0,8
    ld      6,16(5)
    std     0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi     0,7, 48
    sldi     8,6, 64-48
#else
    sldi     0,7, 48
    srdi     8,6, 64-48
#endif
    or      0,0,8
    ld      7,24(5)
    std     0,8(4)
    addi    4,4,16
    addi    5,5,32
    blt     cr6,L(du6_fini)  /* if total DWs = 3, then bypass loop */
    bf      31,L(du6_loop)
    /* there is a third DW to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 48
    sldi     8,7, 64-48
#else
    sldi     0,6, 48
    srdi     8,7, 64-48
#endif
    or      0,0,8
    std     0,0(4)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    addi    4,4,8
    beq     cr6,L(du6_fini)  /* if total DWs = 4, then bypass loop */
    b       L(du6_loop)
    .align 4
L(du6_1dw):
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 48
    sldi     8,7, 64-48
#else
    sldi     0,6, 48
    srdi     8,7, 64-48
#endif
    addi    5,5,16
    or      0,0,8
    bf      31,L(du6_loop)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    std     0,0(4)
    addi    4,4,8
    .align 4
/* copy 32 bytes at a time */
L(du6_loop):
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 48
    sldi   8,7, 64-48
#else
    sldi   0,6, 48
    srdi   8,7, 64-48
#endif
    or    0,0,8
    ld    6,0(5)
    std   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 48
    sldi   8,6, 64-48
#else
    sldi   0,7, 48
    srdi   8,6, 64-48
#endif
    or    0,0,8
    ld    7,8(5)
    std   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 48
    sldi   8,7, 64-48
#else
    sldi   0,6, 48
    srdi   8,7, 64-48
#endif
    or    0,0,8
    ld    6,16(5)
    std   0,16(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 48
    sldi   8,6, 64-48
#else
    sldi   0,7, 48
    srdi   8,6, 64-48
#endif
    or    0,0,8
    ld    7,24(5)
    std   0,24(4)
    addi  5,5,32
    addi  4,4,32
    bdnz+ L(du6_loop)
    .align 4
L(du6_fini):
    /* calculate and store the final DW */
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 48
    sldi   8,7, 64-48
#else
    sldi   0,6, 48
    srdi   8,7, 64-48
#endif
    or    0,0,8
    std   0,0(4)
    b     L(du_done)

    .align 4
L(du7_do):
    bf      30,L(du7_1dw)

    /* there are at least two DWs to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 56
    sldi     8,7, 64-56
#else
    sldi     0,6, 56
    srdi     8,7, 64-56
#endif
    or      0,0,8
    ld      6,16(5)
    std     0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi     0,7, 56
    sldi     8,6, 64-56
#else
    sldi     0,7, 56
    srdi     8,6, 64-56
#endif
    or      0,0,8
    ld      7,24(5)
    std     0,8(4)
    addi    4,4,16
    addi    5,5,32
    blt     cr6,L(du7_fini)  /* if total DWs = 3, then bypass loop */
    bf      31,L(du7_loop)
    /* there is a third DW to copy */
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 56
    sldi     8,7, 64-56
#else
    sldi     0,6, 56
    srdi     8,7, 64-56
#endif
    or      0,0,8
    std     0,0(4)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    addi    4,4,8
    beq     cr6,L(du7_fini)  /* if total DWs = 4, then bypass loop */
    b       L(du7_loop)
    .align 4
L(du7_1dw):
#ifdef __LITTLE_ENDIAN__
    srdi     0,6, 56
    sldi     8,7, 64-56
#else
    sldi     0,6, 56
    srdi     8,7, 64-56
#endif
    addi    5,5,16
    or      0,0,8
    bf      31,L(du7_loop)
    mr      6,7
    ld      7,0(5)
    addi    5,5,8
    std     0,0(4)
    addi    4,4,8
    .align 4
/* copy 32 bytes at a time */
L(du7_loop):
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 56
    sldi   8,7, 64-56
#else
    sldi   0,6, 56
    srdi   8,7, 64-56
#endif
    or    0,0,8
    ld    6,0(5)
    std   0,0(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 56
    sldi   8,6, 64-56
#else
    sldi   0,7, 56
    srdi   8,6, 64-56
#endif
    or    0,0,8
    ld    7,8(5)
    std   0,8(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 56
    sldi   8,7, 64-56
#else
    sldi   0,6, 56
    srdi   8,7, 64-56
#endif
    or    0,0,8
    ld    6,16(5)
    std   0,16(4)
#ifdef __LITTLE_ENDIAN__
    srdi   0,7, 56
    sldi   8,6, 64-56
#else
    sldi   0,7, 56
    srdi   8,6, 64-56
#endif
    or    0,0,8
    ld    7,24(5)
    std   0,24(4)
    addi  5,5,32
    addi  4,4,32
    bdnz+ L(du7_loop)
    .align 4
L(du7_fini):
    /* calculate and store the final DW */
#ifdef __LITTLE_ENDIAN__
    srdi   0,6, 56
    sldi   8,7, 64-56
#else
    sldi   0,6, 56
    srdi   8,7, 64-56
#endif
    or    0,0,8
    std   0,0(4)
    b     L(du_done)

    .align 4
L(du_done):
    rldicr 0,31,0,60
    mtcrf 0x01,31
    beq   cr1,0f	/* If the tail is 0 bytes we are done!  */

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
END_GEN_TB (MEMCPY,TB_TOCLESS)
libc_hidden_builtin_def (memcpy)
