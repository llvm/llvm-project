/* Optimized memcpy implementation for PowerPC32 on POWER6.
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

	.machine power6
EALIGN (memcpy, 5, 0)
	CALL_MCOUNT

    stwu   1,-32(1)
    cfi_adjust_cfa_offset(32)
    cmplwi cr1,5,31     /* check for short move.  */
    neg    0,3
    cmplwi cr1,5,31
    clrlwi 10,4,30	/* check alignment of src.  */
    andi.  11,3,3	/* check alignment of dst.  */
    clrlwi 0,0,30	/* Number of bytes until the 1st word of dst.  */
    ble-   cr1,L(word_unaligned_short)	/* If move < 32 bytes.  */
    cmplw  cr6,10,11
    stw    31,24(1)
    stw    30,20(1)
    cfi_offset(31,(24-32))
    cfi_offset(30,(20-32))
    mr     30,3
    beq    .L0
    mtcrf  0x01,0
    subf  31,0,5        /* Length after alignment.  */
    add   12,4,0        /* Compute src addr after alignment.  */
  /* Move 0-3 bytes as needed to get the destination word aligned.  */
1:  bf    31,2f
    lbz   6,0(4)
    bf    30,3f
    lhz   7,1(4)
    stb   6,0(3)
    sth   7,1(3)
    addi  3,3,3
    b     0f
3:
    stb   6,0(3)
    addi  3,3,1
    b     0f
2:  bf    30,0f
    lhz   6,0(4)
    sth   6,0(3)
    addi  3,3,2
0:
    clrlwi 10,12,30	/* check alignment of src again.  */
    srwi   9,31,2	/* Number of full words remaining.  */
    bne-   cr6,L(wdu)   /* If source is not word aligned. .L6 */
    clrlwi 11,31,30  /* calculate the number of tail bytes */
    b      L(word_aligned)
  /* Copy words from source to destination, assuming the destination is
     aligned on a word boundary.

     At this point we know there are at least 29 bytes left (32-3) to copy.
     The next step is to determine if the source is also word aligned.
     If not branch to the unaligned move code at .L6. which uses
     a load, shift, store strategy.

     Otherwise source and destination are word aligned, and we can use
     the optimized word copy loop.  */
    .align  4
.L0:
    mr     31,5
    mr     12,4
    bne-   cr6,L(wdu)   /* If source is not word aligned. .L6 */
    srwi   9,5,2	/* Number of full words remaining.  */
    clrlwi 11,5,30      /* calculate the number of tail bytes */

  /* Move words where destination and source are word aligned.
     Use an unrolled loop to copy 4 words (16-bytes) per iteration.
     If the copy is not an exact multiple of 16 bytes, 1-3
     words are copied as needed to set up the main loop.  After
     the main loop exits there may be a tail of 1-3 bytes. These bytes are
     copied a halfword/byte at a time as needed to preserve alignment.  */
L(word_aligned):
    mtcrf 0x01,9
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

    cfi_same_value (31)
    cfi_same_value (30)
L(word_unaligned_short):
    mtcrf 0x01,5
    cmplwi cr6,5,8
    neg   8,4
    clrrwi	9,4,2
    andi. 0,8,3
    beq   cr6,L(wus_8)	/* Handle moves of 8 bytes.  */
/* At least 9 bytes left.  Get the source word aligned.  */
    cmplwi	cr1,5,16
    mr    12,4
    ble   cr6,L(wus_4)  /* Handle moves of 0-8 bytes.  */
    mr    11,3
    mr    10,5
    cmplwi	cr6,0,2
    beq   L(wus_tail)	/* If the source is already word aligned skip this.  */
/* Copy 1-3 bytes to get source address word aligned.  */
    lwz   6,0(9)
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
    add   11,3,0
    mtcrf 0x01,10
    .align  4
L(wus_tail):
/* At least 6 bytes left and the source is word aligned.  This allows
   some speculative loads up front.  */
/* We need to special case the fall-through because the biggest delays
   are due to address computation not being ready in time for the
   AGEN.  */
    lwz   6,0(12)
    lwz   7,4(12)
    blt   cr1,L(wus_tail8)
    cmplwi	cr0,10,24
L(wus_tail16): /* Move 16 bytes.  */
    stw   6,0(11)
    stw   7,4(11)
    lwz   6,8(12)
    lwz   7,12(12)
    stw   6,8(11)
    stw   7,12(11)
/* Move 8 bytes more.  */
    bf    28,L(wus_tail16p8)
    cmplwi	cr1,10,28
    lwz   6,16(12)
    lwz   7,20(12)
    stw   6,16(11)
    stw   7,20(11)
/* Move 4 bytes more.  */
    bf    29,L(wus_tail16p4)
    lwz   6,24(12)
    stw   6,24(11)
    addi  12,12,28
    addi  11,11,28
    bgt   cr1,L(wus_tail2)
 /* exactly 28 bytes.  Return original dst pointer and exit.  */
    addi  1,1,32
    blr
    .align  4
L(wus_tail16p8):  /* less than 8 bytes left.  */
    beq   cr1,L(wus_tailX) /* exactly 16 bytes, early exit.  */
    cmplwi	cr1,10,20
    bf    29,L(wus_tail16p2)
/* Move 4 bytes more.  */
    lwz   6,16(12)
    stw   6,16(11)
    addi  12,12,20
    addi  11,11,20
    bgt   cr1,L(wus_tail2)
 /* exactly 20 bytes.  Return original dst pointer and exit.  */
    addi  1,1,32
    blr
    .align  4
L(wus_tail16p4):  /* less than 4 bytes left.  */
    addi  12,12,24
    addi  11,11,24
    bgt   cr0,L(wus_tail2)
 /* exactly 24 bytes.  Return original dst pointer and exit.  */
    addi  1,1,32
    blr
    .align  4
L(wus_tail16p2):  /* 16 bytes moved, less than 4 bytes left.  */
    addi  12,12,16
    addi  11,11,16
    b     L(wus_tail2)

    .align  4
L(wus_tail8):  /* Move 8 bytes.  */
/*  r6, r7 already loaded speculatively.  */
    cmplwi	cr1,10,8
    cmplwi	cr0,10,12
    bf    28,L(wus_tail4)
    stw   6,0(11)
    stw   7,4(11)
/* Move 4 bytes more.  */
    bf    29,L(wus_tail8p4)
    lwz   6,8(12)
    stw   6,8(11)
    addi  12,12,12
    addi  11,11,12
    bgt   cr0,L(wus_tail2)
 /* exactly 12 bytes.  Return original dst pointer and exit.  */
    addi  1,1,32
    blr
    .align  4
L(wus_tail8p4):  /* less than 4 bytes left.  */
    addi  12,12,8
    addi  11,11,8
    bgt   cr1,L(wus_tail2)
 /* exactly 8 bytes.  Return original dst pointer and exit.  */
    addi  1,1,32
    blr

    .align  4
L(wus_tail4):  /* Move 4 bytes.  */
/*  r6 already loaded speculatively.  If we are here we know there is
    more than 4 bytes left.  So there is no need to test.  */
    addi  12,12,4
    stw   6,0(11)
    addi  11,11,4
L(wus_tail2):  /* Move 2-3 bytes.  */
    bf    30,L(wus_tail1)
    lhz   6,0(12)
    sth   6,0(11)
    bf    31,L(wus_tailX)
    lbz   7,2(12)
    stb   7,2(11)
    addi  1,1,32
    blr
L(wus_tail1):  /* Move 1 byte.  */
    bf    31,L(wus_tailX)
    lbz   6,0(12)
    stb   6,0(11)
L(wus_tailX):
  /* Return original dst pointer.  */
    addi  1,1,32
    blr

/* Special case to copy 0-8 bytes.  */
    .align  4
L(wus_8):
    lwz   6,0(4)
    lwz   7,4(4)
    stw   6,0(3)
    stw   7,4(3)
  /* Return original dst pointer.  */
    addi  1,1,32
    blr
    .align  4
L(wus_4):
    bf    29,L(wus_2)
    lwz   6,0(4)
    stw   6,0(3)
    bf    30,L(wus_5)
    lhz   7,4(4)
    sth   7,4(3)
    bf    31,L(wus_0)
    lbz   8,6(4)
    stb   8,6(3)
    addi  1,1,32
    blr
    .align  4
L(wus_5):
    bf    31,L(wus_0)
    lbz   6,4(4)
    stb   6,4(3)
  /* Return original dst pointer.  */
    addi 1,1,32
    blr
    .align  4
L(wus_2):  /* Move 2-3 bytes.  */
    bf    30,L(wus_1)
    lhz   6,0(4)
    sth   6,0(3)
    bf    31,L(wus_0)
    lbz   7,2(4)
    stb   7,2(3)
    addi  1,1,32
    blr
    .align  4
L(wus_1):  /* Move 1 byte.  */
    bf    31,L(wus_0)
    lbz   6,0(4)
    stb   6,0(3)
    .align  3
L(wus_0):
  /* Return original dst pointer.  */
    addi  1,1,32
    blr

    .align  4
    cfi_offset(31,(24-32))
    cfi_offset(30,(20-32))
L(wdu):

  /* Copy words where the destination is aligned but the source is
     not.  For power4, power5 and power6 machines there is penalty for
     unaligned loads (src) that cross 32-byte, cacheline, or page
     boundaries. So we want to use simple (unaligned) loads where
     possible but avoid them where we know the load would span a 32-byte
     boundary.

     At this point we know we have at least 29 (32-3) bytes to copy
     the src is unaligned. and we may cross at least one 32-byte
     boundary. Also we have the following register values:
     r3 == adjusted dst, word aligned
     r4 == unadjusted src
     r5 == unadjusted len
     r9 == adjusted Word length
     r10 == src alignment (1-3)
     r12 == adjusted src, not aligned
     r31 == adjusted len

     First we need to copy word up to but not crossing the next 32-byte
     boundary. Then perform aligned loads just before and just after
     the boundary and use shifts and or to generate the next aligned
     word for dst. If more than 32 bytes remain we copy (unaligned src)
     the next 7 words and repeat the loop until less than 32-bytes
     remain.

     Then if more than 4 bytes remain we again use aligned loads,
     shifts and or to generate the next dst word. We then process the
     remaining words using unaligned loads as needed. Finally we check
     if there are more than 0 bytes (1-3) bytes remaining and use
     halfword and or byte load/stores to complete the copy.
*/
    mr      4,12      /* restore unaligned adjusted src ptr */
    clrlwi  0,12,27   /* Find dist from previous 32-byte boundary.  */
    slwi    10,10,3   /* calculate number of bits to shift 1st word left */
    cmplwi  cr5,0,16
    subfic  8,0,32   /* Number of bytes to next 32-byte boundary.  */

    mtcrf   0x01,8
    cmplwi  cr1,10,16
    subfic  9,10,32  /* number of bits to shift 2nd word right */
/*  This test is reversed because the timing to compare the bytes to
    32-byte boundary could not be meet.  So we compare the bytes from
    previous 32-byte boundary and invert the test.  */
    bge     cr5,L(wdu_h32_8)
    .align  4
    lwz   6,0(4)
    lwz   7,4(4)
    addi  12,4,16    /* generate alternate pointers to avoid agen */
    addi  11,3,16    /* timing issues downstream.  */
    stw   6,0(3)
    stw   7,4(3)
    subi  31,31,16
    lwz   6,8(4)
    lwz   7,12(4)
    addi  4,4,16
    stw   6,8(3)
    stw   7,12(3)
    addi  3,3,16
    bf    28,L(wdu_h32_4)
    lwz   6,0(12)
    lwz   7,4(12)
    subi  31,31,8
    addi  4,4,8
    stw   6,0(11)
    stw   7,4(11)
    addi  3,3,8
    bf    29,L(wdu_h32_0)
    lwz   6,8(12)
    addi  4,4,4
    subi  31,31,4
    stw   6,8(11)
    addi  3,3,4
    b     L(wdu_h32_0)
    .align  4
L(wdu_h32_8):
    bf    28,L(wdu_h32_4)
    lwz   6,0(4)
    lwz   7,4(4)
    subi  31,31,8
    bf    29,L(wdu_h32_8x)
    stw   6,0(3)
    stw   7,4(3)
    lwz   6,8(4)
    addi  4,4,12
    subi  31,31,4
    stw   6,8(3)
    addi  3,3,12
    b     L(wdu_h32_0)
    .align  4
L(wdu_h32_8x):
    addi  4,4,8
    stw   6,0(3)
    stw   7,4(3)
    addi  3,3,8
    b     L(wdu_h32_0)
    .align  4
L(wdu_h32_4):
    bf    29,L(wdu_h32_0)
    lwz   6,0(4)
    subi  31,31,4
    addi  4,4,4
    stw   6,0(3)
    addi  3,3,4
    .align  4
L(wdu_h32_0):
/*  set up for 32-byte boundary crossing word move and possibly 32-byte
    move loop.  */
    clrrwi  12,4,2
    cmplwi  cr5,31,32
    bge     cr1,L(wdu2_32)
#if 0
    b       L(wdu1_32)
/*
    cmplwi  cr1,10,8
    beq     cr1,L(wdu1_32)
    cmplwi  cr1,10,16
    beq     cr1,L(wdu2_32)
    cmplwi  cr1,10,24
    beq     cr1,L(wdu3_32)
*/
L(wdu_32):
    lwz     6,0(12)
    cmplwi  cr6,31,4
    srwi    8,31,5    /* calculate the 32 byte loop count */
    slw     0,6,10
    clrlwi  31,31,27   /* The remaining bytes, < 32.  */
    blt     cr5,L(wdu_32tail)
    mtctr   8
    cmplwi  cr6,31,4
    .align  4
L(wdu_loop32):
    /* copy 32 bytes at a time */
    lwz   8,4(12)
    addi  12,12,32
    lwz   7,4(4)
    srw   8,8,9
    or    0,0,8
    stw   0,0(3)
    stw   7,4(3)
    lwz   6,8(4)
    lwz   7,12(4)
    stw   6,8(3)
    stw   7,12(3)
    lwz   6,16(4)
    lwz   7,20(4)
    stw   6,16(3)
    stw   7,20(3)
    lwz   6,24(4)
    lwz   7,28(4)
    lwz   8,0(12)
    addi  4,4,32
    stw   6,24(3)
    stw   7,28(3)
    addi  3,3,32
    slw   0,8,10
    bdnz+ L(wdu_loop32)

L(wdu_32tail):
    mtcrf   0x01,31
    cmplwi  cr5,31,16
    blt     cr6,L(wdu_4tail)
    /* calculate and store the final word */
    lwz   8,4(12)
    srw   8,8,9
    or    6,0,8
    b     L(wdu_32tailx)
#endif
    .align  4
L(wdu1_32):
    lwz     6,-1(4)
    cmplwi  cr6,31,4
    srwi    8,31,5    /* calculate the 32 byte loop count */
#ifdef __LITTLE_ENDIAN__
    srwi    6,6,8
#else
    slwi    6,6,8
#endif
    clrlwi  31,31,27   /* The remaining bytes, < 32.  */
    blt     cr5,L(wdu1_32tail)
    mtctr   8
    cmplwi  cr6,31,4

    lwz   8,3(4)
    lwz   7,4(4)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,24,32
#else
/*  Equivalent to: srwi   8,8,32-8;  or    6,6,8 */
    rlwimi 6,8,8,(32-8),31
#endif
    b      L(wdu1_loop32x)
    .align  4
L(wdu1_loop32):
    /* copy 32 bytes at a time */
    lwz   8,3(4)
    lwz   7,4(4)
    stw   10,-8(3)
    stw   11,-4(3)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,24,32
#else
/*  Equivalent to  srwi   8,8,32-8; or    6,6,8 */
    rlwimi 6,8,8,(32-8),31
#endif
L(wdu1_loop32x):
    lwz   10,8(4)
    lwz   11,12(4)
    stw   6,0(3)
    stw   7,4(3)
    lwz   6,16(4)
    lwz   7,20(4)
    stw   10,8(3)
    stw   11,12(3)
    lwz   10,24(4)
    lwz   11,28(4)
    lwz   8,32-1(4)
    addi  4,4,32
    stw   6,16(3)
    stw   7,20(3)
    addi  3,3,32
#ifdef __LITTLE_ENDIAN__
    srwi  6,8,8
#else
    slwi  6,8,8
#endif
    bdnz+ L(wdu1_loop32)
    stw   10,-8(3)
    stw   11,-4(3)

L(wdu1_32tail):
    mtcrf   0x01,31
    cmplwi  cr5,31,16
    blt     cr6,L(wdu_4tail)
    /* calculate and store the final word */
    lwz   8,3(4)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,24,32
#else
/*  Equivalent to: srwi   8,8,32-8;  or    6,6,8  */
    rlwimi 6,8,8,(32-8),31
#endif
    b     L(wdu_32tailx)

L(wdu2_32):
    bgt     cr1,L(wdu3_32)
    lwz     6,-2(4)
    cmplwi  cr6,31,4
    srwi    8,31,5    /* calculate the 32 byte loop count */
#ifdef __LITTLE_ENDIAN__
    srwi    6,6,16
#else
    slwi    6,6,16
#endif
    clrlwi  31,31,27   /* The remaining bytes, < 32.  */
    blt     cr5,L(wdu2_32tail)
    mtctr   8
    cmplwi  cr6,31,4

    lwz   8,2(4)
    lwz   7,4(4)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,16,32
#else
    rlwimi 6,8,16,(32-16),31
#endif
    b      L(wdu2_loop32x)
    .align  4
L(wdu2_loop32):
    /* copy 32 bytes at a time */
    lwz   8,2(4)
    lwz   7,4(4)
    stw   10,-8(3)
    stw   11,-4(3)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,16,32
#else
    rlwimi 6,8,16,(32-16),31
#endif
L(wdu2_loop32x):
    lwz   10,8(4)
    lwz   11,12(4)
    stw   6,0(3)
    stw   7,4(3)
    lwz   6,16(4)
    lwz   7,20(4)
    stw   10,8(3)
    stw   11,12(3)
    lwz   10,24(4)
    lwz   11,28(4)
/*    lwz   8,0(12) */
    lwz   8,32-2(4)
    addi  4,4,32
    stw   6,16(3)
    stw   7,20(3)
    addi  3,3,32
#ifdef __LITTLE_ENDIAN__
    srwi  6,8,16
#else
    slwi  6,8,16
#endif
    bdnz+ L(wdu2_loop32)
    stw   10,-8(3)
    stw   11,-4(3)

L(wdu2_32tail):
    mtcrf   0x01,31
    cmplwi  cr5,31,16
    blt     cr6,L(wdu_4tail)
    /* calculate and store the final word */
    lwz   8,2(4)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,16,32
#else
    rlwimi 6,8,16,(32-16),31
#endif
    b     L(wdu_32tailx)

L(wdu3_32):
/*    lwz     6,0(12) */
    lwz     6,-3(4)
    cmplwi  cr6,31,4
    srwi    8,31,5    /* calculate the 32 byte loop count */
#ifdef __LITTLE_ENDIAN__
    srwi    6,6,24
#else
    slwi    6,6,24
#endif
    clrlwi  31,31,27   /* The remaining bytes, < 32.  */
    blt     cr5,L(wdu3_32tail)
    mtctr   8
    cmplwi  cr6,31,4

    lwz   8,1(4)
    lwz   7,4(4)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,8,32
#else
    rlwimi 6,8,24,(32-24),31
#endif
    b      L(wdu3_loop32x)
    .align  4
L(wdu3_loop32):
    /* copy 32 bytes at a time */
    lwz   8,1(4)
    lwz   7,4(4)
    stw   10,-8(3)
    stw   11,-4(3)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,8,32
#else
    rlwimi 6,8,24,(32-24),31
#endif
L(wdu3_loop32x):
    lwz   10,8(4)
    lwz   11,12(4)
    stw   6,0(3)
    stw   7,4(3)
    lwz   6,16(4)
    lwz   7,20(4)
    stw   10,8(3)
    stw   11,12(3)
    lwz   10,24(4)
    lwz   11,28(4)
    lwz   8,32-3(4)
    addi  4,4,32
    stw   6,16(3)
    stw   7,20(3)
    addi  3,3,32
#ifdef __LITTLE_ENDIAN__
    srwi  6,8,24
#else
    slwi  6,8,24
#endif
    bdnz+ L(wdu3_loop32)
    stw   10,-8(3)
    stw   11,-4(3)

L(wdu3_32tail):
    mtcrf   0x01,31
    cmplwi  cr5,31,16
    blt     cr6,L(wdu_4tail)
    /* calculate and store the final word */
    lwz   8,1(4)
#ifdef __LITTLE_ENDIAN__
    rldimi 6,8,8,32
#else
    rlwimi 6,8,24,(32-24),31
#endif
    b     L(wdu_32tailx)
    .align  4
L(wdu_32tailx):
    blt     cr5,L(wdu_t32_8)
    lwz   7,4(4)
    addi  12,4,16    /* generate alternate pointers to avoid agen */
    addi  11,3,16    /* timing issues downstream.  */
    stw   6,0(3)
    stw   7,4(3)
    subi  31,31,16
    lwz   6,8(4)
    lwz   7,12(4)
    addi  4,4,16
    stw   6,8(3)
    stw   7,12(3)
    addi  3,3,16
    bf    28,L(wdu_t32_4x)
    lwz   6,0(12)
    lwz   7,4(12)
    addi  4,4,8
    subi  31,31,8
    stw   6,0(11)
    stw   7,4(11)
    addi  3,3,8
    bf    29,L(wdu_t32_0)
    lwz   6,8(12)
    addi  4,4,4
    subi  31,31,4
    stw   6,8(11)
    addi  3,3,4
    b     L(wdu_t32_0)
    .align  4
L(wdu_t32_4x):
    bf    29,L(wdu_t32_0)
    lwz   6,0(4)
    addi  4,4,4
    subi  31,31,4
    stw   6,0(3)
    addi  3,3,4
    b     L(wdu_t32_0)
    .align  4
L(wdu_t32_8):
    bf    28,L(wdu_t32_4)
    lwz   7,4(4)
    subi  31,31,8
    bf    29,L(wdu_t32_8x)
    stw   6,0(3)
    stw   7,4(3)
    lwz   6,8(4)
    subi  31,31,4
    addi  4,4,12
    stw   6,8(3)
    addi  3,3,12
    b     L(wdu_t32_0)
    .align  4
L(wdu_t32_8x):
    addi  4,4,8
    stw   6,0(3)
    stw   7,4(3)
    addi  3,3,8
    b     L(wdu_t32_0)
    .align  4
L(wdu_t32_4):
    subi  31,31,4
    stw   6,0(3)
    addi  4,4,4
    addi  3,3,4
    .align  4
L(wdu_t32_0):
L(wdu_4tail):
    cmplwi  cr6,31,0
    beq   cr6,L(wdus_0)	/* If the tail is 0 bytes we are done!  */
    bf    30,L(wdus_3)
    lhz   7,0(4)
    sth   7,0(3)
    bf    31,L(wdus_0)
    lbz   8,2(4)
    stb   8,2(3)
    mr    3,30
    lwz   30,20(1)
    lwz   31,24(1)
    addi  1,1,32
    blr
    .align  4
L(wdus_3):
    bf    31,L(wus_0)
    lbz   6,0(4)
    stb   6,0(3)
    .align  4
L(wdus_0):
  /* Return original dst pointer.  */
    mr   3,30
    lwz  30,20(1)
    lwz  31,24(1)
    addi 1,1,32
    blr
END (memcpy)

libc_hidden_builtin_def (memcpy)
