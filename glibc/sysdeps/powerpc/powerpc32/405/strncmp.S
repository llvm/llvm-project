/* Optimized strncmp implementation for PowerPC476.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* strncmp

       Register Use
       r0:temp return equality
       r3:source1 address, return equality
       r4:source2 address
       r5:byte count

       Implementation description
       Touch in 3 lines of D-cache.
       If source1 or source2 is unaligned copy 0-3 bytes to make source1 aligned
       Check 2 words from src1 and src2. If unequal jump to end and
       return src1 > src2 or src1 < src2.
       If null check bytes before null and then jump to end and
       return src1 > src2, src1 < src2 or src1 = src2.
       If count = zero check bytes before zero counter and then jump to end and
       return src1 > src2, src1 < src2 or src1 = src2.
       If src1 = src2 and no null, repeat. */

EALIGN (strncmp,5,0)
       neg     r7,r3
       clrlwi  r7,r7,20
       neg     r8,r4
       clrlwi  r8,r8,20
       srwi.   r7,r7,3
       beq     L(prebyte_count_loop)
       srwi.   r8,r8,3
       beq     L(prebyte_count_loop)
       cmplw   r7,r8
       mtctr   r7
       ble     L(preword2_count_loop)
       mtctr   r8

L(preword2_count_loop):
       srwi.   r6,r5,3
       beq     L(prebyte_count_loop)
       mfctr   r7
       cmplw   r6,r7
       bgt     L(set_count_loop)
       mtctr   r6
       clrlwi  r5,r5,29

L(word2_count_loop):
       lwz     r10,0(r3)
       lwz     r6,4(r3)
       addi    r3,r3,0x08
       lwz     r8,0(r4)
       lwz     r9,4(r4)
       addi    r4,r4,0x08
       dlmzb.  r12,r10,r6
       bne     L(end_check)
       cmplw   r10,r8
       bne     L(st1)
       cmplw   r6,r9
       bne     L(st1)
       bdnz    L(word2_count_loop)

L(prebyte_count_loop):
       addi    r5,r5,1
       mtctr   r5
       bdz     L(end_strncmp)

L(byte_count_loop):
       lbz     r6,0(r3)
       addi    r3,r3,1
       lbz     r7,0(r4)
       addi    r4,r4,1
       cmplw   r6,r7
       bne     L(st1)
       cmpwi   r6,0
       beq     L(end_strncmp)
       bdnz    L(byte_count_loop)
       b       L(end_strncmp)

L(set_count_loop):
       slwi    r7,r7,3
       subf    r5,r7,r5
       b       L(word2_count_loop)

L(end_check):
       subfic  r12,r12,4
       blt     L(end_check2)
       rlwinm  r12,r12,3,0,31
       srw     r10,r10,r12
       srw     r8,r8,r12
       cmplw   r10,r8
       bne     L(st1)
       b       L(end_strncmp)

L(end_check2):
       addi    r12,r12,4
       cmplw   r10,r8
       rlwinm  r12,r12,3,0,31
       bne     L(st1)
       srw     r6,r6,r12
       srw     r9,r9,r12
       cmplw   r6,r9
       bne     L(st1)

L(end_strncmp):
       addi    r3,r0,0
       blr

L(st1):
       mfcr    r3
       blr
END (strncmp)
libc_hidden_builtin_def (strncmp)
