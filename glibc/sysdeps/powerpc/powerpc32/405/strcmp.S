/* Optimized strcmp implementation for PowerPC476.
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

/* strcmp

       Register Use
       r0:temp return equality
       r3:source1 address, return equality
       r4:source2 address

       Implementation description
       Check 2 words from src1 and src2. If unequal jump to end and
       return src1 > src2 or src1 < src2.
       If null check bytes before null and then jump to end and
       return src1 > src2, src1 < src2 or src1 = src2.
       If src1 = src2 and no null, repeat. */

EALIGN (strcmp,5,0)
       neg     r7,r3
       clrlwi  r7,r7,20
       neg     r8,r4
       clrlwi  r8,r8,20
       srwi.   r7,r7,5
       beq     L(byte_loop)
       srwi.   r8,r8,5
       beq     L(byte_loop)
       cmplw   r7,r8
       mtctr   r7
       ble     L(big_loop)
       mtctr   r8

L(big_loop):
       lwz     r5,0(r3)
       lwz     r6,4(r3)
       lwz     r8,0(r4)
       lwz     r9,4(r4)
       dlmzb.  r12,r5,r6
       bne     L(end_check)
       cmplw   r5,r8
       bne     L(st1)
       cmplw   r6,r9
       bne     L(st1)
       lwz     r5,8(r3)
       lwz     r6,12(r3)
       lwz     r8,8(r4)
       lwz     r9,12(r4)
       dlmzb.  r12,r5,r6
       bne     L(end_check)
       cmplw   r5,r8
       bne     L(st1)
       cmplw   r6,r9
       bne     L(st1)
       lwz     r5,16(r3)
       lwz     r6,20(r3)
       lwz     r8,16(r4)
       lwz     r9,20(r4)
       dlmzb.  r12,r5,r6
       bne     L(end_check)
       cmplw   r5,r8
       bne     L(st1)
       cmplw   r6,r9
       bne     L(st1)
       lwz     r5,24(r3)
       lwz     r6,28(r3)
       addi    r3,r3,0x20
       lwz     r8,24(r4)
       lwz     r9,28(r4)
       addi    r4,r4,0x20
       dlmzb.  r12,r5,r6
       bne     L(end_check)
       cmplw   r5,r8
       bne     L(st1)
       cmplw   r6,r9
       bne     L(st1)
       bdnz    L(big_loop)
       b       L(byte_loop)

L(end_check):
       subfic  r12,r12,4
       blt     L(end_check2)
       rlwinm  r12,r12,3,0,31
       srw     r5,r5,r12
       srw     r8,r8,r12
       cmplw   r5,r8
       bne     L(st1)
       b       L(end_strcmp)

L(end_check2):
       addi    r12,r12,4
       cmplw   r5,r8
       rlwinm  r12,r12,3,0,31
       bne     L(st1)
       srw     r6,r6,r12
       srw     r9,r9,r12
       cmplw   r6,r9
       bne     L(st1)

L(end_strcmp):
       addi    r3,r0,0
       blr

L(st1):
       mfcr    r3
       blr

L(byte_loop):
       lbz     r5,0(r3)
       addi    r3,r3,1
       lbz     r6,0(r4)
       addi    r4,r4,1
       cmplw   r5,r6
       bne     L(st1)
       cmpwi   r5,0
       beq     L(end_strcmp)
       b       L(byte_loop)
END (strcmp)
libc_hidden_builtin_def (strcmp)
