/* Optimized memcmp implementation for PowerPC476.
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

/* memcmp

       r3:source1 address, return equality
       r4:source2 address
       r5:byte count

       Check 2 words from src1 and src2. If unequal jump to end and
       return src1 > src2 or src1 < src2.
       If count = zero check bytes before zero counter and then jump to end and
       return src1 > src2, src1 < src2 or src1 = src2.
       If src1 = src2 and no null, repeat. */

EALIGN (memcmp, 5, 0)
       srwi.   r6,r5,5
       beq     L(preword2_count_loop)
       mtctr   r6
       clrlwi  r5,r5,27

L(word8_compare_loop):
       lwz     r10,0(r3)
       lwz     r6,4(r3)
       lwz     r8,0(r4)
       lwz     r9,4(r4)
       cmplw   cr5,r8,r10
       cmplw   cr1,r9,r6
       bne     cr5,L(st2)
       bne     cr1,L(st1)
       lwz     r10,8(r3)
       lwz     r6,12(r3)
       lwz     r8,8(r4)
       lwz     r9,12(r4)
       cmplw   cr5,r8,r10
       cmplw   cr1,r9,r6
       bne     cr5,L(st2)
       bne     cr1,L(st1)
       lwz     r10,16(r3)
       lwz     r6,20(r3)
       lwz     r8,16(r4)
       lwz     r9,20(r4)
       cmplw   cr5,r8,r10
       cmplw   cr1,r9,r6
       bne     cr5,L(st2)
       bne     cr1,L(st1)
       lwz     r10,24(r3)
       lwz     r6,28(r3)
       addi    r3,r3,0x20
       lwz     r8,24(r4)
       lwz     r9,28(r4)
       addi    r4,r4,0x20
       cmplw   cr5,r8,r10
       cmplw   cr1,r9,r6
       bne     cr5,L(st2)
       bne     cr1,L(st1)
       bdnz    L(word8_compare_loop)

L(preword2_count_loop):
       srwi.   r6,r5,3
       beq     L(prebyte_count_loop)
       mtctr   r6
       clrlwi  r5,r5,29

L(word2_count_loop):
       lwz     r10,0(r3)
       lwz     r6,4(r3)
       addi    r3,r3,0x08
       lwz     r8,0(r4)
       lwz     r9,4(r4)
       addi    r4,r4,0x08
       cmplw   cr5,r8,r10
       cmplw   cr1,r9,r6
       bne     cr5,L(st2)
       bne     cr1,L(st1)
       bdnz    L(word2_count_loop)

L(prebyte_count_loop):
       addi    r5,r5,1
       mtctr   r5
       bdz     L(end_memcmp)

L(byte_count_loop):
       lbz     r6,0(r3)
       addi    r3,r3,0x01
       lbz     r8,0(r4)
       addi    r4,r4,0x01
       cmplw   cr5,r8,r6
       bne     cr5,L(st2)
       bdnz    L(byte_count_loop)

L(end_memcmp):
       addi    r3,r0,0
       blr

L(l_r):
       addi    r3,r0,1
       blr

L(st1):
       blt     cr1,L(l_r)
       addi    r3,r0,-1
       blr

L(st2):
       blt     cr5,L(l_r)
       addi    r3,r0,-1
       blr
END (memcmp)
libc_hidden_builtin_def (memcmp)
weak_alias (memcmp,bcmp)
