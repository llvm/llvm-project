/* Optimized strlen implementation for PowerPC476.
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

/* strlen

       Register Use
       r3:source address and return length of string
       r4:byte counter

       Implementation description
       Load 2 words at a time and count bytes, if we find null we subtract one from
       the count and return the count value. We need to subtract one because
       we don't count the null character as a byte. */

EALIGN (strlen,5,0)
       neg     r7,r3
       clrlwi. r8,r7,29
       addi    r4,0,0
       beq     L(byte_count_loop)
       mtctr   r8

L(loop):
       lbz     r5,0(r3)
       cmpi    cr5,r5,0x0
       addi    r3,r3,0x1
       addi    r4,r4,0x1
       beq     cr5,L(end_strlen)
       bdnz    L(loop)

L(byte_count_loop):
       lwz     r5,0(r3)
       lwz     r6,4(r3)
       dlmzb.  r12,r5,r6
       add     r4,r4,r12
       bne     L(end_strlen)
       lwz     r5,8(r3)
       lwz     r6,12(r3)
       dlmzb.  r12,r5,r6
       add     r4,r4,r12
       bne     L(end_strlen)
       lwz     r5,16(r3)
       lwz     r6,20(r3)
       dlmzb.  r12,r5,r6
       add     r4,r4,r12
       bne     L(end_strlen)
       lwz     r5,24(r3)
       lwz     r6,28(r3)
       addi    r3,r3,0x20
       dlmzb.  r12,r5,r6
       add     r4,r4,r12
       bne     L(end_strlen)
       b       L(byte_count_loop)

L(end_strlen):
       addi    r3,r4,-1
       blr
END (strlen)
libc_hidden_builtin_def (strlen)
