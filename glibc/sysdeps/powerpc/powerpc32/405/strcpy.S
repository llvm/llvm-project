/* Optimized strcpy implementation for PowerPC476.
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

/* strcpy

       Register Use
       r3:destination and return address
       r4:source address
       r10:temp destination address

       Implementation description
       Loop by checking 2 words at a time, with dlmzb. Check if there is a null
       in the 2 words. If there is a null jump to end checking to determine
       where in the last 8 bytes it is. Copy the appropriate bytes of the last
       8 according to the null position. */

EALIGN (strcpy, 5, 0)
       neg     r7,r4
       subi    r4,r4,1
       clrlwi. r8,r7,29
       subi    r10,r3,1
       beq     L(pre_word8_loop)
       mtctr   r8

L(loop):
       lbzu    r5,0x01(r4)
       cmpi    cr5,r5,0x0
       stbu    r5,0x01(r10)
       beq     cr5,L(end_strcpy)
       bdnz    L(loop)

L(pre_word8_loop):
       subi    r4,r4,3
       subi    r10,r10,3

L(word8_loop):
       lwzu    r5,0x04(r4)
       lwzu    r6,0x04(r4)
       dlmzb.  r11,r5,r6
       bne     L(byte_copy)
       stwu    r5,0x04(r10)
       stwu    r6,0x04(r10)
       lwzu    r5,0x04(r4)
       lwzu    r6,0x04(r4)
       dlmzb.  r11,r5,r6
       bne     L(byte_copy)
       stwu    r5,0x04(r10)
       stwu    r6,0x04(r10)
       lwzu    r5,0x04(r4)
       lwzu    r6,0x04(r4)
       dlmzb.  r11,r5,r6
       bne     L(byte_copy)
       stwu    r5,0x04(r10)
       stwu    r6,0x04(r10)
       lwzu    r5,0x04(r4)
       lwzu    r6,0x04(r4)
       dlmzb.  r11,r5,r6
       bne     L(byte_copy)
       stwu    r5,0x04(r10)
       stwu    r6,0x04(r10)
       b       L(word8_loop)

L(last_bytes_copy):
       stwu    r5,0x04(r10)
       subi    r11,r11,4
       mtctr   r11
       addi    r10,r10,3
       subi    r4,r4,1

L(last_bytes_copy_loop):
       lbzu    r5,0x01(r4)
       stbu    r5,0x01(r10)
       bdnz    L(last_bytes_copy_loop)
       blr

L(byte_copy):
       blt     L(last_bytes_copy)
       mtctr   r11
       addi    r10,r10,3
       subi    r4,r4,5

L(last_bytes_copy_loop2):
       lbzu    r5,0x01(r4)
       stbu    r5,0x01(r10)
       bdnz    L(last_bytes_copy_loop2)

L(end_strcpy):
       blr
END (strcpy)
libc_hidden_builtin_def (strcpy)
