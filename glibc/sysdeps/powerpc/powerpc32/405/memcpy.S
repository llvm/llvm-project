/* Optimized memcpy implementation for PowerPC476.
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

/* memcpy

       r0:return address
       r3:destination address
       r4:source address
       r5:byte count

       Save return address in r0.
       If destinationn and source are unaligned and copy count is greater than 256
       then copy 0-3 bytes to make destination aligned.
       If 32 or more bytes to copy we use 32 byte copy loop.
       Finaly we copy 0-31 extra bytes. */

EALIGN (memcpy, 5, 0)
/* Check if bytes to copy are greater than 256 and if
       source and destination are unaligned */
       cmpwi   r5,0x0100
       addi    r0,r3,0
       ble     L(string_count_loop)
       neg     r6,r3
       clrlwi. r6,r6,30
       beq     L(string_count_loop)
       neg     r6,r4
       clrlwi. r6,r6,30
       beq     L(string_count_loop)
       mtctr   r6
       subf    r5,r6,r5

L(unaligned_bytecopy_loop): /* Align destination by coping 0-3 bytes */
       lbz     r8,0x0(r4)
       addi    r4,r4,1
       stb     r8,0x0(r3)
       addi    r3,r3,1
       bdnz    L(unaligned_bytecopy_loop)
       srwi.   r7,r5,5
       beq     L(preword2_count_loop)
       mtctr   r7

L(word8_count_loop_no_dcbt): /* Copy 32 bytes at a time */
       lwz     r6,0(r4)
       lwz     r7,4(r4)
       lwz     r8,8(r4)
       lwz     r9,12(r4)
       subi    r5,r5,0x20
       stw     r6,0(r3)
       stw     r7,4(r3)
       stw     r8,8(r3)
       stw     r9,12(r3)
       lwz     r6,16(r4)
       lwz     r7,20(r4)
       lwz     r8,24(r4)
       lwz     r9,28(r4)
       addi    r4,r4,0x20
       stw     r6,16(r3)
       stw     r7,20(r3)
       stw     r8,24(r3)
       stw     r9,28(r3)
       addi    r3,r3,0x20
       bdnz    L(word8_count_loop_no_dcbt)

L(preword2_count_loop): /* Copy remaining 0-31 bytes */
       clrlwi. r12,r5,27
       beq     L(end_memcpy)
       mtxer   r12
       lswx    r5,0,r4
       stswx   r5,0,r3
       mr       r3,r0
       blr

L(string_count_loop): /* Copy odd 0-31 bytes */
       clrlwi. r12,r5,28
       add     r3,r3,r5
       add     r4,r4,r5
       beq     L(pre_string_copy)
       mtxer   r12
       subf    r4,r12,r4
       subf    r3,r12,r3
       lswx    r6,0,r4
       stswx   r6,0,r3

L(pre_string_copy): /* Check how many 32 byte chunks to copy */
       srwi.   r7,r5,4
       beq     L(end_memcpy)
       mtctr   r7

L(word4_count_loop_no_dcbt): /* Copy 32 bytes at a time */
       lwz     r6,-4(r4)
       lwz     r7,-8(r4)
       lwz     r8,-12(r4)
       lwzu    r9,-16(r4)
       stw     r6,-4(r3)
       stw     r7,-8(r3)
       stw     r8,-12(r3)
       stwu    r9,-16(r3)
       bdz     L(end_memcpy)
       lwz     r6,-4(r4)
       lwz     r7,-8(r4)
       lwz     r8,-12(r4)
       lwzu    r9,-16(r4)
       stw     r6,-4(r3)
       stw     r7,-8(r3)
       stw     r8,-12(r3)
       stwu    r9,-16(r3)
       bdnz    L(word4_count_loop_no_dcbt)

L(end_memcpy):
       mr       r3,r0
       blr
END (memcpy)
libc_hidden_builtin_def (memcpy)
