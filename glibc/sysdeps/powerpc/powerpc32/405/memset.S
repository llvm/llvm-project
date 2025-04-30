/* Optimized memset for PowerPC405,440,464 (32-byte cacheline).
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* memset

       r3:destination address and return address
       r4:source integer to copy
       r5:byte count
       r11:sources integer to copy in all 32 bits of reg
       r12:temp return address

       Save return address in r12
       If destinationn is unaligned and count is greater tha 255 bytes
       set 0-3 bytes to make destination aligned
       If count is greater tha 255 bytes and setting zero to memory
       use dbcz to set memeory when we can
       otherwsie do the follwoing
       If 16 or more words to set we use 16 word copy loop.
       Finaly we set 0-15 extra bytes with string store. */

EALIGN (memset, 5, 0)
       rlwinm  r11,r4,0,24,31
       rlwimi  r11,r4,8,16,23
       rlwimi  r11,r11,16,0,15
       addi    r12,r3,0
       cmpwi   r5,0x00FF
       ble     L(preword8_count_loop)
       cmpwi   r4,0x00
       beq     L(use_dcbz)
       neg     r6,r3
       clrlwi. r6,r6,30
       beq     L(preword8_count_loop)
       addi    r8,0,1
       mtctr   r6
       subi    r3,r3,1

L(unaligned_bytecopy_loop):
       stbu    r11,0x1(r3)
       subf.   r5,r8,r5
       beq     L(end_memset)
       bdnz    L(unaligned_bytecopy_loop)
       addi    r3,r3,1

L(preword8_count_loop):
       srwi.   r6,r5,4
       beq     L(preword2_count_loop)
       mtctr   r6
       addi    r3,r3,-4
       mr      r8,r11
       mr      r9,r11
       mr      r10,r11

L(word8_count_loop_no_dcbt):
       stwu    r8,4(r3)
       stwu    r9,4(r3)
       subi    r5,r5,0x10
       stwu    r10,4(r3)
       stwu    r11,4(r3)
       bdnz    L(word8_count_loop_no_dcbt)
       addi    r3,r3,4

L(preword2_count_loop):
       clrlwi. r7,r5,28
       beq     L(end_memset)
       mr      r8,r11
       mr      r9,r11
       mr      r10,r11
       mtxer   r7
       stswx   r8,0,r3

L(end_memset):
       addi    r3,r12,0
       blr

L(use_dcbz):
       neg     r6,r3
       clrlwi. r7,r6,28
       beq     L(skip_string_loop)
       mr      r8,r11
       mr      r9,r11
       mr      r10,r11
       subf    r5,r7,r5
       mtxer   r7
       stswx   r8,0,r3
       add     r3,r3,r7

L(skip_string_loop):
       clrlwi  r8,r6,27
       srwi.   r8,r8,4
       beq     L(dcbz_pre_loop)
       mtctr   r8

L(word_loop):
       stw     r11,0(r3)
       subi    r5,r5,0x10
       stw     r11,4(r3)
       stw     r11,8(r3)
       stw     r11,12(r3)
       addi    r3,r3,0x10
       bdnz    L(word_loop)

L(dcbz_pre_loop):
       srwi    r6,r5,5
       mtctr   r6
       addi    r7,0,0

L(dcbz_loop):
       dcbz    r3,r7
       addi    r3,r3,0x20
       subi    r5,r5,0x20
       bdnz    L(dcbz_loop)
       srwi.   r6,r5,4
       beq     L(postword2_count_loop)
       mtctr   r6

L(postword8_count_loop):
       stw     r11,0(r3)
       subi    r5,r5,0x10
       stw     r11,4(r3)
       stw     r11,8(r3)
       stw     r11,12(r3)
       addi    r3,r3,0x10
       bdnz    L(postword8_count_loop)

L(postword2_count_loop):
       clrlwi. r7,r5,28
       beq     L(end_memset)
       mr      r8,r11
       mr      r9,r11
       mr      r10,r11
       mtxer   r7
       stswx   r8,0,r3
       b       L(end_memset)
END (memset)
libc_hidden_builtin_def (memset)
