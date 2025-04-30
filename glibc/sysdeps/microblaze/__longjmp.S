/* Copyright (C) 1997-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* Jump to the position specified by ENV, causing the
   setjmp call there to return VAL, or 1 if VAL is 0.  */

/* void __longjmp (__jmp_buf env, int val) */

ENTRY(__longjmp)
    lwi   r1,r5,0
    lwi   r2,r5,4
    lwi   r13,r5,8
    lwi   r14,r5,12
    lwi   r15,r5,16
    lwi   r16,r5,20
    lwi   r17,r5,24
    lwi   r18,r5,28
    lwi   r19,r5,32
    lwi   r20,r5,36
    lwi   r21,r5,40
    lwi   r22,r5,44
    lwi   r23,r5,48
    lwi   r24,r5,52
    lwi   r25,r5,56
    lwi   r26,r5,60
    lwi   r27,r5,64
    lwi   r28,r5,68
    lwi   r29,r5,72
    lwi   r30,r5,76
    lwi   r31,r5,80
    addk  r3,r6,r0
    beqi  r3,L(return_1)
    rtsd  r15,8
    nop
L(return_1):
    rtsd  r15,8
    addik r3,r0,1	/* delay slot.  */
END(__longjmp)
