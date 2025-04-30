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

/* This source produces 3 functions:
   __sigsetjmp (jmpbuf,savemask)
   setjmp (jmpbuf) - equivalent to __sigsetjump (jmpbuf, 1)
   _setjmp (jmpbuf) - equivalent to __sigsetjump (jmpbuf, 0)
*/

#ifndef setjmp_name
# define setjmp_name __sigsetjmp
# define SET_MASK_ARG
#else
# define SET_MASK_ARG    addik r6,r0,SAVEMASK
#endif

ENTRY (setjmp_name)
    swi   r1,r5,0
    swi   r2,r5,4
    swi   r13,r5,8
    swi   r14,r5,12
    swi   r15,r5,16
    swi   r16,r5,20
    swi   r17,r5,24
    swi   r18,r5,28
    swi   r19,r5,32
    swi   r20,r5,36
    swi   r21,r5,40
    swi   r22,r5,44
    swi   r23,r5,48
    swi   r24,r5,52
    swi   r25,r5,56
    swi   r26,r5,60
    swi   r27,r5,64
    swi   r28,r5,68
    swi   r29,r5,72
    swi   r30,r5,76
    swi   r31,r5,80

    SET_MASK_ARG

    /* Make a tail call to __sigjmp_save; it takes the same args.  */
#ifdef	PIC
    /* We cannot use the PLT, because it requires that r20 be set, but
       we can't save and restore our caller's value.  Instead, we do an
       indirect jump through the GOT, using the temporary register
       %ecx, which is call-clobbered.  */
    mfs   r12,rpc
    addik r12,r12,_GLOBAL_OFFSET_TABLE_+8
    lwi   r12,r12,__sigjmp_save@GOT
    brad  r12
#else
    brid  __sigjmp_save
#endif
    nop
END (setjmp_name)

libc_hidden_def (setjmp_name)
