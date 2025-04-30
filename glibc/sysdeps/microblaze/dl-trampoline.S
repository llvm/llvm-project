/* Copyright (C) 2005-2021 Free Software Foundation, Inc.

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
#include <libc-symbols.h>

#ifndef PROF

    .text
    .globl  _dl_runtime_resolve
    .type   _dl_runtime_resolve, @function
    .align  4
_dl_runtime_resolve:
    addik r1,r1,-40
    swi   r5,r1,12
    swi   r6,r1,16
    swi   r7,r1,20
    swi   r8,r1,24
    swi   r9,r1,28
    swi   r10,r1,32
    swi   r15,r1,0
    addk r5,r0,r4
    brlid r15, _dl_fixup
    addk r6,r0,r3; /* delay slot.  */
    lwi   r10,r1,32
    lwi   r9,r1,28
    lwi   r8,r1,24
    lwi   r7,r1,20
    lwi   r6,r1,16
    lwi   r5,r1,12
    lwi   r15,r1,0
    brad  r3
    addik r1,r1,40; /* delay slot.  */
    .size _dl_runtime_resolve, . - _dl_runtime_resolve

    .text
    .globl  _dl_runtime_profile
    .type   _dl_runtime_profile, @function
    .align  4
_dl_runtime_profile:
    addik r1,r1,-40
    swi   r5,r1,12
    swi   r6,r1,16
    swi   r7,r1,20
    swi   r8,r1,24
    swi   r9,r1,28
    swi   r10,r1,32
    swi   r15,r1,0
    addk r5,r0,r4
    brlid r15, _dl_profile_fixup
    addk r6,r0,r3; /* delay slot.  */
    lwi   r10,r1,32
    lwi   r9,r1,28
    lwi   r8,r1,24
    lwi   r7,r1,20
    lwi   r6,r1,16
    lwi   r5,r1,12
    lwi   r15,r1,0
    brad  r3
    addik r1,r1,40; /* delay slot.  */
    .size _dl_runtime_profile, . - _dl_runtime_profile

#else

    .text
    .globl  _dl_runtime_resolve
    .type   _dl_runtime_resolve, @function
    .align  4
_dl_runtime_resolve:
    addik r1,r1,-40
    swi   r5,r1,12
    swi   r6,r1,16
    swi   r7,r1,20
    swi   r8,r1,24
    swi   r9,r1,28
    swi   r10,r1,32
    swi   r15,r1,0
    addk r5,r0,r4
    brlid r15, fixup
    addk r6,r0,r3; /* delay slot.  */
    lwi   r10,r1,32
    lwi   r9,r1,28
    lwi   r8,r1,24
    lwi   r7,r1,20
    lwi   r6,r1,16
    lwi   r5,r1,12
    lwi   r15,r1,0
    brad  r3
    addik r1,r1,40; /* delay slot.  */
    .size _dl_runtime_resolve, . - _dl_runtime_resolve
    .globl _dl_runtime_profile
    .set _dl_runtime_profile, _dl_runtime_resolve

#endif
