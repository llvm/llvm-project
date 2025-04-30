/* strncpy - copy at most n characters from a string from source to
   destination.  For IBM S390
   This file is part of the GNU C Library.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/*
 * R2 = address of destination (dst)
 * R3 = address of source (src)
 * R4 = max of bytes to copy
 */

#include <ifunc-strncpy.h>
#include "sysdep.h"
#include "asm-syntax.h"

#if HAVE_STRNCPY_Z900_G5
ENTRY(STRNCPY_Z900_G5)
	.text
	st    %r2,24(%r15)          # save dst pointer
	slr   %r2,%r3               # %r3 points to src, %r2+%r3 to dst
	lhi   %r1,3
	nr    %r1,%r4               # last 2 bits of # bytes
	srl   %r4,2
	ltr   %r4,%r4               # less than 4 bytes to copy ?
	jz    .L1
	bras  %r5,.L0               # enter loop & load address of a 0
	.long 0
.L0:    icm   %r0,8,0(%r3)          # first byte
	jz    .L3
	icm   %r0,4,1(%r3)          # second byte
	jz    .L4
	icm   %r0,2,2(%r3)          # third byte
	jz    .L5
	icm   %r0,1,3(%r3)          # fourth byte
	jz    .L6
	st    %r0,0(%r2,%r3)        # store all four to dest.
	la    %r3,4(%r3)
	brct  %r4,.L0
.L1:    ltr   %r1,%r1
	jz    .Lexit
.L2:    icm   %r0,1,0(%r3)
	stc   %r0,0(%r2,%r3)
	la    %r3,1(%r3)
	jz    .L7
	brct  %r1,.L2
	j     .Lexit
.L3:    icm   %r0,4,0(%r5)
.L4:    icm   %r0,2,0(%r5)
.L5:    icm   %r0,1,0(%r5)
.L6:    st    %r0,0(%r2,%r3)
	la    %r3,4(%r3)
	ahi   %r4,-1
	j     .L8
.L7:    ahi   %r1,-1
.L8:    sll   %r4,2
	alr   %r4,%r1
	alr   %r2,%r3               # start of dst area to be zeroed
	lr    %r3,%r4
	slr   %r4,%r4
	slr   %r5,%r5
.L9:    mvcle %r2,%r4,0             # pad dst with zeroes
	jo    .L9
.Lexit: l     %r2,24(%r15)          # return dst pointer
	br    %r14
END(STRNCPY_Z900_G5)

# if ! HAVE_STRNCPY_IFUNC
strong_alias (STRNCPY_Z900_G5, strncpy)
# endif

# if defined SHARED && IS_IN (libc)
strong_alias (STRNCPY_Z900_G5, __GI_strncpy)
# endif
#endif
