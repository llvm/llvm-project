/* strncpy - copy at most n characters from a string from source to
   destination.  64 bit S/390 version
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* INPUT PARAMETERS
     %r2 = address of destination (dst)
     %r3 = address of source (src)
     %r4 = max of bytes to copy.  */

#include <ifunc-strncpy.h>
#include "sysdep.h"
#include "asm-syntax.h"

#if HAVE_STRNCPY_Z900_G5
ENTRY(STRNCPY_Z900_G5)
	.text
	stg   %r2,48(%r15)	    # save dst pointer
	slgr  %r2,%r3		    # %r3 points to src, %r2+%r3 to dst
	lghi  %r1,7
	ngr   %r1,%r4		    # last 3 bits of # bytes
	srlg  %r4,%r4,3
	ltgr  %r4,%r4		    # less than 8 bytes to copy ?
	jz    .L1
	bras  %r5,.L0		    # enter loop & load address of a 0
	.long 0
.L0:	icmh  %r0,8,0(%r3)	    # first byte
	jz    .L3
	icmh  %r0,4,1(%r3)	    # second byte
	jz    .L4
	icmh  %r0,2,2(%r3)	    # third byte
	jz    .L5
	icmh  %r0,1,3(%r3)	    # fourth byte
	jz    .L6
	icm   %r0,8,4(%r3)	    # fifth byte
	jz    .L7
	icm   %r0,4,5(%r3)	    # sixth byte
	jz    .L8
	icm   %r0,2,6(%r3)	    # seventh byte
	jz    .L9
	icm   %r0,1,7(%r3)	    # eigth byte
	jz    .L10
	stg   %r0,0(%r2,%r3)	    # store all eight to dest.
	la    %r3,8(%r3)
	brct  %r4,.L0
.L1:	ltgr  %r1,%r1
	jz    .Lexit
.L2:	icm   %r0,1,0(%r3)
	stc   %r0,0(%r2,%r3)
	la    %r3,1(%r3)
	jz    .L11
	brct  %r1,.L2
	j     .Lexit
.L3:	icmh  %r0,4,0(%r5)
.L4:	icmh  %r0,2,0(%r5)
.L5:	icmh  %r0,1,0(%r5)
.L6:	icm   %r0,8,0(%r5)
.L7:	icm   %r0,4,0(%r5)
.L8:	icm   %r0,2,0(%r5)
.L9:	icm   %r0,1,0(%r5)
.L10:	stg   %r0,0(%r2,%r3)
	la    %r3,8(%r3)
	aghi  %r4,-1
	j     .L12
.L11:	aghi  %r1,-1
.L12:	sllg  %r4,%r4,3
	algr  %r4,%r1
	algr  %r2,%r3		    # start of dst area to be zeroed
	lgr   %r3,%r4
	slgr  %r4,%r4
	slgr  %r5,%r5
.L13:	mvcle %r2,%r4,0		    # pad dst with zeroes
	jo    .L13
.Lexit: lg    %r2,48(%r15)	    # return dst pointer
	br    %r14
END(STRNCPY_Z900_G5)

# if ! HAVE_STRNCPY_IFUNC
strong_alias (STRNCPY_Z900_G5, strncpy)
# endif

# if defined SHARED && IS_IN (libc)
strong_alias (STRNCPY_Z900_G5, __GI_strncpy)
# endif
#endif
