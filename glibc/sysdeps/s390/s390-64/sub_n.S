/* __mpn_sub_n -- Add two limb vectors of the same length > 0 and store
   sum in a third limb vector.  64 bit S/390 version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
   This file is part of the GNU MP Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If not,
   see <https://www.gnu.org/licenses/>.  */

/* INPUT PARAMETERS
     %r2 = res_ptr
     %r3 = s1_ptr
     %r4 = s2_ptr
     %r5 = size.  */

#include "sysdep.h"
#include "asm-syntax.h"

ENTRY(__mpn_sub_n)
        stg    %r6,48(%r15)   # save register 6
	cfi_offset (%r6,-112)
        sgr    %r1,%r1
        lghi   %r0,1          # cannot use ahi to add carry, use slr
.L0:	lg     %r6,0(%r1,%r3) # .L0 -> no carry from last sub
	slg    %r6,0(%r1,%r4)
	stg    %r6,0(%r1,%r2)
	la     %r1,8(%r1)
	brc    4,.L3
.L1:	brct   %r5,.L0
	slgr   %r2,%r2	      # no last carry to return
	j      .Lexit
.L2:	lg     %r6,0(%r1,%r3) # .L2 -> carry from last sub
	slg    %r6,0(%r1,%r4)
	brc    4,.L4
	slgr   %r6,%r0	      # no carry yet, add carry from last sub
	stg    %r6,0(%r1,%r2)
	la     %r1,8(%r1)
	brc    11,.L1	      # new carry ?
.L3:	brct   %r5,.L2
	lgr    %r2,%r0	      # return last carry
	j      .Lexit
.L4:	slgr   %r6,%r0	      # already a carry, add carry from last sub
	stg    %r6,0(%r1,%r2)
	la     %r1,8(%r1)
	brct   %r5,.L2
	lgr    %r2,%r0	      # return last carry
.Lexit: lg     %r6,48(%r15)   # restore register 6
	br     %r14
END(__mpn_sub_n)
