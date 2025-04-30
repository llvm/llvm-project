/* __mpn_sub_n -- Add two limb vectors of the same length > 0 and store
   sum in a third limb vector.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
   This file is part of the GNU MP Library.

   The GNU MP Library is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 2.1 of the License, or (at your
   option) any later version.

   The GNU MP Library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
   License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the GNU MP Library; see the file COPYING.LIB.  If not,
   see <https://www.gnu.org/licenses/>.  */

/*
  INPUT PARAMETERS
  res_ptr	%r2
  s1_ptr	%r3
  s2_ptr	%r4
  size		%r5
*/

#include "sysdep.h"
#include "asm-syntax.h"

ENTRY(__mpn_sub_n)
	st     %r6,24(%r15)   # save register 6
	cfi_offset (%r6, -72)
	sr     %r1,%r1
	lhi    %r0,1          # cannot use ahi to add carry, use slr
.L0:    l      %r6,0(%r1,%r3) # .L0 -> no carry from last sub
	sl     %r6,0(%r1,%r4)
	st     %r6,0(%r1,%r2)
	la     %r1,4(0,%r1)
	brc    4,.L3
.L1:    brct   %r5,.L0
	slr    %r2,%r2        # no last carry to return
	j      .Lexit
.L2:    l      %r6,0(%r1,%r3) # .L2 -> carry from last sub
	sl     %r6,0(%r1,%r4)
	brc    4,.L4
	slr    %r6,%r0        # no carry yet, add carry from last sub
	st     %r6,0(%r1,%r2)
	la     %r1,4(0,%r1)
	brc    11,.L1         # new carry ?
.L3:    brct   %r5,.L2
	lr     %r2,%r0        # return last carry
	j      .Lexit
.L4:    slr    %r6,%r0        # already a carry, add carry from last sub
	st     %r6,0(%r1,%r2)
	la     %r1,4(0,%r1)
	brct   %r5,.L2
	lr     %r2,%r0        # return last carry
.Lexit: l      %r6,24(%r15)   # restore register 6
	br     %r14
END(__mpn_sub_n)
