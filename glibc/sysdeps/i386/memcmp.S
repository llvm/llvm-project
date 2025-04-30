/* Compare two memory blocks for differences in the first COUNT bytes.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include "asm-syntax.h"

#define PARMS	4+4	/* space for 1 saved reg */
#define BLK1	PARMS
#define BLK2	BLK1+4
#define LEN	BLK2+4

	.text
ENTRY (memcmp)

	pushl %esi		/* Save callee-safe registers.  */
	cfi_adjust_cfa_offset (4)
	movl %edi, %edx		/* Note that %edx is not used and can
				   so be used to save %edi.  It's faster.  */
	cfi_register (edi, edx)

	movl BLK1(%esp), %esi
	cfi_rel_offset (esi, 0)
	movl BLK2(%esp), %edi
	movl LEN(%esp), %ecx

	cld			/* Set direction of comparison.  */

	xorl %eax, %eax		/* Default result.  */

	repe			/* Compare at most %ecx bytes.  */
	cmpsb
	jz L(1)			/* If even last byte was equal we return 0.  */

	/* The memory blocks are not equal.  So result of the last
	   subtraction is present in the carry flag.  It is set when
	   the byte in block #2 is bigger.  In this case we have to
	   return -1 (=0xffffffff), else 1.  */
	sbbl %eax, %eax		/* This is tricky.  %eax == 0 and carry is set
				   or not depending on last subtraction.  */

	/* At this point %eax == 0, if the byte of block #1 was bigger, and
	   0xffffffff if the last byte of block #2 was bigger.  The latter
	   case is already correct but the former needs a little adjustment.
	   Note that the following operation does not change 0xffffffff.  */
	orb $1, %al		/* Change 0 to 1.  */

L(1):	popl %esi		/* Restore registers.  */
	cfi_adjust_cfa_offset (-4)
	cfi_restore (esi)
	movl %edx, %edi
	cfi_restore (edi)

	ret
END (memcmp)

#undef bcmp
weak_alias (memcmp, bcmp)
libc_hidden_builtin_def (memcmp)
