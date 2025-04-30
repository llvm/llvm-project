/* memset/bzero -- set memory area to CH/0
   Highly optimized version for ix86, x>=6.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1999.

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
#ifdef USE_AS_BZERO
# define DEST	PARMS
# define LEN	DEST+4
#else
# define RTN	PARMS
# define DEST	RTN
# define CHR	DEST+4
# define LEN	CHR+4
#endif

        .text
#if defined SHARED && IS_IN (libc) && !defined USE_AS_BZERO
ENTRY_CHK (__memset_chk)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END_CHK (__memset_chk)
#endif
ENTRY (memset)

	cld
	pushl	%edi
	cfi_adjust_cfa_offset (4)
	movl	DEST(%esp), %edx
	movl	LEN(%esp), %ecx
#ifdef USE_AS_BZERO
	xorl	%eax, %eax	/* fill with 0 */
#else
	movzbl	CHR(%esp), %eax
#endif
	jecxz	1f
	movl	%edx, %edi
	cfi_rel_offset (edi, 0)
	andl	$3, %edx
	jz	2f	/* aligned */
	jp	3f	/* misaligned at 3, store just one byte below */
	stosb		/* misaligned at 1 or 2, store two bytes */
	decl	%ecx
	jz	1f
3:	stosb
	decl	%ecx
	jz	1f
	xorl	$1, %edx
	jnz	2f	/* was misaligned at 2 or 3, now aligned */
	stosb		/* was misaligned at 1, store third byte */
	decl	%ecx
2:	movl	%ecx, %edx
	shrl	$2, %ecx
	andl	$3, %edx
#ifndef USE_AS_BZERO
	imul	$0x01010101, %eax
#endif
	rep
	stosl
	movl	%edx, %ecx
	rep
	stosb

1:
#ifndef USE_AS_BZERO
	movl DEST(%esp), %eax	/* start address of destination is result */
#endif
	popl	%edi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret
END (memset)
libc_hidden_builtin_def (memset)
