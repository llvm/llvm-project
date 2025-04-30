/* memcpy with REP MOVSB/STOSB
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#ifndef MEMCPY
# define MEMCPY		memcpy
# define MEMCPY_CHK	__memcpy_chk
#endif

#ifdef USE_AS_BCOPY
# define STR2		12
# define STR1		STR2+4
# define N     		STR1+4
#else
# define STR1		12
# define STR2		STR1+4
# define N     		STR2+4
#endif

#define CFI_PUSH(REG)						\
  cfi_adjust_cfa_offset (4);					\
  cfi_rel_offset (REG, 0)

#define CFI_POP(REG)						\
  cfi_adjust_cfa_offset (-4);					\
  cfi_restore (REG)

#define PUSH(REG)	pushl REG; CFI_PUSH (REG)
#define POP(REG)	popl REG; CFI_POP (REG)

	.text
#if defined SHARED && IS_IN (libc) && !defined USE_AS_BCOPY
ENTRY (MEMCPY_CHK)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (MEMCPY_CHK)
#endif
ENTRY (MEMCPY)
	PUSH	(%esi)
	PUSH	(%edi)
	movl	N(%esp), %ecx
	movl	STR1(%esp), %edi
	movl	STR2(%esp), %esi
	mov	%edi, %eax
#ifdef USE_AS_MEMPCPY
	add	%ecx, %eax
#endif

#ifdef USE_AS_MEMMOVE
	cmp	%esi, %edi
	ja	L(copy_backward)
	je	L(bwd_write_0bytes)
#endif

	rep	movsb
	POP	(%edi)
	POP	(%esi)
	ret

#ifdef USE_AS_MEMMOVE
L(copy_backward):
	lea	-1(%edi,%ecx), %edi
	lea	-1(%esi,%ecx), %esi
	std
	rep	movsb
	cld
L(bwd_write_0bytes):
	POP	(%edi)
	POP	(%esi)
	ret
#endif

END (MEMCPY)

#ifndef USE_AS_BCOPY
libc_hidden_builtin_def (MEMCPY)
#endif
