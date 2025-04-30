/* Out-of-line syscall stub for six-argument syscalls from C.
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

#ifndef OPTIMIZE_FOR_GCC_5

/* %eax, %ecx, %edx and %esi contain the values expected by the kernel.
   %edi points to a structure with the values of %ebx, %edi and %ebp.  */

	.hidden __libc_do_syscall

ENTRY (__libc_do_syscall)
	pushl	%ebx
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (ebx, 0)
	pushl	%edi
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (edi, 0)
	pushl	%ebp
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (ebp, 0)
	movl	0(%edi), %ebx
	movl	8(%edi), %ebp
	movl	4(%edi), %edi
	ENTER_KERNEL
	popl	%ebp
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebp)
	popl	%edi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)
	popl	%ebx
	cfi_adjust_cfa_offset (-4)
	cfi_restore (ebx)
	ret
END (__libc_do_syscall)
#endif
