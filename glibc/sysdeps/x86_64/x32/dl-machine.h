/* Machine-dependent ELF dynamic relocation inline functions.  x32 version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Must allow <sysdeps/x86_64/dl-machine.h> to be included more than once.
   See #ifdef RESOLVE_MAP in sysdeps/x86_64/dl-machine.h.  */
#include <sysdeps/x86_64/dl-machine.h>

#ifndef _X32_DL_MACHINE_H
#define _X32_DL_MACHINE_H

#undef ARCH_LA_PLTENTER
#undef ARCH_LA_PLTEXIT
#undef RTLD_START

/* Names of the architecture-specific auditing callback functions.  */
#define ARCH_LA_PLTENTER x32_gnu_pltenter
#define ARCH_LA_PLTEXIT x32_gnu_pltexit

/* Initial entry point code for the dynamic linker.
   The C function `_dl_start' is the real entry point;
   its return value is the user program's entry point.  */
#define RTLD_START asm ("\n\
.text\n\
	.p2align 4\n\
.globl _start\n\
.globl _dl_start_user\n\
_start:\n\
	movl %esp, %edi\n\
	call _dl_start\n\
_dl_start_user:\n\
	# Save the user entry point address in %r12.\n\
	movl %eax, %r12d\n\
	# See if we were run as a command with the executable file\n\
	# name as an extra leading argument.\n\
	movl _dl_skip_args(%rip), %eax\n\
	# Pop the original argument count.\n\
	movl (%rsp), %edx\n\
	# Adjust the stack pointer to skip _dl_skip_args words.\n\
	lea 4(%rsp,%rax,4), %esp\n\
	# Subtract _dl_skip_args from argc.\n\
	subl %eax, %edx\n\
	# Push argc back on the stack.\n\
	subl $4, %esp\n\
	movl %edx, (%rsp)\n\
	# Call _dl_init (struct link_map *main_map, int argc, char **argv, char **env)\n\
	# argc -> rsi\n\
	movl %edx, %esi\n\
	# Save %rsp value in %r13.\n\
	movl %esp, %r13d\n\
	# And align stack for the _dl_init call.\n\
	and $-16, %esp\n\
	# _dl_loaded -> rdi\n\
	movl _rtld_local(%rip), %edi\n\
	# env -> rcx\n\
	lea 8(%r13,%rdx,4), %ecx\n\
	# argv -> rdx\n\
	lea 4(%r13), %edx\n\
	# Clear %rbp to mark outermost frame obviously even for constructors.\n\
	xorl %ebp, %ebp\n\
	# Call the function to run the initializers.\n\
	call _dl_init\n\
	# Pass our finalizer function to the user in %rdx, as per ELF ABI.\n\
	lea _dl_fini(%rip), %edx\n\
	# And make sure %rsp points to argc stored on the stack.\n\
	movl %r13d, %esp\n\
	# Jump to the user's entry point.\n\
	jmp *%r12\n\
.previous\n\
");

#endif /* !_X32_DL_MACHINE_H */
