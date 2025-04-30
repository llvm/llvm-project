/* PLT trampolines.  i386 version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <link-defines.h>

#ifdef HAVE_MPX_SUPPORT
# define PRESERVE_BND_REGS_PREFIX bnd
#else
# define PRESERVE_BND_REGS_PREFIX .byte 0xf2
#endif

	.text
	.globl _dl_runtime_resolve
	.type _dl_runtime_resolve, @function
	cfi_startproc
	.align 16
_dl_runtime_resolve:
	cfi_adjust_cfa_offset (8)
	_CET_ENDBR
	pushl %eax		# Preserve registers otherwise clobbered.
	cfi_adjust_cfa_offset (4)
	pushl %ecx
	cfi_adjust_cfa_offset (4)
	pushl %edx
	cfi_adjust_cfa_offset (4)
	movl 16(%esp), %edx	# Copy args pushed by PLT in register.  Note
	movl 12(%esp), %eax	# that `fixup' takes its parameters in regs.
	call _dl_fixup		# Call resolver.
	popl %edx		# Get register content back.
	cfi_adjust_cfa_offset (-4)
	movl (%esp), %ecx
	movl %eax, (%esp)	# Store the function address.
	movl 4(%esp), %eax
	ret $12			# Jump to function address.
	cfi_endproc
	.size _dl_runtime_resolve, .-_dl_runtime_resolve

# The SHSTK compatible version.
	.text
	.globl _dl_runtime_resolve_shstk
	.type _dl_runtime_resolve_shstk, @function
	cfi_startproc
	.align 16
_dl_runtime_resolve_shstk:
	cfi_adjust_cfa_offset (8)
	_CET_ENDBR
	pushl %eax		# Preserve registers otherwise clobbered.
	cfi_adjust_cfa_offset (4)
	pushl %edx
	cfi_adjust_cfa_offset (4)
	movl 12(%esp), %edx	# Copy args pushed by PLT in register.  Note
	movl 8(%esp), %eax	# that `fixup' takes its parameters in regs.
	call _dl_fixup		# Call resolver.
	movl (%esp), %edx	# Get register content back.
	movl %eax, %ecx		# Store the function address.
	movl 4(%esp), %eax	# Get register content back.
	addl $16, %esp		# Adjust stack: PLT1 + PLT2 + %eax + %edx
	cfi_adjust_cfa_offset (-16)
	jmp *%ecx		# Jump to function address.
	cfi_endproc
	.size _dl_runtime_resolve_shstk, .-_dl_runtime_resolve_shstk

#ifndef PROF
# The SHSTK compatible version.
	.globl _dl_runtime_profile_shstk
	.type _dl_runtime_profile_shstk, @function
	cfi_startproc
	.align 16
_dl_runtime_profile_shstk:
	cfi_adjust_cfa_offset (8)
	_CET_ENDBR
	pushl %esp
	cfi_adjust_cfa_offset (4)
	addl $8, (%esp)		# Account for the pushed PLT data
	pushl %ebp
	cfi_adjust_cfa_offset (4)
	pushl %eax		# Preserve registers otherwise clobbered.
	cfi_adjust_cfa_offset (4)
	pushl %ecx
	cfi_adjust_cfa_offset (4)
	pushl %edx
	cfi_adjust_cfa_offset (4)
	movl %esp, %ecx
	subl $8, %esp
	cfi_adjust_cfa_offset (8)
	movl $-1, 4(%esp)
	leal 4(%esp), %edx
	movl %edx, (%esp)
	pushl %ecx		# Address of the register structure
	cfi_adjust_cfa_offset (4)
	movl 40(%esp), %ecx	# Load return address
	movl 36(%esp), %edx	# Copy args pushed by PLT in register.  Note
	movl 32(%esp), %eax	# that `fixup' takes its parameters in regs.
	call _dl_profile_fixup	# Call resolver.
	cfi_adjust_cfa_offset (-8)
	movl (%esp), %edx
	testl %edx, %edx
	jns 1f
	movl 4(%esp), %edx	# Get register content back.
	movl %eax, %ecx		# Store the function address.
	movl 12(%esp), %eax	# Get register content back.
	# Adjust stack: PLT1 + PLT2 + %esp + %ebp + %eax + %ecx + %edx
	# + free.
	addl $32, %esp
	cfi_adjust_cfa_offset (-32)
	jmp *%ecx		# Jump to function address.
	cfi_endproc
	.size _dl_runtime_profile_shstk, .-_dl_runtime_profile_shstk

	.globl _dl_runtime_profile
	.type _dl_runtime_profile, @function
	cfi_startproc
	.align 16
_dl_runtime_profile:
	cfi_adjust_cfa_offset (8)
	_CET_ENDBR
	pushl %esp
	cfi_adjust_cfa_offset (4)
	addl $8, (%esp)		# Account for the pushed PLT data
	pushl %ebp
	cfi_adjust_cfa_offset (4)
	pushl %eax		# Preserve registers otherwise clobbered.
	cfi_adjust_cfa_offset (4)
	pushl %ecx
	cfi_adjust_cfa_offset (4)
	pushl %edx
	cfi_adjust_cfa_offset (4)
	movl %esp, %ecx
	subl $8, %esp
	cfi_adjust_cfa_offset (8)
	movl $-1, 4(%esp)
	leal 4(%esp), %edx
	movl %edx, (%esp)
	pushl %ecx		# Address of the register structure
	cfi_adjust_cfa_offset (4)
	movl 40(%esp), %ecx	# Load return address
	movl 36(%esp), %edx	# Copy args pushed by PLT in register.  Note
	movl 32(%esp), %eax	# that `fixup' takes its parameters in regs.
	call _dl_profile_fixup	# Call resolver.
	cfi_adjust_cfa_offset (-8)
	movl (%esp), %edx
	testl %edx, %edx
	jns 1f
	popl %edx
	cfi_adjust_cfa_offset (-4)
	popl %edx		# Get register content back.
	cfi_adjust_cfa_offset (-4)
	movl (%esp), %ecx
	movl %eax, (%esp)	# Store the function address.
	movl 4(%esp), %eax
	ret $20			# Jump to function address.

	/*
	    +32     return address
	    +28     PLT1
	    +24     PLT2
	    +20     %esp
	    +16     %ebp
	    +12     %eax
	    +8      %ecx
	    +4      %edx
	   %esp     free
	*/
	cfi_adjust_cfa_offset (8)
1:	movl %ebx, (%esp)
	cfi_rel_offset (ebx, 0)
	movl %edx, %ebx		# This is the frame buffer size
	pushl %edi
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (edi, 0)
	pushl %esi
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (esi, 0)
	leal 44(%esp), %esi
	movl %ebx, %ecx
	orl $4, %ebx		# Increase frame size if necessary to align
				# stack for the function call
	andl $~3, %ebx
	movl %esp, %edi
	subl %ebx, %edi
	movl %esp, %ebx
	cfi_def_cfa_register (ebx)
	movl %edi, %esp
	shrl $2, %ecx
	rep
	movsl
	movl (%ebx), %esi
	cfi_restore (esi)
	movl 4(%ebx), %edi
	cfi_restore (edi)
	/*
	   %ebx+40  return address
	   %ebx+36  PLT1
	   %ebx+32  PLT2
	   %ebx+28  %esp
	   %ebx+24  %ebp
	   %ebx+20  %eax
	   %ebx+16  %ecx
	   %ebx+12  %edx
	   %ebx+8   %ebx
	   %ebx+4   free
	   %ebx     free
	   %esp     copied stack frame
	*/
	movl %eax, (%ebx)
	movl 12(%ebx), %edx
	movl 16(%ebx), %ecx
	movl 20(%ebx), %eax
	call *(%ebx)
	movl %ebx, %esp
	cfi_def_cfa_register (esp)
	movl 8(%esp), %ebx
	cfi_restore (ebx)
	/*
	    +40     return address
	    +36     PLT1
	    +32     PLT2
	    +28     %esp
	    +24     %ebp
	    +20     %eax
	    +16     %ecx
	    +12     %edx
	    +8      free
	    +4      free
	   %esp     free
	*/
#if LONG_DOUBLE_SIZE != 12
# error "long double size must be 12 bytes"
#endif
	# Allocate space for La_i86_retval and subtract 12 free bytes.
	subl $(LRV_SIZE - 12), %esp
	cfi_adjust_cfa_offset (LRV_SIZE - 12)
	movl %eax, LRV_EAX_OFFSET(%esp)
	movl %edx, LRV_EDX_OFFSET(%esp)
	fstpt LRV_ST0_OFFSET(%esp)
	fstpt LRV_ST1_OFFSET(%esp)
#ifdef HAVE_MPX_SUPPORT
	bndmov %bnd0, LRV_BND0_OFFSET(%esp)
	bndmov %bnd1, LRV_BND1_OFFSET(%esp)
#else
	.byte 0x66,0x0f,0x1b,0x44,0x24,LRV_BND0_OFFSET
	.byte 0x66,0x0f,0x1b,0x4c,0x24,LRV_BND1_OFFSET
#endif
	pushl %esp
	cfi_adjust_cfa_offset (4)
	# Address of La_i86_regs area.
	leal (LRV_SIZE + 4)(%esp), %ecx
	# PLT2
	movl (LRV_SIZE + 4 + LR_SIZE)(%esp), %eax
	# PLT1
	movl (LRV_SIZE + 4 + LR_SIZE + 4)(%esp), %edx
	call _dl_call_pltexit
	movl LRV_EAX_OFFSET(%esp), %eax
	movl LRV_EDX_OFFSET(%esp), %edx
	fldt LRV_ST1_OFFSET(%esp)
	fldt LRV_ST0_OFFSET(%esp)
#ifdef HAVE_MPX_SUPPORT
	bndmov LRV_BND0_OFFSET(%esp), %bnd0
	bndmov LRV_BND1_OFFSET(%esp), %bnd1
#else
	.byte 0x66,0x0f,0x1a,0x44,0x24,LRV_BND0_OFFSET
	.byte 0x66,0x0f,0x1a,0x4c,0x24,LRV_BND1_OFFSET
#endif
	# Restore stack before return.
	addl $(LRV_SIZE + 4 + LR_SIZE + 4), %esp
	cfi_adjust_cfa_offset (-(LRV_SIZE + 4 + LR_SIZE + 4))
	PRESERVE_BND_REGS_PREFIX
	ret
	cfi_endproc
	.size _dl_runtime_profile, .-_dl_runtime_profile
#endif
