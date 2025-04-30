/* PLT trampolines.  SH version.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

	.text
	.globl _dl_runtime_resolve
	.type _dl_runtime_resolve, @function
	cfi_startproc
	.align 5
_dl_runtime_resolve:
	mov.l r2,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r3,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r4,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r5,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r6,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r7,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r12,@-r15
	cfi_adjust_cfa_offset (4)
	sts.l macl,@-r15
	cfi_adjust_cfa_offset (4)
	sts.l mach,@-r15
	cfi_adjust_cfa_offset (4)
	movt r3			! Save T flag.
	mov.l r3,@-r15
	cfi_adjust_cfa_offset (4)
#ifdef HAVE_FPU
	sts.l	fpscr,@-r15
	cfi_adjust_cfa_offset (4)
	mov	#8,r3
	swap.w	r3,r3
	lds	r3,fpscr
	fmov.s	fr11,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr10,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr9,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr8,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr7,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr6,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr5,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr4,@-r15
	cfi_adjust_cfa_offset (4)
#endif
	sts.l pr,@-r15
	cfi_adjust_cfa_offset (4)
	tst r0,r0
	bt 1f
	mov r0,r2
1:
	mov r0,r4		! PLT type
	mov r2,r5		! link map address
#ifdef SHARED
	mov.l 2f,r2
	mova 2f,r0
	add r0,r2		! Get GOT address in r2
	mov.l 3f,r0
	add r2,r0
#else
	mov.l 3f,r0
#endif
	jsr @r0			! Call resolver.
	 mov r1,r6		! reloc offset
	lds.l @r15+,pr		! Get register content back.
	cfi_adjust_cfa_offset (-4)
#ifdef HAVE_FPU
	fmov.s	@r15+,fr4
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr5
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr6
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr7
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr8
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr9
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr10
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr11
	cfi_adjust_cfa_offset (-4)
	lds.l	@r15+,fpscr
	cfi_adjust_cfa_offset (-4)
#endif
	mov.l @r15+,r3
	cfi_adjust_cfa_offset (-4)
	shal r3			! Lode T flag.
	lds.l @r15+,mach
	cfi_adjust_cfa_offset (-4)
	lds.l @r15+,macl
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r12
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r7
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r6
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r5
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r4
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r3
	cfi_adjust_cfa_offset (-4)
	jmp @r0			! Jump to function address.
	 mov.l @r15+,r2
	cfi_adjust_cfa_offset (-4)
	.align 2
#ifdef SHARED
2:	.long _GLOBAL_OFFSET_TABLE_
3:	.long _dl_fixup@GOTOFF
#else
3:	.long _dl_fixup
#endif
	cfi_endproc
	.size _dl_runtime_resolve, .-_dl_runtime_resolve


	.globl _dl_runtime_profile
	.type _dl_runtime_profile,@function
	cfi_startproc
	.align 5
_dl_runtime_profile:
	mov.l r12,@-r15
	cfi_adjust_cfa_offset (4)
#ifdef HAVE_FPU
	sts.l	fpscr,@-r15
	cfi_adjust_cfa_offset (4)
	mov	#8,r12
	swap.w	r12,r12
	lds	r12,fpscr
	fmov.s	fr11,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr10,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr9,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr8,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr7,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr6,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr5,@-r15
	cfi_adjust_cfa_offset (4)
	fmov.s	fr4,@-r15
	cfi_adjust_cfa_offset (4)
#else
	add #-36,r15
	cfi_adjust_cfa_offset (36)
#endif
	mov.l r7,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r6,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r5,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r4,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r3,@-r15
	cfi_adjust_cfa_offset (4)
	mov.l r2,@-r15
	cfi_adjust_cfa_offset (4)
	sts.l macl,@-r15
	cfi_adjust_cfa_offset (4)
	sts.l mach,@-r15
	cfi_adjust_cfa_offset (4)
	movt r3			! Save T flag.
	mov.l r3,@-r15
	cfi_adjust_cfa_offset (4)
	sts.l pr,@-r15
	cfi_adjust_cfa_offset (4)
	tst r0,r0
	bt 1f
	mov r0,r2
1:
	mov r0,r4		! PLT type
	mov r2,r5		! link map address
	sts pr,r7		! return address
	add #-24,r15
	cfi_adjust_cfa_offset (24)
	mov #40,r0
	add r15,r0
	mov.l r0,@r15		! Address of the register structure
	mov #-1,r0
	mov.l r0,@(8,r15)
	mov #8,r0
	add r15,r0
	mov.l r0,@(4,r15)
	mov.l r5,@(12,r15)
	mov.l r1,@(16,r15)
#ifdef SHARED
	mov.l 2f,r12
	mova 2f,r0
	add r0,r12		! Get GOT address in r12
	mov.l 3f,r0
	add r12,r0
#else
	mov.l 3f,r0
#endif
	jsr @r0			! Call resolver.
	 mov r1,r6		! reloc offset
	mov.l @(8,r15),r1
	cmp/pz r1
	bt 4f
	add #24,r15
	cfi_adjust_cfa_offset (-24)
	lds.l @r15+,pr		! Get register content back.
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r3
	cfi_adjust_cfa_offset (-4)
	shal r3			! Lode T flag.
	lds.l @r15+,mach
	cfi_adjust_cfa_offset (-4)
	lds.l @r15+,macl
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r2
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r3
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r4
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r5
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r6
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r7
	cfi_adjust_cfa_offset (-4)
#ifdef HAVE_FPU
	fmov.s	@r15+,fr4
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr5
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr6
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr7
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr8
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr9
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr10
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr11
	cfi_adjust_cfa_offset (-4)
	lds.l	@r15+,fpscr
	cfi_adjust_cfa_offset (-4)
#else
	add #36,r15
	cfi_adjust_cfa_offset (-36)
#endif
	jmp @r0			! Jump to function address.
	 mov.l @r15+,r12
	cfi_adjust_cfa_offset (-4)
	.align 2
#ifdef SHARED
2:	.long _GLOBAL_OFFSET_TABLE_
3:	.long _dl_profile_fixup@GOTOFF
#else
3:	.long _dl_profile_fixup
#endif

	cfi_adjust_cfa_offset (104)
4:
	mov #104,r3
	add r15,r3		! Original stack
	mov.l r8,@(20,r15)
	cfi_rel_offset (r8, 20)
	mov r15,r8
	sub r1,r15
	shlr2 r15
	shll2 r15
	mov r15,r4
	shlr2 r1
	tst r1,r1
5:
	bt/s 6f
	 dt r1
	mov.l @r3+,r2
	mov.l r2,@r4
	bra 5b
	 add #4,r4
6:
	mov.l @r8,r12
	mov.l @r12+,r2
	mov.l @r12+,r3
	mov.l @r12+,r4
	mov.l @r12+,r5
	mov.l @r12+,r6
	mov.l @r12+,r7
#ifdef HAVE_FPU
	fmov.s	@r12+,fr4
	fmov.s	@r12+,fr5
	fmov.s	@r12+,fr6
	fmov.s	@r12+,fr7
	fmov.s	@r12+,fr8
	fmov.s	@r12+,fr9
	fmov.s	@r12+,fr10
	fmov.s	@r12+,fr11
	lds.l	@r12+,fpscr
#else
	add #36,r2
#endif
	jsr @r0			! Call function.
	 nop
	mov r8,r15
	mov.l @(12,r15),r4	! link map address
	mov.l @(16,r15),r5	! reloc offset
	mov.l @r15,r6		! input registers
#ifdef HAVE_FPU
	mov #16,r8
	add r15,r8
	fmov.s fr1,@-r8
	fmov.s fr0,@-r8
#else
	mov #8,r8
	add r15,r8
#endif
	mov.l r1,@-r8
	mov.l r0,@-r8
	mov.l @(20,r15),r8
	cfi_restore (r8)
#ifdef SHARED
	mov.l 7f,r12
	mova 7f,r0
	add r0,r12		! Get GOT address in r12
	mov.l 8f,r0
	add r12,r0
#else
	mov.l 8f,r0
#endif
	jsr @r0
	 mov r15,r7		! output registers
	mov.l @r15+,r0
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r1
	cfi_adjust_cfa_offset (-4)
#ifdef HAVE_FPU
	fmov.s @r15+,fr0
	cfi_adjust_cfa_offset (-4)
	fmov.s @r15+,fr1
	cfi_adjust_cfa_offset (-4)
	add #8,r15
	cfi_adjust_cfa_offset (-8)
#else
	add #16,r15
	cfi_adjust_cfa_offset (-16)
#endif
	lds.l @r15+,pr		! Get register content back.
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r3
	cfi_adjust_cfa_offset (-4)
	shal r3			! Lode T flag.
	lds.l @r15+,mach
	cfi_adjust_cfa_offset (-4)
	lds.l @r15+,macl
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r2
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r3
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r4
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r5
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r6
	cfi_adjust_cfa_offset (-4)
	mov.l @r15+,r7
	cfi_adjust_cfa_offset (-4)
#ifdef HAVE_FPU
	fmov.s	@r15+,fr4
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr5
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr6
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr7
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr8
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr9
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr10
	cfi_adjust_cfa_offset (-4)
	fmov.s	@r15+,fr11
	cfi_adjust_cfa_offset (-4)
	lds.l	@r15+,fpscr
	cfi_adjust_cfa_offset (-4)
#else
	add #36,r15
	cfi_adjust_cfa_offset (-36)
#endif
	rts			! Jump to function address.
	 mov.l @r15+,r12
	cfi_adjust_cfa_offset (-4)
	cfi_endproc
	.align 2
#ifdef SHARED
7:	.long _GLOBAL_OFFSET_TABLE_
8:	.long _dl_call_pltexit@GOTOFF
#else
8:	.long _dl_call_pltexit
#endif
	.size _dl_runtime_profile, .-_dl_runtime_profile
