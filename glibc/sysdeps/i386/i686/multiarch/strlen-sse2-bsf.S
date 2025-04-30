/* strlen with SSE2 and BSF
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   Contributed by Intel Corporation.
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

#if defined SHARED && IS_IN (libc)

#include <sysdep.h>

#define CFI_PUSH(REG)						\
  cfi_adjust_cfa_offset (4);					\
  cfi_rel_offset (REG, 0)

#define CFI_POP(REG)						\
  cfi_adjust_cfa_offset (-4);					\
  cfi_restore (REG)

#define PUSH(REG)	pushl REG; CFI_PUSH (REG)
#define POP(REG)	popl REG; CFI_POP (REG)
#define PARMS		4 + 8	/* Preserve ESI and EDI.  */
#define	STR		PARMS
#define ENTRANCE	PUSH (%esi); PUSH (%edi); cfi_remember_state
#define RETURN		POP (%edi); POP (%esi); ret; \
			cfi_restore_state; cfi_remember_state

	.text
ENTRY ( __strlen_sse2_bsf)
	ENTRANCE
	mov	STR(%esp), %edi
	xor	%eax, %eax
	mov	%edi, %ecx
	and	$0x3f, %ecx
	pxor	%xmm0, %xmm0
	cmp	$0x30, %ecx
	ja	L(next)
	movdqu	(%edi), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit_less16)
	mov	%edi, %eax
	and	$-16, %eax
	jmp	L(align16_start)
L(next):

	mov	%edi, %eax
	and	$-16, %eax
	pcmpeqb	(%eax), %xmm0
	mov	$-1, %esi
	sub	%eax, %ecx
	shl	%cl, %esi
	pmovmskb %xmm0, %edx
	and	%esi, %edx
	jnz	L(exit)
L(align16_start):
	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1
	pxor	%xmm2, %xmm2
	pxor	%xmm3, %xmm3
	.p2align 4
L(align16_loop):
	pcmpeqb	16(%eax), %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit16)

	pcmpeqb	32(%eax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit32)

	pcmpeqb	48(%eax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit48)

	pcmpeqb	64(%eax), %xmm3
	pmovmskb %xmm3, %edx
	lea	64(%eax), %eax
	test	%edx, %edx
	jz	L(align16_loop)
L(exit):
	sub	%edi, %eax
L(exit_less16):
	bsf	%edx, %edx
	add	%edx, %eax
	RETURN
L(exit16):
	sub	%edi, %eax
	bsf	%edx, %edx
	add	%edx, %eax
	add	$16, %eax
	RETURN
L(exit32):
	sub	%edi, %eax
	bsf	%edx, %edx
	add	%edx, %eax
	add	$32, %eax
	RETURN
L(exit48):
	sub	%edi, %eax
	bsf	%edx, %edx
	add	%edx, %eax
	add	$48, %eax
	POP (%edi)
	POP (%esi)
	ret

END ( __strlen_sse2_bsf)

#endif
