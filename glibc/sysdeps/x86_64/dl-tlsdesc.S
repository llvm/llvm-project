/* Thread-local storage handling in the ELF dynamic linker.  x86_64 version.
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
#include <tls.h>
#include "tlsdesc.h"

	.text

     /* This function is used to compute the TP offset for symbols in
	Static TLS, i.e., whose TP offset is the same for all
	threads.

	The incoming %rax points to the TLS descriptor, such that
	0(%rax) points to _dl_tlsdesc_return itself, and 8(%rax) holds
	the TP offset of the symbol corresponding to the object
	denoted by the argument.  */

	.hidden _dl_tlsdesc_return
	.global	_dl_tlsdesc_return
	.type	_dl_tlsdesc_return,@function
	cfi_startproc
	.align 16
_dl_tlsdesc_return:
	_CET_ENDBR
	movq	8(%rax), %rax
	ret
	cfi_endproc
	.size	_dl_tlsdesc_return, .-_dl_tlsdesc_return

     /* This function is used for undefined weak TLS symbols, for
	which the base address (i.e., disregarding any addend) should
	resolve to NULL.

	%rax points to the TLS descriptor, such that 0(%rax) points to
	_dl_tlsdesc_undefweak itself, and 8(%rax) holds the addend.
	We return the addend minus the TP, such that, when the caller
	adds TP, it gets the addend back.  If that's zero, as usual,
	that's most likely a NULL pointer.  */

	.hidden _dl_tlsdesc_undefweak
	.global	_dl_tlsdesc_undefweak
	.type	_dl_tlsdesc_undefweak,@function
	cfi_startproc
	.align 16
_dl_tlsdesc_undefweak:
	_CET_ENDBR
	movq	8(%rax), %rax
	subq	%fs:0, %rax
	ret
	cfi_endproc
	.size	_dl_tlsdesc_undefweak, .-_dl_tlsdesc_undefweak

#ifdef SHARED
	.hidden _dl_tlsdesc_dynamic
	.global	_dl_tlsdesc_dynamic
	.type	_dl_tlsdesc_dynamic,@function

     /* %rax points to the TLS descriptor, such that 0(%rax) points to
	_dl_tlsdesc_dynamic itself, and 8(%rax) points to a struct
	tlsdesc_dynamic_arg object.  It must return in %rax the offset
	between the thread pointer and the object denoted by the
	argument, without clobbering any registers.

	The assembly code that follows is a rendition of the following
	C code, hand-optimized a little bit.

ptrdiff_t
_dl_tlsdesc_dynamic (register struct tlsdesc *tdp asm ("%rax"))
{
  struct tlsdesc_dynamic_arg *td = tdp->arg;
  dtv_t *dtv = *(dtv_t **)((char *)__thread_pointer + DTV_OFFSET);
  if (__builtin_expect (td->gen_count <= dtv[0].counter
			&& (dtv[td->tlsinfo.ti_module].pointer.val
			    != TLS_DTV_UNALLOCATED),
			1))
    return dtv[td->tlsinfo.ti_module].pointer.val + td->tlsinfo.ti_offset
      - __thread_pointer;

  return __tls_get_addr_internal (&td->tlsinfo) - __thread_pointer;
}
*/
	cfi_startproc
	.align 16
_dl_tlsdesc_dynamic:
	_CET_ENDBR
	/* Preserve call-clobbered registers that we modify.
	   We need two scratch regs anyway.  */
	movq	%rsi, -16(%rsp)
	movq	%fs:DTV_OFFSET, %rsi
	movq	%rdi, -8(%rsp)
	movq	TLSDESC_ARG(%rax), %rdi
	movq	(%rsi), %rax
	cmpq	%rax, TLSDESC_GEN_COUNT(%rdi)
	ja	.Lslow
	movq	TLSDESC_MODID(%rdi), %rax
	salq	$4, %rax
	movq	(%rax,%rsi), %rax
	cmpq	$-1, %rax
	je	.Lslow
	addq	TLSDESC_MODOFF(%rdi), %rax
.Lret:
	movq	-16(%rsp), %rsi
	subq	%fs:0, %rax
	movq	-8(%rsp), %rdi
	ret
.Lslow:
	/* Besides rdi and rsi, saved above, save rdx, rcx, r8, r9,
	   r10 and r11.  Also, align the stack, that's off by 8 bytes.	*/
	subq	$72, %rsp
	cfi_adjust_cfa_offset (72)
	movq	%rdx, 8(%rsp)
	movq	%rcx, 16(%rsp)
	movq	%r8, 24(%rsp)
	movq	%r9, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%r11, 48(%rsp)
	/* %rdi already points to the tlsinfo data structure.  */
	call	HIDDEN_JUMPTARGET (__tls_get_addr)
	movq	8(%rsp), %rdx
	movq	16(%rsp), %rcx
	movq	24(%rsp), %r8
	movq	32(%rsp), %r9
	movq	40(%rsp), %r10
	movq	48(%rsp), %r11
	addq	$72, %rsp
	cfi_adjust_cfa_offset (-72)
	jmp	.Lret
	cfi_endproc
	.size	_dl_tlsdesc_dynamic, .-_dl_tlsdesc_dynamic
#endif /* SHARED */
