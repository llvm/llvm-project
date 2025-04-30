/* Thread-local storage handling in the ELF dynamic linker.  i386 version.
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

	The incoming %eax points to the TLS descriptor, such that
	0(%eax) points to _dl_tlsdesc_return itself, and 4(%eax) holds
	the TP offset of the symbol corresponding to the object
	denoted by the argument.  */

	.hidden _dl_tlsdesc_return
	.global	_dl_tlsdesc_return
	.type	_dl_tlsdesc_return,@function
	cfi_startproc
	.align 16
_dl_tlsdesc_return:
	_CET_ENDBR
	movl	4(%eax), %eax
	ret
	cfi_endproc
	.size	_dl_tlsdesc_return, .-_dl_tlsdesc_return

     /* This function is used for undefined weak TLS symbols, for
	which the base address (i.e., disregarding any addend) should
	resolve to NULL.

	%eax points to the TLS descriptor, such that 0(%eax) points to
	_dl_tlsdesc_undefweak itself, and 4(%eax) holds the addend.
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
	movl	4(%eax), %eax
	subl	%gs:0, %eax
	ret
	cfi_endproc
	.size	_dl_tlsdesc_undefweak, .-_dl_tlsdesc_undefweak

#ifdef SHARED
	.hidden _dl_tlsdesc_dynamic
	.global	_dl_tlsdesc_dynamic
	.type	_dl_tlsdesc_dynamic,@function

     /* This function is used for symbols that need dynamic TLS.

	%eax points to the TLS descriptor, such that 0(%eax) points to
	_dl_tlsdesc_dynamic itself, and 4(%eax) points to a struct
	tlsdesc_dynamic_arg object.  It must return in %eax the offset
	between the thread pointer and the object denoted by the
	argument, without clobbering any registers.

	The assembly code that follows is a rendition of the following
	C code, hand-optimized a little bit.

ptrdiff_t
__attribute__ ((__regparm__ (1)))
_dl_tlsdesc_dynamic (struct tlsdesc *tdp)
{
  struct tlsdesc_dynamic_arg *td = tdp->arg;
  dtv_t *dtv = *(dtv_t **)((char *)__thread_pointer + DTV_OFFSET);
  if (__builtin_expect (td->gen_count <= dtv[0].counter
			&& (dtv[td->tlsinfo.ti_module].pointer.val
			    != TLS_DTV_UNALLOCATED),
			1))
    return dtv[td->tlsinfo.ti_module].pointer.val + td->tlsinfo.ti_offset
      - __thread_pointer;

  return ___tls_get_addr (&td->tlsinfo) - __thread_pointer;
}
*/
	cfi_startproc
	.align 16
_dl_tlsdesc_dynamic:
	_CET_ENDBR
	/* Like all TLS resolvers, preserve call-clobbered registers.
	   We need two scratch regs anyway.  */
	subl	$28, %esp
	cfi_adjust_cfa_offset (28)
	movl	%ecx, 20(%esp)
	movl	%edx, 24(%esp)
	movl	TLSDESC_ARG(%eax), %eax
	movl	%gs:DTV_OFFSET, %edx
	movl	TLSDESC_GEN_COUNT(%eax), %ecx
	cmpl	(%edx), %ecx
	ja	.Lslow
	movl	TLSDESC_MODID(%eax), %ecx
	movl	(%edx,%ecx,8), %edx
	cmpl	$-1, %edx
	je	.Lslow
	movl	TLSDESC_MODOFF(%eax), %eax
	addl	%edx, %eax
.Lret:
	movl	20(%esp), %ecx
	subl	%gs:0, %eax
	movl	24(%esp), %edx
	addl	$28, %esp
	cfi_adjust_cfa_offset (-28)
	ret
	.p2align 4,,7
.Lslow:
	cfi_adjust_cfa_offset (28)
	call	HIDDEN_JUMPTARGET (___tls_get_addr)
	jmp	.Lret
	cfi_endproc
	.size	_dl_tlsdesc_dynamic, .-_dl_tlsdesc_dynamic
#endif /* SHARED */
