/* Stack-aligning implementation of __tls_get_addr.  x86-64 version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#ifdef SHARED

# include <sysdep.h>
# include "tlsdesc.h"
# include "rtld-offsets.h"

/* See __tls_get_addr and __tls_get_addr_slow in dl-tls.c.  This function
   call __tls_get_addr_slow on both slow paths.  It realigns the stack
   before the call to work around GCC PR58066.  */

ENTRY (__tls_get_addr)
	mov 	%fs:DTV_OFFSET, %RDX_LP
	mov	GL_TLS_GENERATION_OFFSET+_rtld_local(%rip), %RAX_LP
	/* GL(dl_tls_generation) == dtv[0].counter */
	cmp	%RAX_LP, (%rdx)
	jne	1f
	mov	TI_MODULE_OFFSET(%rdi), %RAX_LP
	/* dtv[ti->ti_module] */
# ifdef __LP64__
	salq	$4, %rax
	movq	(%rdx,%rax), %rax
# else
	movl	(%rdx,%rax, 8), %eax
# endif
	cmp	$-1, %RAX_LP
	je	1f
	add	TI_OFFSET_OFFSET(%rdi), %RAX_LP
	ret
1:
	/* On the slow path, align the stack.  */
	pushq	%rbp
	cfi_def_cfa_offset (16)
	cfi_offset (%rbp, -16)
	mov	%RSP_LP, %RBP_LP
	cfi_def_cfa_register (%rbp)
	and	$-16, %RSP_LP
	call	__tls_get_addr_slow
	mov	%RBP_LP, %RSP_LP
	popq	%rbp
	cfi_def_cfa (%rsp, 8)
	ret
END (__tls_get_addr)
#endif /* SHARED */
