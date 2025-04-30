/* Test CET property note parser.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <cet.h>

	.text
	.p2align 4,,15
	.globl	bar
	.type	bar, @function
/* Since this function doesn't start with ENDBR, it should lead to the
   IBT violation when called indirectly.  */
bar:
	.cfi_startproc
	ret
	.cfi_endproc
	.size	bar, .-bar

#if __SIZEOF_PTRDIFF_T__  == 8
# define ALIGN 3
#elif __SIZEOF_PTRDIFF_T__  == 4
# define ALIGN 2
#endif

/* In NT_GNU_PROPERTY_TYPE_0 note, add a GNU_PROPERTY_STACK_SIZE property
   before the GNU_PROPERTY_X86_FEATURE_1_AND property.  */
	.section ".note.gnu.property", "a"
	.p2align ALIGN
	.long 1f - 0f		/* name length */
	.long 5f - 2f		/* data length */
	.long 5			/* note type */
0:	.asciz "GNU"		/* vendor name */
1:
	.p2align ALIGN
2:
	.long 1			/* pr_type.  */
	.long 4f - 3f		/* pr_datasz.  */
3:
#if __SIZEOF_PTRDIFF_T__  == 8
	.long 0x800
	.long 0x800
#else
	.long 0x08000800
#endif
4:
	.p2align ALIGN
5:

	.section	.note.GNU-stack,"",@progbits
