/* Truncate float value.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <machine/asm.h>
#include <libm-alias-float.h>

ENTRY(__truncf)
	flds	4(%esp)
	subl	$32, %esp
	cfi_adjust_cfa_offset (32)
	fnstenv	4(%esp)
	movl	$0xc00, %edx
	orl	4(%esp), %edx
	movl	%edx, (%esp)
	fldcw	(%esp)
	frndint
	fldenv	4(%esp)
	addl	$32, %esp
	cfi_adjust_cfa_offset (-32)
	ret
END(__truncf)
libm_alias_float (__trunc, trunc)
