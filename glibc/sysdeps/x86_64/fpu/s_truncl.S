/* Truncate long double value.
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

#include <libm-alias-ldouble.h>
#include <machine/asm.h>

ENTRY(__truncl)
	fldt	8(%rsp)
	fnstenv	-28(%rsp)
	movl	$0xc00, %edx
	orl	-28(%rsp), %edx
	movl	%edx, -32(%rsp)
	fldcw	-32(%rsp)
	frndint
	fnstsw
	andl	$0x1, %eax
	orl	%eax, -24(%rsp)
	fldenv	-28(%rsp)
	ret
END(__truncl)
libm_alias_ldouble (__trunc, trunc)
