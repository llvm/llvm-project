/* pre-.hidden libgcc compatibility
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

	.file	"libgcc-compat.S"

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_2_6)

	.symver __ashldi3_v_glibc20,__ashldi3@GLIBC_2.0
	.symver __ashrdi3_v_glibc20,__ashrdi3@GLIBC_2.0
	.symver __lshrdi3_v_glibc20,__lshrdi3@GLIBC_2.0
	.symver __cmpdi2_v_glibc20,__cmpdi2@GLIBC_2.0
	.symver __ucmpdi2_v_glibc20,__ucmpdi2@GLIBC_2.0
#if !defined _SOFT_FLOAT && !defined __NO_FPRS__
	.symver __fixdfdi_v_glibc20,__fixdfdi@GLIBC_2.0
	.symver __fixunsdfdi_v_glibc20,__fixunsdfdi@GLIBC_2.0
	.symver __fixsfdi_v_glibc20,__fixsfdi@GLIBC_2.0
	.symver __fixunssfdi_v_glibc20,__fixunssfdi@GLIBC_2.0
	.symver __floatdidf_v_glibc20,__floatdidf@GLIBC_2.0
	.symver __floatdisf_v_glibc20,__floatdisf@GLIBC_2.0
#endif

#ifdef HAVE_DOT_HIDDEN
	.hidden __ashldi3
	.hidden __ashrdi3
	.hidden __lshrdi3
	.hidden __cmpdi2
	.hidden __ucmpdi2
# if !defined _SOFT_FLOAT && !defined __NO_FPRS__
	.hidden __fixdfdi
	.hidden __fixsfdi
	.hidden __fixunsdfdi
	.hidden __fixunssfdi
	.hidden __floatdidf
	.hidden __floatdisf
# endif
#endif

	.section	".text"

	.align 2
	.globl __ashldi3_v_glibc20
	.type	__ashldi3_v_glibc20,@function
__ashldi3_v_glibc20:
	b __ashldi3@local
.Lfe5:
	.size	__ashldi3_v_glibc20,.Lfe5-__ashldi3_v_glibc20
	.align 2
	.globl __ashrdi3_v_glibc20
	.type	__ashrdi3_v_glibc20,@function
__ashrdi3_v_glibc20:
	b __ashrdi3@local
.Lfe6:
	.size	__ashrdi3_v_glibc20,.Lfe6-__ashrdi3_v_glibc20
	.align 2
	.globl __lshrdi3_v_glibc20
	.type	__lshrdi3_v_glibc20,@function
__lshrdi3_v_glibc20:
	b __lshrdi3@local
.Lfe7:
	.size	__lshrdi3_v_glibc20,.Lfe7-__lshrdi3_v_glibc20
	.align 2
	.globl __cmpdi2_v_glibc20
	.type	__cmpdi2_v_glibc20,@function
__cmpdi2_v_glibc20:
	b __cmpdi2@local
.Lfe8:
	.size	__cmpdi2_v_glibc20,.Lfe8-__cmpdi2_v_glibc20
	.align 2
	.globl __ucmpdi2_v_glibc20
	.type	__ucmpdi2_v_glibc20,@function
__ucmpdi2_v_glibc20:
	b __ucmpdi2@local
.Lfe9:
	.size	__ucmpdi2_v_glibc20,.Lfe9-__ucmpdi2_v_glibc20
#if !defined _SOFT_FLOAT && !defined __NO_FPRS__
	.align 2
	.globl __fixdfdi_v_glibc20
	.type	__fixdfdi_v_glibc20,@function
__fixdfdi_v_glibc20:
	b __fixdfdi@local
.Lfe10:
	.size	__fixdfdi_v_glibc20,.Lfe10-__fixdfdi_v_glibc20
	.align 2
	.globl __fixunsdfdi_v_glibc20
	.type	__fixunsdfdi_v_glibc20,@function
__fixunsdfdi_v_glibc20:
	b __fixunsdfdi@local
.Lfe11:
	.size	__fixunsdfdi_v_glibc20,.Lfe11-__fixunsdfdi_v_glibc20
	.align 2
	.globl __fixsfdi_v_glibc20
	.type	__fixsfdi_v_glibc20,@function
__fixsfdi_v_glibc20:
	b __fixsfdi@local
.Lfe12:
	.size	__fixsfdi_v_glibc20,.Lfe12-__fixsfdi_v_glibc20
	.align 2
	.globl __fixunssfdi_v_glibc20
	.type	__fixunssfdi_v_glibc20,@function
__fixunssfdi_v_glibc20:
	b __fixunssfdi@local
.Lfe13:
	.size	__fixunssfdi_v_glibc20,.Lfe13-__fixunssfdi_v_glibc20
	.align 2
	.globl __floatdidf_v_glibc20
	.type	__floatdidf_v_glibc20,@function
__floatdidf_v_glibc20:
	b __floatdidf@local
.Lfe14:
	.size	__floatdidf_v_glibc20,.Lfe14-__floatdidf_v_glibc20
	.align 2
	.globl __floatdisf_v_glibc20
	.type	__floatdisf_v_glibc20,@function
__floatdisf_v_glibc20:
	b __floatdisf@local
.Lfe15:
	.size	__floatdisf_v_glibc20,.Lfe15-__floatdisf_v_glibc20
#endif

#endif
