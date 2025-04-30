/* memcmp - compare two memory blocks.  31/64 bit S/390 version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include "asm-syntax.h"
#include <ifunc-memcmp.h>

/* INPUT PARAMETERS
     %r2 = address of first memory area
     %r3 = address of second memory area
     %r4 = number of bytes to compare.  */

       .text

#if HAVE_MEMCMP_Z900_G5
# if defined __s390x__
#  define LTGR	ltgr
#  define AGHI	aghi
#  define BRCTG	brctg
# else
#  define LTGR	ltr
#  define AGHI	ahi
#  define BRCTG	brct
# endif /* ! defined __s390x__  */
ENTRY(MEMCMP_Z900_G5)
# if defined __s390x__
	.machine "z900"
# else
	.machine "g5"
	basr    %r5,0
.L_Z900_G5_16:
#  define Z900_G5_EX_D .L_Z900_G5_15-.L_Z900_G5_16
# endif /* ! defined __s390x__  */
	LTGR    %r4,%r4
	je      .L_Z900_G5_4
	AGHI    %r4,-1
# if defined __s390x__
	srlg    %r1,%r4,8
	larl    %r5,.L_Z900_G5_15
#  define Z900_G5_EX_D 0
# else
	lr	%r1,%r4
	srl	%r1,8
# endif /* ! defined __s390x__  */
	LTGR    %r1,%r1
	jne     .L_Z900_G5_12
.L_Z900_G5_3:
	ex      %r4,Z900_G5_EX_D(%r5)
.L_Z900_G5_4:
	ipm     %r2
# if defined __s390x__
	sllg    %r2,%r2,34
	srag    %r2,%r2,62
# else
	sll     %r2,2
	sra     %r2,30
# endif /* ! defined __s390x__  */
	br      %r14
.L_Z900_G5_12:
	clc     0(256,%r3),0(%r2)
	jne     .L_Z900_G5_4
	la      %r3,256(%r3)
	la      %r2,256(%r2)
	BRCTG   %r1,.L_Z900_G5_12
	j       .L_Z900_G5_3
.L_Z900_G5_15:
	clc     0(1,%r3),0(%r2)
END(MEMCMP_Z900_G5)
# undef LTGR
# undef AGHI
# undef BRCTG
#endif /* HAVE_MEMCMP_Z900_G5  */

#if HAVE_MEMCMP_Z10
ENTRY(MEMCMP_Z10)
	.machine "z10"
	.machinemode "zarch_nohighgprs"
# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__  */
	ltgr    %r4,%r4
	je      .L_Z10_4
	aghi    %r4,-1
	srlg    %r1,%r4,8
	cgijlh  %r1,0,.L_Z10_12
.L_Z10_3:
	exrl    %r4,.L_Z10_15
.L_Z10_4:
	ipm     %r2
	sllg    %r2,%r2,34
	srag    %r2,%r2,62
	br      %r14
.L_Z10_12:
	pfd     1,512(%r3)
	pfd     1,512(%r2)
	clc     0(256,%r3),0(%r2)
	jne     .L_Z10_4
	la      %r3,256(%r3)
	la      %r2,256(%r2)
	brctg   %r1,.L_Z10_12
	j       .L_Z10_3
.L_Z10_15:
	clc     0(1,%r3),0(%r2)
END(MEMCMP_Z10)
#endif /* HAVE_MEMCMP_Z10  */

#if HAVE_MEMCMP_Z196
ENTRY(MEMCMP_Z196)
	.machine "z196"
	.machinemode "zarch_nohighgprs"
# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__  */
	ltgr    %r4,%r4
	je      .L_Z196_4
	aghi    %r4,-1
	srlg    %r1,%r4,8
	ltgr    %r1,%r1
	jne     .L_Z196_2
.L_Z196_3:
	exrl    %r4,.L_Z196_14
.L_Z196_4:
	ipm     %r2
	sllg    %r2,%r2,34
	srag    %r2,%r2,62
	br      %r14
.L_Z196_17:
	la      %r3,256(%r3)
	la      %r2,256(%r2)
	aghi    %r1,-1
	je      .L_Z196_3
.L_Z196_2:
	pfd     1,512(%r3)
	pfd     1,512(%r2)
	clc     0(256,%r3),0(%r2)
	je      .L_Z196_17
	ipm     %r2
	sllg    %r2,%r2,34
	srag    %r2,%r2,62
	br      %r14
.L_Z196_14:
	clc     0(1,%r3),0(%r2)
END(MEMCMP_Z196)
#endif /* HAVE_MEMCMP_Z196  */

#if ! HAVE_MEMCMP_IFUNC
/* If we don't use ifunc, define an alias for memcmp here.
   Otherwise see sysdeps/s390/memcmp.c.  */
strong_alias (MEMCMP_DEFAULT, memcmp)
weak_alias (memcmp, bcmp)
#endif

#if defined SHARED && IS_IN (libc)
/* Defines the internal symbols.
   Compare to libc_hidden_builtin_def (memcmp) in string/memcmp.c.  */
strong_alias (MEMCMP_DEFAULT, __GI_memcmp)
#endif
