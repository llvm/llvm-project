/* Set a block of memory to some byte value.  31/64 bit S/390 version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
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
#include <ifunc-memset.h>

/* INPUT PARAMETERS - MEMSET
     %r2 = address of memory area
     %r3 = byte to fill memory with
     %r4 = number of bytes to fill.

   INPUT PARAMETERS - BZERO
     %r2 = address of memory area
     %r3 = number of bytes to fill.  */

       .text

#if HAVE_MEMSET_Z900_G5
# if defined __s390x__
#  define LTGR	ltgr
#  define CGHI	cghi
#  define LGR	lgr
#  define AGHI	aghi
#  define BRCTG	brctg
# else
#  define LTGR	ltr
#  define CGHI	chi
#  define LGR	lr
#  define AGHI	ahi
#  define BRCTG	brct
# endif /* ! defined __s390x__  */

ENTRY(BZERO_Z900_G5)
	LGR	%r4,%r3
	xr	%r3,%r3
	j	.L_Z900_G5_start
END(BZERO_Z900_G5)

ENTRY(MEMSET_Z900_G5)
.L_Z900_G5_start:
#if defined __s390x__
	.machine "z900"
#else
	.machine "g5"
#endif /* ! defined __s390x__  */
	LTGR    %r4,%r4
	je      .L_Z900_G5_4
	stc     %r3,0(%r2)
	CGHI    %r4,1
	LGR     %r1,%r2
	je      .L_Z900_G5_4
	AGHI    %r4,-2
#if defined __s390x__
	larl    %r5,.L_Z900_G5_18
	srlg    %r3,%r4,8
# define Z900_G5_EX_D 0
#else
	basr    %r5,0
.L_Z900_G5_19:
# define Z900_G5_EX_D .L_Z900_G5_18-.L_Z900_G5_19
	lr      %r3,%r4
	srl     %r3,8
#endif /* ! defined __s390x__  */
	LTGR    %r3,%r3
	jne     .L_Z900_G5_14
.L_Z900_G5_3:
	ex      %r4,Z900_G5_EX_D(%r5)
.L_Z900_G5_4:
	br      %r14
.L_Z900_G5_14:
	mvc     1(256,%r1),0(%r1)
	la      %r1,256(%r1)
	BRCTG   %r3,.L_Z900_G5_14
	j       .L_Z900_G5_3
.L_Z900_G5_18:
	mvc     1(1,%r1),0(%r1)
END(MEMSET_Z900_G5)
# undef LTGR
# undef CGHI
# undef LGR
# undef AGHI
# undef BRCTG
#endif /*  HAVE_MEMSET_Z900_G5  */

#if HAVE_MEMSET_Z10
ENTRY(BZERO_Z10)
	.machine "z10"
	.machinemode "zarch_nohighgprs"
	lgr	%r4,%r3
	xr	%r3,%r3
	j	.L_Z10_start
END(BZERO_Z10)

ENTRY(MEMSET_Z10)
.L_Z10_start:
	.machine "z10"
	.machinemode "zarch_nohighgprs"
# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__  */
	cgije   %r4,0,.L_Z10_4
	stc     %r3,0(%r2)
	lgr     %r1,%r2
	cgije   %r4,1,.L_Z10_4
	aghi    %r4,-2
	srlg    %r5,%r4,8
	cgijlh  %r5,0,.L_Z10_15
.L_Z10_3:
	exrl    %r4,.L_Z10_18
.L_Z10_4:
	br      %r14
.L_Z10_15:
	cgfi	%r5,163840          # Switch to mvcle for >40MB
	jh	__memset_mvcle
.L_Z10_14:
	pfd     2,1024(%r1)
	mvc     1(256,%r1),0(%r1)
	la      %r1,256(%r1)
	brctg   %r5,.L_Z10_14
	j       .L_Z10_3
.L_Z10_18:
	mvc     1(1,%r1),0(%r1)
END(MEMSET_Z10)
#endif /* HAVE_MEMSET_Z10  */

#if HAVE_MEMSET_Z196
ENTRY(BZERO_Z196)
	.machine "z196"
	.machinemode "zarch_nohighgprs"
	lgr	%r4,%r3
	xr	%r3,%r3
	j	.L_Z196_start
END(BZERO_Z196)

ENTRY(MEMSET_Z196)
.L_Z196_start:
	.machine "z196"
	.machinemode "zarch_nohighgprs"
# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__  */
	clgfi	%r4,1
	jl	.L_Z196_4	    # n == 0
	stc     %r3,0(%r2)
	je      .L_Z196_4	    # n == 1
	aghi	%r4,-2
	lgr     %r1,%r2
	risbg	%r5,%r4,8,128+63,56 # r5 = n / 256
	jne     .L_Z196_1	    # Jump away if r5 != 0
.L_Z196_3:
	exrl    %r4,.L_Z196_17
.L_Z196_4:
	br      %r14
.L_Z196_1:
	cgfi	%r5,1048576
	jh	__memset_mvcle	    # Switch to mvcle for >256MB
.L_Z196_2:
	pfd     2,1024(%r1)
	mvc     1(255,%r1),0(%r1)
	aghi    %r5,-1
	la      %r1,256(%r1)
	stc     %r3,0(%r1)
	jne     .L_Z196_2
	j       .L_Z196_3
.L_Z196_17:
	mvc     1(1,%r1),0(%r1)
END(MEMSET_Z196)
#endif /* HAVE_MEMSET_Z196  */

#if HAVE_MEMSET_MVCLE
ENTRY(__memset_mvcle)
	aghi	%r4,2               # take back the change done by the caller
	lgr	%r0,%r2		    # save source address
	lgr	%r1,%r3		    # move pad byte to R1
	lgr	%r3,%r4             # move length to r3
	sgr	%r4,%r4		    # no source for MVCLE, only a pad byte
	sgr	%r5,%r5
.L0:	mvcle	%r2,%r4,0(%r1)	    # thats it, MVCLE is your friend
	jo	.L0
	lgr	%r2,%r0		    # return value is source address
.L1:
	br	%r14
END(__memset_mvcle)
#endif /* HAVE_MEMSET_MVCLE  */

#if ! HAVE_MEMSET_IFUNC
/* If we don't use ifunc, define an alias for memset here.
   Otherwise see sysdeps/s390/memset.c.  */
strong_alias (MEMSET_DEFAULT, memset)
/* Same for bzero.  If ifunc is used, see
   sysdeps/s390/bzero.c.  */
strong_alias (BZERO_DEFAULT, __bzero)
weak_alias (__bzero, bzero)
#endif

#if defined SHARED && IS_IN (libc)
/* Defines the internal symbol.
   Compare to libc_hidden_builtin_def (memset) in string/memset.c.  */
strong_alias (MEMSET_DEFAULT, __GI_memset)
#endif
