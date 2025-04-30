/* memcpy - copy a block from source to destination.  31/64 bit S/390 version.
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
#include <ifunc-memcpy.h>

/* INPUT PARAMETERS
     %r2 = address of destination memory area
     %r3 = address of source memory area
     %r4 = number of bytes to copy.  */

       .text

#if defined __s390x__
# define LTGR	ltgr
# define CGHI	cghi
# define LGR	lgr
# define AGHI	aghi
# define BRCTG	brctg
#else
# define LTGR	ltr
# define CGHI	chi
# define LGR	lr
# define AGHI	ahi
# define BRCTG	brct
#endif /* ! defined __s390x__  */

#if HAVE_MEMCPY_Z900_G5
ENTRY(MEMPCPY_Z900_G5)
# if defined __s390x__
	.machine "z900"
# else
	.machine "g5"
# endif /* ! defined __s390x__  */
	LGR     %r1,%r2             # Use as dest
	la      %r2,0(%r4,%r2)      # Return dest + n
	j	.L_Z900_G5_start
END(MEMPCPY_Z900_G5)

ENTRY(MEMCPY_Z900_G5)
# if defined __s390x__
	.machine "z900"
# else
	.machine "g5"
# endif /* ! defined __s390x__  */
	LGR     %r1,%r2             # r1: Use as dest ; r2: Return dest
.L_Z900_G5_start:
	LTGR    %r4,%r4
	je      .L_Z900_G5_4
	AGHI    %r4,-1
# if defined __s390x__
	srlg	%r5,%r4,8
# else
	lr	%r5,%r4
	srl	%r5,8
# endif /* ! defined __s390x__  */
	LTGR    %r5,%r5
	jne     .L_Z900_G5_13
.L_Z900_G5_3:
# if defined __s390x__
	larl    %r5,.L_Z900_G5_15
#  define Z900_G5_EX_D 0
# else
	basr    %r5,0
.L_Z900_G5_14:
#  define Z900_G5_EX_D .L_Z900_G5_15-.L_Z900_G5_14
# endif /* ! defined __s390x__  */
	ex      %r4,Z900_G5_EX_D(%r5)
.L_Z900_G5_4:
	br      %r14
.L_Z900_G5_13:
	CGHI	%r5,4096            # Switch to mvcle for copies >1MB
	jh      __memcpy_mvcle
.L_Z900_G5_12:
	mvc     0(256,%r1),0(%r3)
	la      %r1,256(%r1)
	la      %r3,256(%r3)
	BRCTG   %r5,.L_Z900_G5_12
	j       .L_Z900_G5_3
.L_Z900_G5_15:
	mvc     0(1,%r1),0(%r3)
END(MEMCPY_Z900_G5)
#endif /* HAVE_MEMCPY_Z900_G5  */

ENTRY(__memcpy_mvcle)
	# Using as standalone function will result in unexpected
	# results since the length field is incremented by 1 in order to
	# compensate the changes already done in the functions above.
	LGR     %r0,%r2             # backup return dest [ + n ]
	AGHI    %r4,1               # length + 1
	LGR     %r5,%r4             # source length
	LGR     %r4,%r3             # source address
	LGR     %r2,%r1             # destination address
	LGR     %r3,%r5             # destination length = source length
.L_MVCLE_1:
	mvcle   %r2,%r4,0           # thats it, MVCLE is your friend
	jo      .L_MVCLE_1
	LGR     %r2,%r0             # return destination address
	br      %r14
END(__memcpy_mvcle)

#undef LTGR
#undef CGHI
#undef LGR
#undef AGHI
#undef BRCTG

#if HAVE_MEMCPY_Z10
ENTRY(MEMPCPY_Z10)
	.machine "z10"
	.machinemode "zarch_nohighgprs"
	lgr     %r1,%r2         # Use as dest
	la      %r2,0(%r4,%r2)  # Return dest + n
	j	.L_Z10_start
END(MEMPCPY_Z10)

ENTRY(MEMCPY_Z10)
	.machine "z10"
	.machinemode "zarch_nohighgprs"
	lgr     %r1,%r2         # r1: Use as dest ; r2: Return dest
.L_Z10_start:
# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__  */
	cgije   %r4,0,.L_Z10_4
	aghi    %r4,-1
	srlg    %r5,%r4,8
	cgijlh  %r5,0,.L_Z10_13
.L_Z10_3:
	exrl    %r4,.L_Z10_15
.L_Z10_4:
	br      %r14
.L_Z10_13:
	cgfi    %r5,65535	# Switch to mvcle for copies >16MB
	jh      __memcpy_mvcle
.L_Z10_12:
	pfd     1,768(%r3)
	pfd     2,768(%r1)
	mvc     0(256,%r1),0(%r3)
	la      %r1,256(%r1)
	la      %r3,256(%r3)
	brctg   %r5,.L_Z10_12
	j       .L_Z10_3
.L_Z10_15:
	mvc     0(1,%r1),0(%r3)
END(MEMCPY_Z10)
#endif /* HAVE_MEMCPY_Z10  */

#if HAVE_MEMCPY_Z196
ENTRY(MEMPCPY_Z196)
	.machine "z196"
	.machinemode "zarch_nohighgprs"
	lgr     %r1,%r2         # Use as dest
	la      %r2,0(%r4,%r2)  # Return dest + n
	j	.L_Z196_start
END(MEMPCPY_Z196)

ENTRY(MEMCPY_Z196)
	.machine "z196"
	.machinemode "zarch_nohighgprs"
	lgr     %r1,%r2         # r1: Use as dest ; r2: Return dest
.L_Z196_start:
# if !defined __s390x__
	llgfr	%r4,%r4
# endif /* !defined __s390x__  */
	ltgr    %r4,%r4
	je      .L_Z196_4
.L_Z196_start2:
	aghi    %r4,-1
	risbg	%r5,%r4,8,128+63,56 # r0 = r5 / 256
	jne     .L_Z196_5
.L_Z196_3:
	exrl    %r4,.L_Z196_14
.L_Z196_4:
	br      %r14
.L_Z196_5:
	cgfi	%r5,255		# Switch to loop with pfd for copies >=64kB
	jh	.L_Z196_6
.L_Z196_2:
	mvc     0(256,%r1),0(%r3)
	aghi    %r5,-1
	la      %r1,256(%r1)
	la      %r3,256(%r3)
	jne     .L_Z196_2
	j       .L_Z196_3
.L_Z196_6:
	cgfi    %r5,262144      # Switch to mvcle for copies >64MB
	jh      __memcpy_mvcle
.L_Z196_7:
	pfd     1,1024(%r3)
	pfd     2,1024(%r1)
	mvc     0(256,%r1),0(%r3)
	aghi    %r5,-1
	la      %r1,256(%r1)
	la      %r3,256(%r3)
	jne     .L_Z196_7
	j       .L_Z196_3
.L_Z196_14:
	mvc     0(1,%r1),0(%r3)
END(MEMCPY_Z196)
#endif /* HAVE_MEMCPY_Z196  */

#if HAVE_MEMMOVE_Z13
ENTRY(MEMMOVE_Z13)
	.machine "z13"
	.machinemode "zarch_nohighgprs"
# if !defined __s390x__
	/* Note: The 31bit dst and src pointers are prefixed with zeroes.  */
	llgfr	%r4,%r4
	llgfr	%r3,%r3
	llgfr	%r2,%r2
# endif /* !defined __s390x__ */
	sgrk	%r0,%r2,%r3
	clgijh	%r4,16,.L_MEMMOVE_Z13_LARGE
	aghik	%r5,%r4,-1
.L_MEMMOVE_Z13_SMALL:
	jl .L_MEMMOVE_Z13_END		/* Jump away if len was zero.  */
	/* Store up to 16 bytes with vll/vstl which needs the index
	   instead of lengths.  */
	vll	%v16,%r5,0(%r3)
	vstl	%v16,%r5,0(%r2)
.L_MEMMOVE_Z13_END:
	br      %r14
.L_MEMMOVE_Z13_LARGE:
	lgr     %r1,%r2			/* For memcpy: r1: Use as dest ;
					   r2: Return dest  */
	/* The unsigned comparison (dst - src >= len) determines if we can
	   execute the forward case with memcpy.  */
#if ! HAVE_MEMCPY_Z196
# error The z13 variant of memmove needs the z196 variant of memcpy!
#endif
	clgrjhe %r0,%r4,.L_Z196_start2
	risbgn	%r5,%r4,4,128+63,60	/* r5 = r4 / 16  */
	aghi	%r4,-16
	clgijhe	%r5,8,.L_MEMMOVE_Z13_LARGE_64B
.L_MEMMOVE_Z13_LARGE_16B_LOOP:
	/* Store at least 16 bytes with vl/vst. The number of 16byte blocks
	   is stored in r5.  */
	vl	%v16,0(%r4,%r3)
	vst	%v16,0(%r4,%r2)
	aghi	%r4,-16
	brctg	%r5,.L_MEMMOVE_Z13_LARGE_16B_LOOP
	aghik	%r5,%r4,15
	j	.L_MEMMOVE_Z13_SMALL
.L_MEMMOVE_Z13_LARGE_64B:
	/* Store at least 128 bytes with 4x vl/vst. The number of 64byte blocks
	   will be stored in r0.  */
	aghi	%r4,-48
	srlg	%r0,%r5,2		/* r5 = %r0 / 4
					   => Number of 64byte blocks.  */
.L_MEMMOVE_Z13_LARGE_64B_LOOP:
	vl	%v20,48(%r4,%r3)
	vl	%v19,32(%r4,%r3)
	vl	%v18,16(%r4,%r3)
	vl	%v17,0(%r4,%r3)
	vst	%v20,48(%r4,%r2)
	vst	%v19,32(%r4,%r2)
	vst	%v18,16(%r4,%r2)
	vst	%v17,0(%r4,%r2)
	aghi	%r4,-64
	brctg	%r0,.L_MEMMOVE_Z13_LARGE_64B_LOOP
	aghi	%r4,48
	/* Recalculate the number of 16byte blocks.  */
	risbg	%r5,%r5,62,128+63,0	/* r5 = r5 & 3
					   => Remaining 16byte blocks.  */
	jne	.L_MEMMOVE_Z13_LARGE_16B_LOOP
	aghik	%r5,%r4,15
	j	.L_MEMMOVE_Z13_SMALL
END(MEMMOVE_Z13)
#endif /* HAVE_MEMMOVE_Z13  */

#if HAVE_MEMMOVE_ARCH13
ENTRY(MEMMOVE_ARCH13)
	.machine "arch13"
	.machinemode "zarch_nohighgprs"
# if ! defined __s390x__
	/* Note: The 31bit dst and src pointers are prefixed with zeroes.  */
	llgfr	%r4,%r4
	llgfr	%r3,%r3
	llgfr	%r2,%r2
# endif /* ! defined __s390x__ */
	sgrk	%r5,%r2,%r3
	aghik	%r0,%r4,-1	/* Both vstl and mvcrl needs highest index.  */
	clgijh	%r4,16,.L_MEMMOVE_ARCH13_LARGE
.L_MEMMOVE_ARCH13_SMALL:
	jl .L_MEMMOVE_ARCH13_END /* Return if len was zero (cc of aghik).  */
	/* Store up to 16 bytes with vll/vstl (needs highest index).  */
	vll	%v16,%r0,0(%r3)
	vstl	%v16,%r0,0(%r2)
.L_MEMMOVE_ARCH13_END:
	br      %r14
.L_MEMMOVE_ARCH13_LARGE:
	lgr     %r1,%r2	/* For memcpy: r1: Use as dest ; r2: Return dest  */
	/* The unsigned comparison (dst - src >= len) determines if we can
	   execute the forward case with memcpy.  */
#if ! HAVE_MEMCPY_Z196
# error The arch13 variant of memmove needs the z196 variant of memcpy!
#endif
	/* Backward case.  */
	clgrjhe %r5,%r4,.L_Z196_start2
	clgijh	%r0,255,.L_MEMMOVE_ARCH13_LARGER_256B
	/* Move up to 256bytes with mvcrl (move right to left).  */
	mvcrl	0(%r1),0(%r3)	/* Move (r0 + 1) bytes from r3 to r1.  */
	br      %r14
.L_MEMMOVE_ARCH13_LARGER_256B:
	/* First move the "remaining" block of up to 256 bytes at the end of
	   src/dst buffers.  Then move blocks of 256bytes in a loop starting
	   with the block at the end.
	   (If src/dst pointers are aligned e.g. to 256 bytes, then the pointers
	   passed to mvcrl instructions are aligned, too)  */
	risbgn	%r5,%r0,8,128+63,56	/* r5 = r0 / 256  */
	risbgn	%r0,%r0,56,128+63,0	/* r0 = r0 & 0xFF  */
	slgr	%r4,%r0
	lay	%r1,-1(%r4,%r1)
	lay	%r3,-1(%r4,%r3)
	mvcrl	0(%r1),0(%r3)	/* Move (r0 + 1) bytes from r3 to r1.  */
	lghi	%r0,255		/* Always copy 256 bytes in the loop below!  */
.L_MEMMOVE_ARCH13_LARGE_256B_LOOP:
	aghi	%r1,-256
	aghi	%r3,-256
	mvcrl	0(%r1),0(%r3)	/* Move (r0 + 1) bytes from r3 to r1.  */
	brctg	%r5,.L_MEMMOVE_ARCH13_LARGE_256B_LOOP
	br      %r14
END(MEMMOVE_ARCH13)
#endif /* HAVE_MEMMOVE_ARCH13  */

#if ! HAVE_MEMCPY_IFUNC
/* If we don't use ifunc, define an alias for mem[p]cpy here.
   Otherwise see sysdeps/s390/mem[p]cpy.c.  */
strong_alias (MEMCPY_DEFAULT, memcpy)
strong_alias (MEMPCPY_DEFAULT, __mempcpy)
weak_alias (__mempcpy, mempcpy)
#endif

#if ! HAVE_MEMMOVE_IFUNC
/* If we don't use ifunc, define an alias for memmove here.
   Otherwise see sysdeps/s390/memmove.c.  */
# if ! HAVE_MEMMOVE_C
/* If the c variant is needed, then sysdeps/s390/memmove-c.c
   defines memmove.
   Otherwise MEMMOVE_DEFAULT is implemented here and we have to define it.  */
strong_alias (MEMMOVE_DEFAULT, memmove)
# endif
#endif

#if defined SHARED && IS_IN (libc)
/* Defines the internal symbols.
   Compare to libc_hidden_[builtin_]def (mem[p]cpy) in string/mem[p]cpy.c.  */
strong_alias (MEMCPY_DEFAULT, __GI_memcpy)
strong_alias (MEMPCPY_DEFAULT, __GI_mempcpy)
strong_alias (MEMPCPY_DEFAULT, __GI___mempcpy)
# if ! HAVE_MEMMOVE_C
/* If the c variant is needed, then sysdeps/s390/memmove-c.c
   defines the internal symbol.
   Otherwise MEMMOVE_DEFAULT is implemented here and we have to define it.  */
strong_alias (MEMMOVE_DEFAULT, __GI_memmove)
# endif
#endif
