/* Search a character in a block of memory.  31/64 bit S/390 version.
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

/* INPUT PARAMETERS
     %r2 = address to memory area
     %r3 = character to find
     %r4 = number of bytes to search.  */

#include <ifunc-memchr.h>
#include "sysdep.h"
#include "asm-syntax.h"

#if HAVE_MEMCHR_Z900_G5
# if defined __s390x__
#  define SLGR	slgr
#  define LGHI	lghi
#  define NGR	ngr
#  define LGR	lgr
# else
#  define SLGR	slr
#  define LGHI	lhi
#  define NGR	nr
#  define LGR	lr
# endif /* ! defined __s390x__  */

	.text
ENTRY(MEMCHR_Z900_G5)
	LGHI  %r0,0xff
	NGR   %r0,%r3
	LGR   %r1,%r2
# if ! defined __s390x__
	tmlh  %r4,32768
	jo    3f		/* Jump away if n >= 0x80000000  */
# endif
	la    %r2,0(%r4,%r1)
0:	srst  %r2,%r1
	jo    0b
	brc   13,1f
	SLGR  %r2,%r2
1:	br    %r14
# if ! defined __s390x__
	/* On s390 (31bit), the pointer to the first byte after s (stored in
	   r2) always wraps around with n >= 0x80000000 and can lead to stop
	   searching before end of s.  Thus just use r2=0 in this case.
	   If r2 < r1, the srst instruction stops searching with cc=2 "not
	   found" when wrapping around from top address to zero.  */
3:	SLGR  %r2,%r2
	j     0b
# endif
END(MEMCHR_Z900_G5)

# if ! HAVE_MEMCHR_IFUNC
strong_alias (MEMCHR_Z900_G5, __memchr)
weak_alias (__memchr, memchr)
# endif

# if defined SHARED && IS_IN (libc)
strong_alias (MEMCHR_Z900_G5, __GI_memchr)
# endif
#endif
