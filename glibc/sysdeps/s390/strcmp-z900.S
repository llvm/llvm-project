/* strcmp - compare two string.  64 bit S/390 version.
   This file is part of the GNU C Library.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).

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
     %r2 = address of string 1
     %r3 = address of string 2.  */

#include <ifunc-strcmp.h>
#include "sysdep.h"
#include "asm-syntax.h"

#if HAVE_STRCMP_Z900_G5
# if defined __s390x__
#  define SLGR	slgr
#  define LGHI	lghi
# else
#  define SLGR	slr
#  define LGHI	lhi
# endif /* ! defined __s390x__  */

	.text
ENTRY(STRCMP_Z900_G5)
	SLGR   %r0,%r0
0:	clst  %r2,%r3
	jo    0b
	jp    1f
	jm    2f
	SLGR  %r2,%r2
	br    %r14
1:	LGHI  %r2,1
	br    %r14
2:	LGHI  %r2,-1
	br    %r14
END(STRCMP_Z900_G5)

# if ! HAVE_STRCMP_IFUNC
strong_alias (STRCMP_Z900_G5, strcmp)
# endif

# if defined SHARED && IS_IN (libc)
strong_alias (STRCMP_Z900_G5, __GI_strcmp)
# endif
#endif
