/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Paul Mackerras <paulus@au.ibm.com>, 2003.

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

#include <errno.h>
#include "pthreadP.h"
#include <shlib-compat.h>

int
__pthread_spin_trylock (pthread_spinlock_t *lock)
{
  unsigned int old;
  int err = EBUSY;

  asm ("1:	lwarx	%0,0,%2" MUTEX_HINT_ACQ "\n"
       "	cmpwi	0,%0,0\n"
       "	bne	2f\n"
       "	stwcx.	%3,0,%2\n"
       "	bne-	1b\n"
       "	li	%1,0\n"
                __ARCH_ACQ_INSTR "\n"
       "2:	"
       : "=&r" (old), "=&r" (err)
       : "r" (lock), "r" (1), "1" (err)
       : "cr0", "memory");

  return err;
}
versioned_symbol (libc, __pthread_spin_trylock, pthread_spin_trylock,
		  GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_trylock, pthread_spin_trylock,
	       GLIBC_2_2);
#endif
