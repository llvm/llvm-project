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

#include "pthreadP.h"
#include <shlib-compat.h>

int
__pthread_spin_lock (pthread_spinlock_t *lock)
{
  unsigned int __tmp;

  asm volatile (
       "1:	lwarx	%0,0,%1" MUTEX_HINT_ACQ "\n"
       "	cmpwi	0,%0,0\n"
       "	bne-	2f\n"
       "	stwcx.	%2,0,%1\n"
       "	bne-	2f\n"
                __ARCH_ACQ_INSTR "\n"
       "	.subsection 1\n"
       "2:	lwzx	%0,0,%1\n"
       "	cmpwi	0,%0,0\n"
       "	bne	2b\n"
       "	b	1b\n"
       "	.previous"
       : "=&r" (__tmp)
       : "r" (lock), "r" (1)
       : "cr0", "memory");
  return 0;
}
versioned_symbol (libc, __pthread_spin_lock, pthread_spin_lock, GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_lock, pthread_spin_lock, GLIBC_2_2);
#endif
