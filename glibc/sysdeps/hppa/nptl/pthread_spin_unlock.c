/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include "pthreadP.h"
#include <shlib-compat.h>

int
__pthread_spin_unlock (pthread_spinlock_t *lock)
{
  /* CONCURRENCTY NOTES:

     The atomic_exchange_rel synchronizes-with the atomic_exhange_acq in
     pthread_spin_lock.

     On hppa we must not use a plain `stw` to reset the guard lock.  This
     has to do with the kernel compare-and-swap helper that is used to
     implement all of the atomic operations.

     The kernel CAS helper uses its own internal locks and that means that
     to create a true happens-before relationship between any two threads,
     the second thread must observe the internal lock having a value of 0
     (it must attempt to take the lock with ldcw).  This creates the
     ordering required for a second thread to observe the effects of the
     RMW of the kernel CAS helper in any other thread.

     Therefore if a variable is used in an atomic macro it must always be
     manipulated with atomic macros in order for memory ordering rules to
     be preserved.  */
  atomic_exchange_rel (lock, 0);
  return 0;
}
versioned_symbol (libc, __pthread_spin_unlock, pthread_spin_unlock,
                  GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_unlock, pthread_spin_unlock,
               GLIBC_2_2);
#endif
