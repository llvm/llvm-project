/* pthread_spin_unlock -- unlock a spin lock.  PowerPC version.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#include "pthreadP.h"
#include <lowlevellock.h>
#include <shlib-compat.h>

int
__pthread_spin_unlock (pthread_spinlock_t *lock)
{
  atomic_store_release (lock, 0);
  return 0;
}
versioned_symbol (libc, __pthread_spin_unlock, pthread_spin_unlock,
                  GLIBC_2_34);

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_unlock, pthread_spin_unlock,
               GLIBC_2_2);
#endif
