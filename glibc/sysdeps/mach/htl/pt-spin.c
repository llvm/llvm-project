/* Spin locks.  Mach version.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <machine-lock.h>

/* In glibc.  */
extern void __spin_lock_solid (__spin_lock_t *lock);

/* Lock the spin lock object LOCK.  If the lock is held by another
   thread spin until it becomes available.  */
int
_pthread_spin_lock (__spin_lock_t *lock)
{
  __spin_lock_solid (lock);
  return 0;
}
