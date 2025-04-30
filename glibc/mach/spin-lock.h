/* Definitions of user-visible names for spin locks.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#ifndef _SPIN_LOCK_H
#define _SPIN_LOCK_H

#include <lock-intern.h>	/* This does all the work.  */

typedef __spin_lock_t spin_lock_t;
#define SPIN_LOCK_INITIALIZER	__SPIN_LOCK_INITIALIZER

#define spin_lock_init(lock)	__spin_lock_init (lock)
#define spin_lock(lock)		__spin_lock (lock)
#define spin_try_lock(lock)	__spin_try_lock (lock)
#define spin_unlock(lock)	__spin_unlock (lock)
#define spin_lock_locked(lock)	__spin_lock_locked (lock)

#endif /* spin-lock.h */
