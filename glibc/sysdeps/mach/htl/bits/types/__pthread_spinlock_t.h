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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _BITS_TYPES___PTHREAD_SPINLOCK_T_H
#define _BITS_TYPES___PTHREAD_SPINLOCK_T_H	1

#include <features.h>

__BEGIN_DECLS

/* The type of a spin lock object.  */
typedef volatile int __pthread_spinlock_t;

/* Initializer for a spin lock object.  */
#define __PTHREAD_SPIN_LOCK_INITIALIZER 0

__END_DECLS

#endif /* bits/types/__pthread_spinlock_t.h */
