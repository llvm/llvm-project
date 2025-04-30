/* Dynamic package initialization data structures.  Generic version.
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

#ifndef _BITS_TYPES_STRUCT___PTHREAD_ONCE_H
#define _BITS_TYPES_STRUCT___PTHREAD_ONCE_H	1

#include <bits/types/__pthread_spinlock_t.h>

struct __pthread_once
{
  int __run;
  __pthread_spinlock_t __lock;
};

#define __PTHREAD_ONCE_INIT \
	0, __PTHREAD_SPIN_LOCK_INITIALIZER

#endif /* bits/types/struct___pthread_once.h */
