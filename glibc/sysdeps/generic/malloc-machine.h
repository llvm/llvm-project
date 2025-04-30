/* Basic platform-independent macro definitions for mutexes,
   thread-specific data and parameters for malloc.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#ifndef _GENERIC_MALLOC_MACHINE_H
#define _GENERIC_MALLOC_MACHINE_H

#include <atomic.h>

#ifndef atomic_full_barrier
# define atomic_full_barrier() __asm ("" ::: "memory")
#endif

#ifndef atomic_read_barrier
# define atomic_read_barrier() atomic_full_barrier ()
#endif

#ifndef atomic_write_barrier
# define atomic_write_barrier() atomic_full_barrier ()
#endif

#ifndef DEFAULT_TOP_PAD
# define DEFAULT_TOP_PAD 131072
#endif

#endif /* !defined(_GENERIC_MALLOC_MACHINE_H) */
