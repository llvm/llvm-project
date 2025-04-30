/* Pthread mutex tunable parameters.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#ifndef _PTHREAD_MUTEX_CONF_H
#define _PTHREAD_MUTEX_CONF_H 1

#include <adaptive_spin_count.h>

#if HAVE_TUNABLES
struct mutex_config
{
  int spin_count;
};

extern struct mutex_config __mutex_aconf;
libc_hidden_proto (__mutex_aconf)

extern void __pthread_tunables_init (void) attribute_hidden;
#else
static inline void
__pthread_tunables_init (void)
{
  /* No tunables to initialize.  */
}
#endif

#endif
