/* Internal C11 threads definitions - linux version
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <sysdeps/pthread/thrd_priv.h>

#if __TIMESIZE == 64
# define __cnd_timedwait64 __cnd_timedwait
# define __mtx_timedlock64 __mtx_timedlock
# define __thrd_sleep64 __thrd_sleep
#else
extern int __cnd_timedwait64 (cnd_t *restrict cond, mtx_t *restrict mutex,
                              const struct __timespec64 *restrict time_point);
libc_hidden_proto (__cnd_timedwait64)
extern int __mtx_timedlock64 (mtx_t *restrict mutex,
                              const struct __timespec64 *restrict time_point);
libc_hidden_proto (__mtx_timedlock64)
extern int __thrd_sleep64 (const struct __timespec64 *time_point,
                           struct __timespec64 *remaining);
libc_hidden_proto (__thrd_sleep64)
#endif
