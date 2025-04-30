/* C11 threads conditional timed wait implementation.
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

#include "thrd_priv.h"

int
cnd_timedwait (cnd_t *restrict cond, mtx_t *restrict mutex,
	       const struct timespec* restrict time_point)
{
  int err_code = __pthread_cond_timedwait ((pthread_cond_t *) cond,
					   (pthread_mutex_t *) mutex,
					   time_point);
  return thrd_err_map (err_code);
}
