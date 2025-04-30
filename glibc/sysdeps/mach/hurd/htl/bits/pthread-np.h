/* Non-portable functions. Hurd on Mach version.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
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

/*
 * Never include this file directly; use <pthread.h> instead.
 */

#ifndef _BITS_PTHREAD_NP_H
#define _BITS_PTHREAD_NP_H	1

/* Same as pthread_cond_wait, but for Hurd-specific cancellation.
   See hurd_thread_cancel.  */
extern int pthread_hurd_cond_wait_np (pthread_cond_t *__restrict __cond,
				      pthread_mutex_t *__restrict __mutex);

/* Same as pthread_cond_timedwait, but for Hurd-specific cancellation.
   See hurd_thread_cancel.  */
extern int pthread_hurd_cond_timedwait_np (pthread_cond_t *__restrict __cond,
					   pthread_mutex_t *__restrict __mutex,
					   const struct timespec *__abstime);

#endif /* bits/pthread-np.h */
