/* Libc stubs for pthread functions.  Hurd pthread version.
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

#include <errno.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <shlib-compat.h>
#include <pthread-functions.h>
#include <libc-lock.h>
#include <pt-internal.h>

/* Pointers to the libc functions.  */
struct pthread_functions __libc_pthread_functions attribute_hidden;
int __libc_pthread_functions_init attribute_hidden;


#define FORWARD2(name, rettype, decl, params, defaction) \
rettype									      \
name decl								      \
{									      \
  if (!__libc_pthread_functions_init)					      \
    defaction;								      \
									      \
  return PTHFCT_CALL (ptr_##name, params);				      \
}

/* Same as FORWARD2, only without return.  */
#define FORWARD_NORETURN(name, rettype, decl, params, defaction) \
rettype									      \
name decl								      \
{									      \
  if (!__libc_pthread_functions_init)					      \
    defaction;								      \
									      \
  PTHFCT_CALL (ptr_##name, params);					      \
}

#define FORWARD(name, decl, params, defretval) \
  FORWARD2 (name, int, decl, params, return defretval)

FORWARD (pthread_attr_destroy, (pthread_attr_t *attr), (attr), 0)

FORWARD (pthread_attr_init, (pthread_attr_t *attr), (attr), 0)

FORWARD (pthread_attr_getdetachstate,
	 (const pthread_attr_t *attr, int *detachstate), (attr, detachstate),
	 0)
FORWARD (pthread_attr_setdetachstate, (pthread_attr_t *attr, int detachstate),
	 (attr, detachstate), 0)

FORWARD (pthread_attr_getinheritsched,
	 (const pthread_attr_t *attr, int *inherit), (attr, inherit), 0)
FORWARD (pthread_attr_setinheritsched, (pthread_attr_t *attr, int inherit),
	 (attr, inherit), 0)

FORWARD (pthread_attr_getschedparam,
	 (const pthread_attr_t *attr, struct sched_param *param),
	 (attr, param), 0)
FORWARD (pthread_attr_setschedparam,
	 (pthread_attr_t *attr, const struct sched_param *param),
	 (attr, param), 0)

FORWARD (pthread_attr_getschedpolicy,
	 (const pthread_attr_t *attr, int *policy), (attr, policy), 0)
FORWARD (pthread_attr_setschedpolicy, (pthread_attr_t *attr, int policy),
	 (attr, policy), 0)

FORWARD (pthread_attr_getscope,
	 (const pthread_attr_t *attr, int *scope), (attr, scope), 0)
FORWARD (pthread_attr_setscope, (pthread_attr_t *attr, int scope),
	 (attr, scope), 0)


FORWARD (pthread_condattr_destroy, (pthread_condattr_t *attr), (attr), 0)
FORWARD (pthread_condattr_init, (pthread_condattr_t *attr), (attr), 0)


FORWARD (pthread_cond_broadcast, (pthread_cond_t *cond), (cond), 0)
FORWARD (pthread_cond_destroy, (pthread_cond_t *cond), (cond), 0)
FORWARD (pthread_cond_init,
	 (pthread_cond_t *cond, const pthread_condattr_t *cond_attr),
	 (cond, cond_attr), 0)
FORWARD (pthread_cond_signal, (pthread_cond_t *cond), (cond), 0)
FORWARD (pthread_cond_wait, (pthread_cond_t *cond, pthread_mutex_t *mutex),
	 (cond, mutex), 0)
FORWARD (pthread_cond_timedwait,
	 (pthread_cond_t *cond, pthread_mutex_t *mutex,
	  const struct timespec *abstime), (cond, mutex, abstime), 0)

FORWARD (pthread_equal, (pthread_t thread1, pthread_t thread2),
	 (thread1, thread2), 1)


/* Use an alias to avoid warning, as pthread_exit is declared noreturn.  */
FORWARD_NORETURN (__pthread_exit, void, (void *retval), (retval),
		  exit (EXIT_SUCCESS))
strong_alias (__pthread_exit, pthread_exit);


FORWARD (pthread_getschedparam,
	 (pthread_t target_thread, int *policy, struct sched_param *param),
	 (target_thread, policy, param), 0)
FORWARD (pthread_setschedparam,
	 (pthread_t target_thread, int policy,
	  const struct sched_param *param), (target_thread, policy, param), 0)


FORWARD (pthread_mutex_destroy, (pthread_mutex_t *mutex), (mutex), 0)

FORWARD (pthread_mutex_init,
	 (pthread_mutex_t *mutex, const pthread_mutexattr_t *mutexattr),
	 (mutex, mutexattr), 0)

FORWARD (pthread_mutex_lock, (pthread_mutex_t *mutex), (mutex), 0)

FORWARD (pthread_mutex_unlock, (pthread_mutex_t *mutex), (mutex), 0)


FORWARD2 (pthread_self, pthread_t, (void), (), return 0)


FORWARD (__pthread_setcancelstate, (int state, int *oldstate),
	 (state, oldstate), 0)
strong_alias (__pthread_setcancelstate, pthread_setcancelstate);

FORWARD (pthread_setcanceltype, (int type, int *oldtype), (type, oldtype), 0)

FORWARD2 (__pthread_get_cleanup_stack, struct __pthread_cancelation_handler **,
	  (void), (), return &__pthread_cleanup_stack);
