/* Declarations of internal pthread functions used by libc.  Hurd version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef _PTHREADP_H
#define _PTHREADP_H	1

#define __PTHREAD_HTL

#include <pthread.h>
#include <link.h>

/* Attribute to indicate thread creation was issued from C11 thrd_create.  */
#define ATTR_C11_THREAD ((void*)(uintptr_t)-1)

extern void __pthread_init_static_tls (struct link_map *) attribute_hidden;

/* These represent the interface used by glibc itself.  */

extern struct __pthread **__pthread_threads;

extern int __pthread_mutex_init (pthread_mutex_t *__mutex, const pthread_mutexattr_t *__attr);
extern int __pthread_mutex_destroy (pthread_mutex_t *__mutex);
extern int __pthread_mutex_lock (pthread_mutex_t *__mutex);
extern int __pthread_mutex_trylock (pthread_mutex_t *_mutex);
extern int __pthread_mutex_timedlock (pthread_mutex_t *__mutex,
     const struct timespec *__abstime);
extern int __pthread_mutex_unlock (pthread_mutex_t *__mutex);
extern int __pthread_mutexattr_init (pthread_mutexattr_t *attr);
extern int __pthread_mutexattr_settype (pthread_mutexattr_t *attr, int kind);

extern int __pthread_cond_init (pthread_cond_t *cond,
				const pthread_condattr_t *cond_attr);
extern int __pthread_cond_signal (pthread_cond_t *cond);
extern int __pthread_cond_broadcast (pthread_cond_t *cond);
extern int __pthread_cond_wait (pthread_cond_t *cond, pthread_mutex_t *mutex);
extern int __pthread_cond_timedwait (pthread_cond_t *cond,
				     pthread_mutex_t *mutex,
				     const struct timespec *abstime);
extern int __pthread_cond_clockwait (pthread_cond_t *cond,
				     pthread_mutex_t *mutex,
				     clockid_t clockid,
				     const struct timespec *abstime)
  __nonnull ((1, 2, 4));
extern int __pthread_cond_destroy (pthread_cond_t *cond);

typedef struct __cthread *__cthread_t;
typedef int __cthread_key_t;
typedef void *	(*__cthread_fn_t)(void *__arg);

__cthread_t __cthread_fork (__cthread_fn_t, void *);
int __pthread_create (pthread_t *newthread,
		      const pthread_attr_t *attr,
		      void *(*start_routine) (void *), void *arg);

void __cthread_detach (__cthread_t);
int __pthread_detach (pthread_t __threadp);
void __pthread_exit (void *value) __attribute__ ((__noreturn__));
int __pthread_join (pthread_t, void **);
int __cthread_keycreate (__cthread_key_t *);
int __cthread_getspecific (__cthread_key_t, void **);
int __cthread_setspecific (__cthread_key_t, void *);
int __pthread_key_create (pthread_key_t *key, void (*destr) (void *));
void *__pthread_getspecific (pthread_key_t key);
int __pthread_setspecific (pthread_key_t key, const void *value);
int __pthread_key_delete (pthread_key_t key);
int __pthread_once (pthread_once_t *once_control, void (*init_routine) (void));

int __pthread_setcancelstate (int state, int *oldstate);

int __pthread_getattr_np (pthread_t, pthread_attr_t *);
int __pthread_attr_getstackaddr (const pthread_attr_t *__restrict __attr,
				 void **__restrict __stackaddr);
int __pthread_attr_setstackaddr (pthread_attr_t *__attr, void *__stackaddr);
int __pthread_attr_getstacksize (const pthread_attr_t *__restrict __attr,
				 size_t *__restrict __stacksize);
int __pthread_attr_setstacksize (pthread_attr_t *__attr, size_t __stacksize);
int __pthread_attr_setstack (pthread_attr_t *__attr, void *__stackaddr,
			     size_t __stacksize);
int __pthread_attr_getstack (const pthread_attr_t *, void **, size_t *);
void __pthread_testcancel (void);

#if IS_IN (libpthread)
hidden_proto (__pthread_create)
hidden_proto (__pthread_detach)
hidden_proto (__pthread_key_create)
hidden_proto (__pthread_getspecific)
hidden_proto (__pthread_setspecific)
hidden_proto (__pthread_mutex_init)
hidden_proto (__pthread_mutex_destroy)
hidden_proto (__pthread_mutex_lock)
hidden_proto (__pthread_mutex_trylock)
hidden_proto (__pthread_mutex_unlock)
hidden_proto (__pthread_mutex_timedlock)
hidden_proto (__pthread_get_cleanup_stack)
#endif

#define ASSERT_TYPE_SIZE(type, size) 					\
  _Static_assert (sizeof (type) == size,				\
		  "sizeof (" #type ") != " #size)

#endif	/* pthreadP.h */
