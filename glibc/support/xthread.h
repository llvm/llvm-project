/* Support functionality for using threads.
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

#ifndef SUPPORT_THREAD_H
#define SUPPORT_THREAD_H

#include <pthread.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/* Terminate the process (with exit status 0) after SECONDS have
   elapsed, from a helper thread.  The process is terminated with the
   exit function, so atexit handlers are executed.  */
void delayed_exit (int seconds);

/* Terminate the process (with exit status 1) if VALUE is not zero.
   In that case, print a failure message to standard output mentioning
   FUNCTION.  The process is terminated with the exit function, so
   atexit handlers are executed.  */
void xpthread_check_return (const char *function, int value);

/* The following functions call the corresponding libpthread functions
   and terminate the process on error.  */

void xpthread_barrier_init (pthread_barrier_t *barrier,
                            pthread_barrierattr_t *attr, unsigned int count);
void xpthread_barrier_destroy (pthread_barrier_t *barrier);
void xpthread_barrierattr_destroy (pthread_barrierattr_t *);
void xpthread_barrierattr_init (pthread_barrierattr_t *);
void xpthread_barrierattr_setpshared (pthread_barrierattr_t *, int pshared);
void xpthread_mutexattr_destroy (pthread_mutexattr_t *);
void xpthread_mutexattr_init (pthread_mutexattr_t *);
void xpthread_mutexattr_setprotocol (pthread_mutexattr_t *, int);
void xpthread_mutexattr_setpshared (pthread_mutexattr_t *, int);
void xpthread_mutexattr_setrobust (pthread_mutexattr_t *, int);
void xpthread_mutexattr_settype (pthread_mutexattr_t *, int);
void xpthread_mutex_init (pthread_mutex_t *, const pthread_mutexattr_t *);
void xpthread_mutex_destroy (pthread_mutex_t *);
void xpthread_mutex_lock (pthread_mutex_t *mutex);
void xpthread_mutex_unlock (pthread_mutex_t *mutex);
void xpthread_mutex_consistent (pthread_mutex_t *);
void xpthread_spin_lock (pthread_spinlock_t *lock);
void xpthread_spin_unlock (pthread_spinlock_t *lock);
void xpthread_cond_wait (pthread_cond_t * cond, pthread_mutex_t * mutex);
pthread_t xpthread_create (pthread_attr_t *attr,
                           void *(*thread_func) (void *), void *closure);
void xpthread_detach (pthread_t thr);
void xpthread_cancel (pthread_t thr);
void *xpthread_join (pthread_t thr);
void xpthread_once (pthread_once_t *guard, void (*func) (void));
void xpthread_attr_destroy (pthread_attr_t *attr);
void xpthread_attr_init (pthread_attr_t *attr);
#ifdef __linux__
void xpthread_attr_setaffinity_np (pthread_attr_t *attr,
				   size_t cpusetsize,
				   const cpu_set_t *cpuset);
#endif
void xpthread_attr_setdetachstate (pthread_attr_t *attr,
				   int detachstate);
void xpthread_attr_setstack (pthread_attr_t *attr, void *stackaddr,
			     size_t stacksize);
void xpthread_attr_setstacksize (pthread_attr_t *attr,
				 size_t stacksize);
void xpthread_attr_setguardsize (pthread_attr_t *attr,
				 size_t guardsize);

void xpthread_kill (pthread_t thr, int signo);

/* Return the stack size used on support_set_small_thread_stack_size.  */
size_t support_small_thread_stack_size (void);
/* Set the stack size in ATTR to a small value, but still large enough
   to cover most internal glibc stack usage.  */
void support_set_small_thread_stack_size (pthread_attr_t *attr);

/* Return a pointer to a thread attribute which requests a small
   stack.  The caller must not free this pointer.  */
pthread_attr_t *support_small_stack_thread_attribute (void);

/* This function returns non-zero if pthread_barrier_wait returned
   PTHREAD_BARRIER_SERIAL_THREAD.  */
int xpthread_barrier_wait (pthread_barrier_t *barrier);

void xpthread_rwlock_init (pthread_rwlock_t *rwlock,
			  const pthread_rwlockattr_t *attr);
void xpthread_rwlockattr_init (pthread_rwlockattr_t *attr);
void xpthread_rwlockattr_setkind_np (pthread_rwlockattr_t *attr, int pref);
void xpthread_rwlock_wrlock (pthread_rwlock_t *rwlock);
void xpthread_rwlock_rdlock (pthread_rwlock_t *rwlock);
void xpthread_rwlock_unlock (pthread_rwlock_t *rwlock);
void xpthread_rwlock_destroy (pthread_rwlock_t *rwlock);
pthread_key_t xpthread_key_create (void (*destr_function) (void *));
void xpthread_key_delete (pthread_key_t key);

__END_DECLS

#endif /* SUPPORT_THREAD_H */
