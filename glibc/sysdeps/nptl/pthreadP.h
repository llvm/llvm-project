/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#define __PTHREAD_NPTL

#include <pthread.h>
#include <setjmp.h>
#include <stdbool.h>
#include <sys/syscall.h>
#include <nptl/descr.h>
#include <tls.h>
#include <lowlevellock.h>
#include <stackinfo.h>
#include <internaltypes.h>
#include <atomic.h>
#include <kernel-features.h>
#include <errno.h>
#include <internal-signals.h>
#include "pthread_mutex_conf.h"


/* Atomic operations on TLS memory.  */
#ifndef THREAD_ATOMIC_CMPXCHG_VAL
# define THREAD_ATOMIC_CMPXCHG_VAL(descr, member, new, old) \
  atomic_compare_and_exchange_val_acq (&(descr)->member, new, old)
#endif

#ifndef THREAD_ATOMIC_BIT_SET
# define THREAD_ATOMIC_BIT_SET(descr, member, bit) \
  atomic_bit_set (&(descr)->member, bit)
#endif


static inline short max_adaptive_count (void)
{
#if HAVE_TUNABLES
  return __mutex_aconf.spin_count;
#else
  return DEFAULT_ADAPTIVE_COUNT;
#endif
}


/* Magic cookie representing robust mutex with dead owner.  */
#define PTHREAD_MUTEX_INCONSISTENT	INT_MAX
/* Magic cookie representing not recoverable robust mutex.  */
#define PTHREAD_MUTEX_NOTRECOVERABLE	(INT_MAX - 1)


/* Internal mutex type value.  */
enum
{
  PTHREAD_MUTEX_KIND_MASK_NP = 3,

  PTHREAD_MUTEX_ELISION_NP    = 256,
  PTHREAD_MUTEX_NO_ELISION_NP = 512,

  PTHREAD_MUTEX_ROBUST_NORMAL_NP = 16,
  PTHREAD_MUTEX_ROBUST_RECURSIVE_NP
  = PTHREAD_MUTEX_ROBUST_NORMAL_NP | PTHREAD_MUTEX_RECURSIVE_NP,
  PTHREAD_MUTEX_ROBUST_ERRORCHECK_NP
  = PTHREAD_MUTEX_ROBUST_NORMAL_NP | PTHREAD_MUTEX_ERRORCHECK_NP,
  PTHREAD_MUTEX_ROBUST_ADAPTIVE_NP
  = PTHREAD_MUTEX_ROBUST_NORMAL_NP | PTHREAD_MUTEX_ADAPTIVE_NP,
  PTHREAD_MUTEX_PRIO_INHERIT_NP = 32,
  PTHREAD_MUTEX_PI_NORMAL_NP
  = PTHREAD_MUTEX_PRIO_INHERIT_NP | PTHREAD_MUTEX_NORMAL,
  PTHREAD_MUTEX_PI_RECURSIVE_NP
  = PTHREAD_MUTEX_PRIO_INHERIT_NP | PTHREAD_MUTEX_RECURSIVE_NP,
  PTHREAD_MUTEX_PI_ERRORCHECK_NP
  = PTHREAD_MUTEX_PRIO_INHERIT_NP | PTHREAD_MUTEX_ERRORCHECK_NP,
  PTHREAD_MUTEX_PI_ADAPTIVE_NP
  = PTHREAD_MUTEX_PRIO_INHERIT_NP | PTHREAD_MUTEX_ADAPTIVE_NP,
  PTHREAD_MUTEX_PI_ROBUST_NORMAL_NP
  = PTHREAD_MUTEX_PRIO_INHERIT_NP | PTHREAD_MUTEX_ROBUST_NORMAL_NP,
  PTHREAD_MUTEX_PI_ROBUST_RECURSIVE_NP
  = PTHREAD_MUTEX_PRIO_INHERIT_NP | PTHREAD_MUTEX_ROBUST_RECURSIVE_NP,
  PTHREAD_MUTEX_PI_ROBUST_ERRORCHECK_NP
  = PTHREAD_MUTEX_PRIO_INHERIT_NP | PTHREAD_MUTEX_ROBUST_ERRORCHECK_NP,
  PTHREAD_MUTEX_PI_ROBUST_ADAPTIVE_NP
  = PTHREAD_MUTEX_PRIO_INHERIT_NP | PTHREAD_MUTEX_ROBUST_ADAPTIVE_NP,
  PTHREAD_MUTEX_PRIO_PROTECT_NP = 64,
  PTHREAD_MUTEX_PP_NORMAL_NP
  = PTHREAD_MUTEX_PRIO_PROTECT_NP | PTHREAD_MUTEX_NORMAL,
  PTHREAD_MUTEX_PP_RECURSIVE_NP
  = PTHREAD_MUTEX_PRIO_PROTECT_NP | PTHREAD_MUTEX_RECURSIVE_NP,
  PTHREAD_MUTEX_PP_ERRORCHECK_NP
  = PTHREAD_MUTEX_PRIO_PROTECT_NP | PTHREAD_MUTEX_ERRORCHECK_NP,
  PTHREAD_MUTEX_PP_ADAPTIVE_NP
  = PTHREAD_MUTEX_PRIO_PROTECT_NP | PTHREAD_MUTEX_ADAPTIVE_NP,
  PTHREAD_MUTEX_ELISION_FLAGS_NP
  = PTHREAD_MUTEX_ELISION_NP | PTHREAD_MUTEX_NO_ELISION_NP,

  PTHREAD_MUTEX_TIMED_ELISION_NP =
	  PTHREAD_MUTEX_TIMED_NP | PTHREAD_MUTEX_ELISION_NP,
  PTHREAD_MUTEX_TIMED_NO_ELISION_NP =
	  PTHREAD_MUTEX_TIMED_NP | PTHREAD_MUTEX_NO_ELISION_NP,
};
#define PTHREAD_MUTEX_PSHARED_BIT 128

/* See concurrency notes regarding __kind in struct __pthread_mutex_s
   in sysdeps/nptl/bits/thread-shared-types.h.  */
#define PTHREAD_MUTEX_TYPE(m) \
  (atomic_load_relaxed (&((m)->__data.__kind)) & 127)
/* Don't include NO_ELISION, as that type is always the same
   as the underlying lock type.  */
#define PTHREAD_MUTEX_TYPE_ELISION(m) \
  (atomic_load_relaxed (&((m)->__data.__kind))	\
   & (127 | PTHREAD_MUTEX_ELISION_NP))

#if LLL_PRIVATE == 0 && LLL_SHARED == 128
# define PTHREAD_MUTEX_PSHARED(m) \
  (atomic_load_relaxed (&((m)->__data.__kind)) & 128)
#else
# define PTHREAD_MUTEX_PSHARED(m) \
  ((atomic_load_relaxed (&((m)->__data.__kind)) & 128)	\
   ? LLL_SHARED : LLL_PRIVATE)
#endif

/* The kernel when waking robust mutexes on exit never uses
   FUTEX_PRIVATE_FLAG FUTEX_WAKE.  */
#define PTHREAD_ROBUST_MUTEX_PSHARED(m) LLL_SHARED

/* Ceiling in __data.__lock.  __data.__lock is signed, so don't
   use the MSB bit in there, but in the mask also include that bit,
   so that the compiler can optimize & PTHREAD_MUTEX_PRIO_CEILING_MASK
   masking if the value is then shifted down by
   PTHREAD_MUTEX_PRIO_CEILING_SHIFT.  */
#define PTHREAD_MUTEX_PRIO_CEILING_SHIFT	19
#define PTHREAD_MUTEX_PRIO_CEILING_MASK		0xfff80000


/* Flags in mutex attr.  */
#define PTHREAD_MUTEXATTR_PROTOCOL_SHIFT	28
#define PTHREAD_MUTEXATTR_PROTOCOL_MASK		0x30000000
#define PTHREAD_MUTEXATTR_PRIO_CEILING_SHIFT	12
#define PTHREAD_MUTEXATTR_PRIO_CEILING_MASK	0x00fff000
#define PTHREAD_MUTEXATTR_FLAG_ROBUST		0x40000000
#define PTHREAD_MUTEXATTR_FLAG_PSHARED		0x80000000
#define PTHREAD_MUTEXATTR_FLAG_BITS \
  (PTHREAD_MUTEXATTR_FLAG_ROBUST | PTHREAD_MUTEXATTR_FLAG_PSHARED \
   | PTHREAD_MUTEXATTR_PROTOCOL_MASK | PTHREAD_MUTEXATTR_PRIO_CEILING_MASK)


/* For the following, see pthread_rwlock_common.c.  */
#define PTHREAD_RWLOCK_WRPHASE		1
#define PTHREAD_RWLOCK_WRLOCKED		2
#define PTHREAD_RWLOCK_RWAITING		4
#define PTHREAD_RWLOCK_READER_SHIFT	3
#define PTHREAD_RWLOCK_READER_OVERFLOW	((unsigned int) 1 \
					 << (sizeof (unsigned int) * 8 - 1))
#define PTHREAD_RWLOCK_WRHANDOVER	((unsigned int) 1 \
					 << (sizeof (unsigned int) * 8 - 1))
#define PTHREAD_RWLOCK_FUTEX_USED	2


/* Bits used in robust mutex implementation.  */
#define FUTEX_WAITERS		0x80000000
#define FUTEX_OWNER_DIED	0x40000000
#define FUTEX_TID_MASK		0x3fffffff


/* pthread_once definitions.  See __pthread_once for how these are used.  */
#define __PTHREAD_ONCE_INPROGRESS	1
#define __PTHREAD_ONCE_DONE		2
#define __PTHREAD_ONCE_FORK_GEN_INCR	4

/* Attribute to indicate thread creation was issued from C11 thrd_create.  */
#define ATTR_C11_THREAD ((void*)(uintptr_t)-1)


/* Condition variable definitions.  See __pthread_cond_wait_common.
   Need to be defined here so there is one place from which
   nptl_lock_constants can grab them.  */
#define __PTHREAD_COND_CLOCK_MONOTONIC_MASK 2
#define __PTHREAD_COND_SHARED_MASK 1


/* Internal variables.  */


/* Default pthread attributes.  */
extern union pthread_attr_transparent __default_pthread_attr;
libc_hidden_proto (__default_pthread_attr)
extern int __default_pthread_attr_lock;
libc_hidden_proto (__default_pthread_attr_lock)
/* Called from __libc_freeres to deallocate the default attribute.  */
extern void __default_pthread_attr_freeres (void) attribute_hidden;

/* Attribute handling.  */
extern struct pthread_attr *__attr_list attribute_hidden;
extern int __attr_list_lock attribute_hidden;

/* Concurrency handling.  */
extern int __concurrency_level attribute_hidden;

/* Thread-local data key handling.  */
extern struct pthread_key_struct __pthread_keys[PTHREAD_KEYS_MAX];
libc_hidden_proto (__pthread_keys)

/* Number of threads running.  */
extern unsigned int __nptl_nthreads;
libc_hidden_proto (__nptl_nthreads)

#ifndef __ASSUME_SET_ROBUST_LIST
/* True if the set_robust_list system call works.  Initialized in
   __tls_init_tp.  */
extern bool __nptl_set_robust_list_avail;
rtld_hidden_proto (__nptl_set_robust_list_avail)
#endif

/* Thread Priority Protection.  */
extern int __sched_fifo_min_prio;
libc_hidden_proto (__sched_fifo_min_prio)
extern int __sched_fifo_max_prio;
libc_hidden_proto (__sched_fifo_max_prio)
extern void __init_sched_fifo_prio (void);
libc_hidden_proto (__init_sched_fifo_prio)
extern int __pthread_tpp_change_priority (int prev_prio, int new_prio);
libc_hidden_proto (__pthread_tpp_change_priority)
extern int __pthread_current_priority (void);
libc_hidden_proto (__pthread_current_priority)

/* This will not catch all invalid descriptors but is better than
   nothing.  And if the test triggers the thread descriptor is
   guaranteed to be invalid.  */
#define INVALID_TD_P(pd) __builtin_expect ((pd)->tid <= 0, 0)
#define INVALID_NOT_TERMINATED_TD_P(pd) __builtin_expect ((pd)->tid < 0, 0)

extern void __pthread_unwind (__pthread_unwind_buf_t *__buf)
     __cleanup_fct_attribute __attribute ((__noreturn__))
#if !defined SHARED && !IS_IN (libpthread)
     weak_function
#endif
     ;
libc_hidden_proto (__pthread_unwind)
extern void __pthread_unwind_next (__pthread_unwind_buf_t *__buf)
     __cleanup_fct_attribute __attribute ((__noreturn__))
#ifndef SHARED
     weak_function
#endif
     ;
/* NB: No hidden proto for __pthread_unwind_next: inside glibc, the
   legacy unwinding mechanism is used.  */

extern void __pthread_register_cancel (__pthread_unwind_buf_t *__buf)
     __cleanup_fct_attribute;
libc_hidden_proto (__pthread_register_cancel)
extern void __pthread_unregister_cancel (__pthread_unwind_buf_t *__buf)
     __cleanup_fct_attribute;
libc_hidden_proto (__pthread_unregister_cancel)

/* Called when a thread reacts on a cancellation request.  */
static inline void
__attribute ((noreturn, always_inline))
__do_cancel (void)
{
  struct pthread *self = THREAD_SELF;

  /* Make sure we get no more cancellations.  */
  THREAD_ATOMIC_BIT_SET (self, cancelhandling, EXITING_BIT);

  __pthread_unwind ((__pthread_unwind_buf_t *)
		    THREAD_GETMEM (self, cleanup_jmp_buf));
}


/* Internal prototypes.  */

/* Deallocate a thread's stack after optionally making sure the thread
   descriptor is still valid.  */
extern void __nptl_free_tcb (struct pthread *pd);
libc_hidden_proto (__nptl_free_tcb)

/* Change the permissions of a thread stack.  Called from
   _dl_make_stacks_executable and pthread_create.  */
int
__nptl_change_stack_perm (struct pthread *pd);
rtld_hidden_proto (__nptl_change_stack_perm)

/* longjmp handling.  */
extern void __pthread_cleanup_upto (__jmp_buf target, char *targetframe);
libc_hidden_proto (__pthread_cleanup_upto)


/* Functions with versioned interfaces.  */
extern int __pthread_create (pthread_t *newthread,
			     const pthread_attr_t *attr,
			     void *(*start_routine) (void *), void *arg);
libc_hidden_proto (__pthread_create)
extern int __pthread_create_2_0 (pthread_t *newthread,
				 const pthread_attr_t *attr,
				 void *(*start_routine) (void *), void *arg);
extern int __pthread_attr_init (pthread_attr_t *attr);
libc_hidden_proto (__pthread_attr_init)
extern int __pthread_attr_init_2_0 (pthread_attr_t *attr);

/* Part of the legacy thread events interface (which has been
   superseded by PTRACE_O_TRACECLONE).  This can be set by the
   debugger before initialization is complete.  */
extern bool __nptl_initial_report_events;
rtld_hidden_proto (__nptl_initial_report_events)

/* Event handlers for libthread_db interface.  */
extern void __nptl_create_event (void);
extern void __nptl_death_event (void);
libc_hidden_proto (__nptl_create_event)
libc_hidden_proto (__nptl_death_event)

/* The fork generation counter, defined in libpthread.  */
extern unsigned long int __fork_generation attribute_hidden;

/* Pointer to the fork generation counter in the thread library.  */
extern unsigned long int *__fork_generation_pointer attribute_hidden;

extern size_t __pthread_get_minstack (const pthread_attr_t *attr);
libc_hidden_proto (__pthread_get_minstack)

/* Namespace save aliases.  */
extern int __pthread_getschedparam (pthread_t thread_id, int *policy,
				    struct sched_param *param);
libc_hidden_proto (__pthread_getschedparam)
extern int __pthread_setschedparam (pthread_t thread_id, int policy,
				    const struct sched_param *param);
extern int __pthread_mutex_init (pthread_mutex_t *__mutex,
				 const pthread_mutexattr_t *__mutexattr);
libc_hidden_proto (__pthread_mutex_init)
extern int __pthread_mutex_destroy (pthread_mutex_t *__mutex);
libc_hidden_proto (__pthread_mutex_destroy)
extern int __pthread_mutex_trylock (pthread_mutex_t *_mutex);
libc_hidden_proto (__pthread_mutex_trylock)
extern int __pthread_mutex_lock (pthread_mutex_t *__mutex);
libc_hidden_proto (__pthread_mutex_lock)
extern int __pthread_mutex_timedlock (pthread_mutex_t *__mutex,
     const struct timespec *__abstime);
extern int __pthread_mutex_cond_lock (pthread_mutex_t *__mutex)
     attribute_hidden;
extern void __pthread_mutex_cond_lock_adjust (pthread_mutex_t *__mutex)
     attribute_hidden;
extern int __pthread_mutex_unlock (pthread_mutex_t *__mutex);
libc_hidden_proto (__pthread_mutex_unlock)
extern int __pthread_mutex_unlock_usercnt (pthread_mutex_t *__mutex,
					   int __decr);
libc_hidden_proto (__pthread_mutex_unlock_usercnt)
extern int __pthread_mutexattr_init (pthread_mutexattr_t *attr);
libc_hidden_proto (__pthread_mutexattr_init)
extern int __pthread_mutexattr_destroy (pthread_mutexattr_t *attr);
extern int __pthread_mutexattr_settype (pthread_mutexattr_t *attr, int kind);
libc_hidden_proto (__pthread_mutexattr_settype)
extern int __pthread_attr_destroy (pthread_attr_t *attr);
libc_hidden_proto (__pthread_attr_destroy)
extern int __pthread_attr_getdetachstate (const pthread_attr_t *attr,
					  int *detachstate);
extern int __pthread_attr_setdetachstate (pthread_attr_t *attr,
					  int detachstate);
extern int __pthread_attr_getinheritsched (const pthread_attr_t *attr,
					   int *inherit);
extern int __pthread_attr_setinheritsched (pthread_attr_t *attr, int inherit);
extern int __pthread_attr_getschedparam (const pthread_attr_t *attr,
					 struct sched_param *param);
extern int __pthread_attr_setschedparam (pthread_attr_t *attr,
					 const struct sched_param *param);
extern int __pthread_attr_getschedpolicy (const pthread_attr_t *attr,
					  int *policy);
extern int __pthread_attr_setschedpolicy (pthread_attr_t *attr, int policy);
extern int __pthread_attr_getscope (const pthread_attr_t *attr, int *scope);
extern int __pthread_attr_setscope (pthread_attr_t *attr, int scope);
extern int __pthread_attr_getstackaddr (const pthread_attr_t *__restrict
					__attr, void **__restrict __stackaddr);
extern int __pthread_attr_setstackaddr (pthread_attr_t *__attr,
					void *__stackaddr);
extern int __pthread_attr_getstacksize (const pthread_attr_t *__restrict
					__attr,
					size_t *__restrict __stacksize);
extern int __pthread_attr_setstacksize (pthread_attr_t *__attr,
					size_t __stacksize);
extern int __pthread_attr_getstack (const pthread_attr_t *__restrict __attr,
				    void **__restrict __stackaddr,
				    size_t *__restrict __stacksize);
extern int __pthread_attr_setstack (pthread_attr_t *__attr, void *__stackaddr,
				    size_t __stacksize);
int __pthread_attr_setaffinity_np (pthread_attr_t *, size_t, const cpu_set_t *);
libc_hidden_proto (__pthread_attr_setaffinity_np)
extern __typeof (pthread_getattr_default_np) __pthread_getattr_default_np;
libc_hidden_proto (__pthread_getattr_default_np)
extern int __pthread_rwlock_init (pthread_rwlock_t *__restrict __rwlock,
				  const pthread_rwlockattr_t *__restrict
				  __attr);
extern int __pthread_rwlock_destroy (pthread_rwlock_t *__rwlock);
extern int __pthread_rwlock_rdlock (pthread_rwlock_t *__rwlock);
libc_hidden_proto (__pthread_rwlock_rdlock)
extern int __pthread_rwlock_tryrdlock (pthread_rwlock_t *__rwlock);
extern int __pthread_rwlock_wrlock (pthread_rwlock_t *__rwlock);
libc_hidden_proto (__pthread_rwlock_wrlock)
extern int __pthread_rwlock_trywrlock (pthread_rwlock_t *__rwlock);
extern int __pthread_rwlock_unlock (pthread_rwlock_t *__rwlock);
extern int __pthread_cond_broadcast (pthread_cond_t *cond);
libc_hidden_proto (__pthread_cond_broadcast)
extern int __pthread_cond_destroy (pthread_cond_t *cond);
libc_hidden_proto (__pthread_cond_destroy)
extern int __pthread_cond_init (pthread_cond_t *cond,
				const pthread_condattr_t *cond_attr);
libc_hidden_proto (__pthread_cond_init)
extern int __pthread_cond_signal (pthread_cond_t *cond);
libc_hidden_proto (__pthread_cond_signal)
extern int __pthread_cond_wait (pthread_cond_t *cond, pthread_mutex_t *mutex);
libc_hidden_proto (__pthread_cond_wait)

#if __TIMESIZE == 64
# define __pthread_clockjoin_np64 __pthread_clockjoin_np
# define __pthread_timedjoin_np64 __pthread_timedjoin_np
# define __pthread_cond_timedwait64 __pthread_cond_timedwait
# define __pthread_cond_clockwait64 __pthread_cond_clockwait
# define __pthread_rwlock_clockrdlock64 __pthread_rwlock_clockrdlock
# define __pthread_rwlock_clockwrlock64 __pthread_rwlock_clockwrlock
# define __pthread_rwlock_timedrdlock64 __pthread_rwlock_timedrdlock
# define __pthread_rwlock_timedwrlock64 __pthread_rwlock_timedwrlock
# define __pthread_mutex_clocklock64 __pthread_mutex_clocklock
# define __pthread_mutex_timedlock64 __pthread_mutex_timedlock
#else
extern int __pthread_clockjoin_np64 (pthread_t threadid, void **thread_return,
                                     clockid_t clockid,
                                     const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_clockjoin_np64)
extern int __pthread_timedjoin_np64 (pthread_t threadid, void **thread_return,
                                     const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_timedjoin_np64)
extern int __pthread_cond_timedwait64 (pthread_cond_t *cond,
                                       pthread_mutex_t *mutex,
                                       const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_cond_timedwait64)
extern int __pthread_cond_clockwait64 (pthread_cond_t *cond,
                                       pthread_mutex_t *mutex,
                                       clockid_t clockid,
                                       const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_cond_clockwait64)
extern int __pthread_rwlock_clockrdlock64 (pthread_rwlock_t *rwlock,
                                           clockid_t clockid,
                                           const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_rwlock_clockrdlock64)
extern int __pthread_rwlock_clockwrlock64 (pthread_rwlock_t *rwlock,
                                           clockid_t clockid,
                                           const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_rwlock_clockwrlock64)
extern int __pthread_rwlock_timedrdlock64 (pthread_rwlock_t *rwlock,
                                           const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_rwlock_timedrdlock64)
extern int __pthread_rwlock_timedwrlock64 (pthread_rwlock_t *rwlock,
                                           const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_rwlock_timedwrlock64)
extern int __pthread_mutex_clocklock64 (pthread_mutex_t *mutex,
                                        clockid_t clockid,
                                        const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_mutex_clocklock64)
extern int __pthread_mutex_timedlock64 (pthread_mutex_t *mutex,
                                        const struct __timespec64 *abstime);
libc_hidden_proto (__pthread_mutex_timedlock64)
#endif

extern int __pthread_cond_timedwait (pthread_cond_t *cond,
				     pthread_mutex_t *mutex,
				     const struct timespec *abstime);
libc_hidden_proto (__pthread_cond_timedwait)
extern int __pthread_cond_clockwait (pthread_cond_t *cond,
				     pthread_mutex_t *mutex,
				     clockid_t clockid,
				     const struct timespec *abstime)
  __nonnull ((1, 2, 4));
libc_hidden_proto (__pthread_cond_clockwait)

extern int __pthread_mutex_clocklock (pthread_mutex_t *mutex,
				      clockid_t clockid,
				      const struct timespec *abstime);
libc_hidden_proto (__pthread_mutex_clocklock)
extern int __pthread_mutex_timedlock (pthread_mutex_t *mutex,
				      const struct timespec *abstime);
libc_hidden_proto (__pthread_mutex_timedlock)

extern int __pthread_condattr_destroy (pthread_condattr_t *attr);
extern int __pthread_condattr_init (pthread_condattr_t *attr);
extern int __pthread_key_create (pthread_key_t *key, void (*destr) (void *));
libc_hidden_proto (__pthread_key_create)
extern int __pthread_key_delete (pthread_key_t key);
libc_hidden_proto (__pthread_key_delete)
extern void *__pthread_getspecific (pthread_key_t key);
libc_hidden_proto (__pthread_getspecific)
extern int __pthread_setspecific (pthread_key_t key, const void *value);
libc_hidden_proto (__pthread_setspecific)
extern int __pthread_once (pthread_once_t *once_control,
			   void (*init_routine) (void));
libc_hidden_proto (__pthread_once)
extern int __pthread_atfork (void (*prepare) (void), void (*parent) (void),
			     void (*child) (void));
libc_hidden_proto (__pthread_self)
extern int __pthread_equal (pthread_t thread1, pthread_t thread2);
extern int __pthread_detach (pthread_t th);
libc_hidden_proto (__pthread_detach)
extern int __pthread_kill (pthread_t threadid, int signo);
libc_hidden_proto (__pthread_kill)
extern int __pthread_cancel (pthread_t th);
extern int __pthread_kill_internal (pthread_t threadid, int signo)
  attribute_hidden;
extern void __pthread_exit (void *value) __attribute__ ((__noreturn__));
libc_hidden_proto (__pthread_exit)
extern int __pthread_join (pthread_t threadid, void **thread_return);
libc_hidden_proto (__pthread_join)
extern int __pthread_setcanceltype (int type, int *oldtype);
libc_hidden_proto (__pthread_setcanceltype)
extern void __pthread_testcancel (void);
libc_hidden_proto (__pthread_testcancel)
extern int __pthread_clockjoin_ex (pthread_t, void **, clockid_t,
				   const struct __timespec64 *, bool)
  attribute_hidden;
extern int __pthread_sigmask (int, const sigset_t *, sigset_t *);
libc_hidden_proto (__pthread_sigmask);


#if IS_IN (libpthread)
hidden_proto (__pthread_rwlock_unlock)
#endif

extern int __pthread_cond_broadcast_2_0 (pthread_cond_2_0_t *cond);
extern int __pthread_cond_destroy_2_0 (pthread_cond_2_0_t *cond);
extern int __pthread_cond_init_2_0 (pthread_cond_2_0_t *cond,
				    const pthread_condattr_t *cond_attr);
extern int __pthread_cond_signal_2_0 (pthread_cond_2_0_t *cond);
extern int __pthread_cond_timedwait_2_0 (pthread_cond_2_0_t *cond,
					 pthread_mutex_t *mutex,
					 const struct timespec *abstime);
extern int __pthread_cond_wait_2_0 (pthread_cond_2_0_t *cond,
				    pthread_mutex_t *mutex);

extern int __pthread_getaffinity_np (pthread_t th, size_t cpusetsize,
				     cpu_set_t *cpuset);
libc_hidden_proto (__pthread_getaffinity_np)

/* Special internal version of pthread_attr_setsigmask_np which does
   not filter out internal signals from *SIGMASK.  This can be used to
   launch threads with internal signals blocked.  */
  extern int __pthread_attr_setsigmask_internal (pthread_attr_t *attr,
						 const sigset_t *sigmask);
libc_hidden_proto (__pthread_attr_setsigmask_internal)

extern __typeof (pthread_attr_getsigmask_np) __pthread_attr_getsigmask_np;
libc_hidden_proto (__pthread_attr_getsigmask_np)

/* Special versions which use non-exported functions.  */
extern void __pthread_cleanup_push (struct _pthread_cleanup_buffer *buffer,
				    void (*routine) (void *), void *arg);
libc_hidden_proto (__pthread_cleanup_push)

/* Replace cleanup macros defined in <pthread.h> with internal
   versions that don't depend on unwind info and better support
   cancellation.  */
# undef pthread_cleanup_push
# define pthread_cleanup_push(routine,arg)              \
  { struct _pthread_cleanup_buffer _buffer;             \
  __pthread_cleanup_push (&_buffer, (routine), (arg));

extern void __pthread_cleanup_pop (struct _pthread_cleanup_buffer *buffer,
				   int execute);
libc_hidden_proto (__pthread_cleanup_pop)
# undef pthread_cleanup_pop
# define pthread_cleanup_pop(execute)                   \
  __pthread_cleanup_pop (&_buffer, (execute)); }

#if defined __EXCEPTIONS && !defined __cplusplus
/* Structure to hold the cleanup handler information.  */
struct __pthread_cleanup_combined_frame
{
  void (*__cancel_routine) (void *);
  void *__cancel_arg;
  int __do_it;
  struct _pthread_cleanup_buffer __buffer;
};

/* Special cleanup macros which register cleanup both using
   __pthread_cleanup_{push,pop} and using cleanup attribute.  This is needed
   for pthread_once, so that it supports both throwing exceptions from the
   pthread_once callback (only cleanup attribute works there) and cancellation
   of the thread running the callback if the callback or some routines it
   calls don't have unwind information.  */

static __always_inline void
__pthread_cleanup_combined_routine (struct __pthread_cleanup_combined_frame
				    *__frame)
{
  if (__frame->__do_it)
    {
      __frame->__cancel_routine (__frame->__cancel_arg);
      __frame->__do_it = 0;
      __pthread_cleanup_pop (&__frame->__buffer, 0);
    }
}

static inline void
__pthread_cleanup_combined_routine_voidptr (void *__arg)
{
  struct __pthread_cleanup_combined_frame *__frame
    = (struct __pthread_cleanup_combined_frame *) __arg;
  if (__frame->__do_it)
    {
      __frame->__cancel_routine (__frame->__cancel_arg);
      __frame->__do_it = 0;
    }
}

# define pthread_cleanup_combined_push(routine, arg) \
  do {									      \
    void (*__cancel_routine) (void *) = (routine);			      \
    struct __pthread_cleanup_combined_frame __clframe			      \
      __attribute__ ((__cleanup__ (__pthread_cleanup_combined_routine)))      \
      = { .__cancel_routine = __cancel_routine, .__cancel_arg = (arg),	      \
	  .__do_it = 1 };						      \
    __pthread_cleanup_push (&__clframe.__buffer,			      \
			    __pthread_cleanup_combined_routine_voidptr,	      \
			    &__clframe);

# define pthread_cleanup_combined_pop(execute) \
    __pthread_cleanup_pop (&__clframe.__buffer, 0);			      \
    __clframe.__do_it = 0;						      \
    if (execute)							      \
      __cancel_routine (__clframe.__cancel_arg);			      \
  } while (0)

#endif /* __EXCEPTIONS && !defined __cplusplus */

extern void __pthread_cleanup_push_defer (struct _pthread_cleanup_buffer *buffer,
					  void (*routine) (void *), void *arg);
extern void __pthread_cleanup_pop_restore (struct _pthread_cleanup_buffer *buffer,
					   int execute);

/* Old cleanup interfaces, still used in libc.so.  */
extern void _pthread_cleanup_push (struct _pthread_cleanup_buffer *buffer,
				   void (*routine) (void *), void *arg);
extern void _pthread_cleanup_pop (struct _pthread_cleanup_buffer *buffer,
				  int execute);
extern void _pthread_cleanup_push_defer (struct _pthread_cleanup_buffer *buffer,
					 void (*routine) (void *), void *arg);
extern void _pthread_cleanup_pop_restore (struct _pthread_cleanup_buffer *buffer,
					  int execute);

extern void __nptl_deallocate_tsd (void);
libc_hidden_proto (__nptl_deallocate_tsd)

void __nptl_setxid_sighandler (int sig, siginfo_t *si, void *ctx);
libc_hidden_proto (__nptl_setxid_sighandler)
extern int __nptl_setxid (struct xid_command *cmdp) attribute_hidden;

extern void __wait_lookup_done (void) attribute_hidden;

/* Allocates the extension space for ATTR.  Returns an error code on
   memory allocation failure, zero on success.  If ATTR already has an
   extension space, this function does nothing.  */
int __pthread_attr_extension (struct pthread_attr *attr) attribute_hidden
  __attribute_warn_unused_result__;

#ifdef SHARED
# define PTHREAD_STATIC_FN_REQUIRE(name)
#else
# define PTHREAD_STATIC_FN_REQUIRE(name) __asm (".globl " #name);
#endif

/* Make a deep copy of the attribute *SOURCE in *TARGET.  *TARGET is
   not assumed to have been initialized.  Returns 0 on success, or a
   positive error code otherwise.  */
int __pthread_attr_copy (pthread_attr_t *target, const pthread_attr_t *source);
libc_hidden_proto (__pthread_attr_copy)

/* Returns 0 if POL is a valid scheduling policy.  */
static inline int
check_sched_policy_attr (int pol)
{
  if (pol == SCHED_OTHER || pol == SCHED_FIFO || pol == SCHED_RR)
    return 0;

  return EINVAL;
}

/* Returns 0 if PR is within the accepted range of priority values for
   the scheduling policy POL or EINVAL otherwise.  */
static inline int
check_sched_priority_attr (int pr, int pol)
{
  int min = __sched_get_priority_min (pol);
  int max = __sched_get_priority_max (pol);

  if (min >= 0 && max >= 0 && pr >= min && pr <= max)
    return 0;

  return EINVAL;
}

/* Returns 0 if ST is a valid stack size for a thread stack and EINVAL
   otherwise.  */
static inline int
check_stacksize_attr (size_t st)
{
  if (st >= PTHREAD_STACK_MIN)
    return 0;

  return EINVAL;
}

#define ASSERT_TYPE_SIZE(type, size) 					\
  _Static_assert (sizeof (type) == size,				\
		  "sizeof (" #type ") != " #size)

#define ASSERT_PTHREAD_INTERNAL_SIZE(type, internal) 			\
  _Static_assert (sizeof ((type) { { 0 } }).__size >= sizeof (internal),\
		  "sizeof (" #type ".__size) < sizeof (" #internal ")")

#define ASSERT_PTHREAD_STRING(x) __STRING (x)
#define ASSERT_PTHREAD_INTERNAL_OFFSET(type, member, offset)		\
  _Static_assert (offsetof (type, member) == offset,			\
		  "offset of " #member " field of " #type " != "	\
		  ASSERT_PTHREAD_STRING (offset))
#define ASSERT_PTHREAD_INTERNAL_MEMBER_SIZE(type, member, mtype)	\
  _Static_assert (sizeof (((type) { 0 }).member) != 8,	\
		  "sizeof (" #type "." #member ") != sizeof (" #mtype "))")

#endif	/* pthreadP.h */
