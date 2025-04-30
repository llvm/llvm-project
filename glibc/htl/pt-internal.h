/* Internal defenitions for pthreads library.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#ifndef _PT_INTERNAL_H
#define _PT_INTERNAL_H	1

#include <pthread.h>
#include <stddef.h>
#include <sched.h>
#include <signal.h>
#include <assert.h>
#include <bits/types/res_state.h>

#include <atomic.h>

#include <pt-key.h>

#include <pt-sysdep.h>
#include <pt-machdep.h>

#if IS_IN (libpthread)
# include <ldsodefs.h>
#endif

#include <tls.h>

/* Thread state.  */
enum pthread_state
{
  /* The thread is running and joinable.  */
  PTHREAD_JOINABLE = 0,
  /* The thread is running and detached.  */
  PTHREAD_DETACHED,
  /* A joinable thread exited and its return code is available.  */
  PTHREAD_EXITED,
  /* The thread structure is unallocated and available for reuse.  */
  PTHREAD_TERMINATED
};

#ifndef PTHREAD_KEY_MEMBERS
# define PTHREAD_KEY_MEMBERS
#endif

#ifndef PTHREAD_SYSDEP_MEMBERS
# define PTHREAD_SYSDEP_MEMBERS
#endif

/* This structure describes a POSIX thread.  */
struct __pthread
{
  /* Thread ID.  */
  pthread_t thread;

  unsigned int nr_refs;		/* Detached threads have a self reference only,
				   while joinable threads have two references.
				   These are used to keep the structure valid at
				   thread destruction.  Detaching/joining a thread
				   drops a reference.  */

  /* Cancellation.  */
  pthread_mutex_t cancel_lock;	/* Protect cancel_xxx members.  */
  void (*cancel_hook) (void *);	/* Called to unblock a thread blocking
				   in a cancellation point (namely,
				   __pthread_cond_timedwait_internal).  */
  void *cancel_hook_arg;
  int cancel_state;
  int cancel_type;
  int cancel_pending;

  /* Thread stack.  */
  void *stackaddr;
  size_t stacksize;
  size_t guardsize;
  int stack;			/* Nonzero if the stack was allocated.  */

  /* Exit status.  */
  void *status;

  /* Thread state.  */
  enum pthread_state state;
  pthread_mutex_t state_lock;	/* Locks the state.  */
  pthread_cond_t state_cond;	/* Signalled when the state changes.  */

  /* Resolver state.  */
  struct __res_state res_state;

  /* Indicates whether is a C11 thread created by thrd_creat.  */
  bool c11;

  /* Initial sigset for the thread.  */
  sigset_t init_sigset;

  /* Thread context.  */
  struct pthread_mcontext mcontext;

  PTHREAD_KEY_MEMBERS

  PTHREAD_SYSDEP_MEMBERS

  tcbhead_t *tcb;

  /* Queue links.  Since PREVP is used to determine if a thread has been
     awaken, it must be protected by the queue lock.  */
  struct __pthread *next, **prevp;
};

/* Enqueue an element THREAD on the queue *HEAD.  */
static inline void
__pthread_enqueue (struct __pthread **head, struct __pthread *thread)
{
  assert (thread->prevp == 0);

  thread->next = *head;
  thread->prevp = head;
  if (*head)
    (*head)->prevp = &thread->next;
  *head = thread;
}

/* Dequeue the element THREAD from the queue it is connected to.  */
static inline void
__pthread_dequeue (struct __pthread *thread)
{
  assert (thread);
  assert (thread->prevp);

  if (thread->next)
    thread->next->prevp = thread->prevp;
  *thread->prevp = thread->next;
  thread->prevp = 0;
}

/* Iterate over QUEUE storing each element in ELEMENT.  */
#define __pthread_queue_iterate(queue, element)				\
  for (struct __pthread *__pdi_next = (queue);				\
       ((element) = __pdi_next)						\
	 && ((__pdi_next = __pdi_next->next),				\
	     1);							\
       )

/* Iterate over QUEUE dequeuing each element, storing it in
   ELEMENT.  */
#define __pthread_dequeuing_iterate(queue, element)			\
  for (struct __pthread *__pdi_next = (queue);				\
       ((element) = __pdi_next)						\
	 && ((__pdi_next = __pdi_next->next),				\
	     ((element)->prevp = 0),					\
	     1);							\
       )

/* The total number of threads currently active.  */
extern unsigned int __pthread_total;

/* The total number of thread IDs currently in use, or on the list of
   available thread IDs.  */
extern int __pthread_num_threads;

/* Concurrency hint.  */
extern int __pthread_concurrency;

/* Array of __pthread structures and its lock.  Indexed by the pthread
   id minus one.  (Why not just use the pthread id?  Because some
   brain-dead users of the pthread interface incorrectly assume that 0
   is an invalid pthread id.)  */
extern struct __pthread **__pthread_threads;
extern int __pthread_max_threads;
extern pthread_rwlock_t __pthread_threads_lock;

#define __pthread_getid(thread) \
  ({ struct __pthread *__t = NULL;                                           \
     __pthread_rwlock_rdlock (&__pthread_threads_lock);                      \
     if (thread <= __pthread_max_threads)                                    \
       __t = __pthread_threads[thread - 1];                                  \
     __pthread_rwlock_unlock (&__pthread_threads_lock);                      \
     __t; })

#define __pthread_setid(thread, pthread) \
  __pthread_rwlock_wrlock (&__pthread_threads_lock);                         \
  __pthread_threads[thread - 1] = pthread;                                   \
  __pthread_rwlock_unlock (&__pthread_threads_lock);

/* Similar to pthread_self, but returns the thread descriptor instead
   of the thread ID.  */
#ifndef _pthread_self
extern struct __pthread *_pthread_self (void);
#endif

/* Stores the stack of cleanup handlers for the thread.  */
extern __thread struct __pthread_cancelation_handler *__pthread_cleanup_stack;


/* Initialize the pthreads library.  */
extern void ___pthread_init (void);

/* Internal version of pthread_create.  Rather than return the new
   tid, we return the whole __pthread structure in *PTHREAD.  */
extern int __pthread_create_internal (struct __pthread **__restrict pthread,
				      const pthread_attr_t *__restrict attr,
				      void *(*start_routine) (void *),
				      void *__restrict arg);

/* Allocate a new thread structure and a pthread thread ID (but not a
   kernel thread or a stack).  THREAD has one reference.  */
extern int __pthread_alloc (struct __pthread **thread);

/* Deallocate the thread structure.  This is the dual of
   __pthread_alloc (N.B. it does not call __pthread_stack_dealloc nor
   __pthread_thread_terminate).  THREAD loses one reference and is
   released if the reference counter drops to 0.  */
extern void __pthread_dealloc (struct __pthread *thread);


/* Allocate a stack of size STACKSIZE.  The stack base shall be
   returned in *STACKADDR.  */
extern int __pthread_stack_alloc (void **stackaddr, size_t stacksize);

/* Deallocate the stack STACKADDR of size STACKSIZE.  */
extern void __pthread_stack_dealloc (void *stackaddr, size_t stacksize);


/* Setup thread THREAD's context.  */
extern int __pthread_setup (struct __pthread *__restrict thread,
			    void (*entry_point) (struct __pthread *,
						 void *(*)(void *),
						 void *),
			    void *(*start_routine) (void *),
			    void *__restrict arg);


/* Allocate a kernel thread (and any miscellaneous system dependent
   resources) for THREAD; it must not be placed on the run queue.  */
extern int __pthread_thread_alloc (struct __pthread *thread);

/* Start THREAD making it eligible to run.  */
extern int __pthread_thread_start (struct __pthread *thread);

/* Terminate the kernel thread associated with THREAD, and deallocate its
   stack as well as any other kernel resource associated with it.
   In addition, THREAD looses one reference.

   This function can be called by any thread, including the target thread.
   Since some resources that are destroyed along the kernel thread are
   stored in thread-local variables, the conditions required for this
   function to behave correctly are a bit unusual : as long as the target
   thread hasn't been started, any thread can terminate it, but once it
   has started, no other thread can terminate it, so that thread-local
   variables created by that thread are correctly released.  */
extern void __pthread_thread_terminate (struct __pthread *thread);


/* Called by a thread just before it calls the provided start
   routine.  */
extern void __pthread_startup (void);

/* Block THREAD.  */
extern void __pthread_block (struct __pthread *thread);

/* Block THREAD until *ABSTIME is reached.  */
extern error_t __pthread_timedblock (struct __pthread *__restrict thread,
				     const struct timespec *__restrict abstime,
				     clockid_t clock_id);

/* Block THREAD with interrupts.  */
extern error_t __pthread_block_intr (struct __pthread *thread);

/* Block THREAD until *ABSTIME is reached, with interrupts.  */
extern error_t __pthread_timedblock_intr (struct __pthread *__restrict thread,
					  const struct timespec *__restrict abstime,
					  clockid_t clock_id);

/* Wakeup THREAD.  */
extern void __pthread_wakeup (struct __pthread *thread);


/* Perform a cancelation.  The CANCEL_LOCK member of the given thread must
   be locked before calling this function, which must unlock it.  */
extern int __pthread_do_cancel (struct __pthread *thread);


/* Initialize the thread specific data structures.  THREAD must be the
   calling thread.  */
extern error_t __pthread_init_specific (struct __pthread *thread);

/* Call the destructors on all of the thread specific data in THREAD.
   THREAD must be the calling thread.  */
extern void __pthread_destroy_specific (struct __pthread *thread);


/* Initialize newly create thread *THREAD's signal state data
   structures.  */
extern error_t __pthread_sigstate_init (struct __pthread *thread);

/* Destroy the signal state data structures associcated with thread
   *THREAD.  */
extern void __pthread_sigstate_destroy (struct __pthread *thread);

/* Modify thread *THREAD's signal state.  */
extern error_t __pthread_sigstate (struct __pthread *__restrict thread, int how,
				   const sigset_t *__restrict set,
				   sigset_t *__restrict oset,
				   int clear_pending);

/* If supported, check that MUTEX is locked by the caller.  */
extern int __pthread_mutex_checklocked (pthread_mutex_t *mtx);


/* Default thread attributes.  */
extern struct __pthread_attr __pthread_default_attr;

/* Default barrier attributes.  */
extern const struct __pthread_barrierattr __pthread_default_barrierattr;

/* Default rdlock attributes.  */
extern const struct __pthread_rwlockattr __pthread_default_rwlockattr;

/* Default condition attributes.  */
extern const struct __pthread_condattr __pthread_default_condattr;

/* Semaphore encoding.
   See nptl implementation for the details.  */
struct new_sem
{
#if __HAVE_64B_ATOMICS
  /* The data field holds both value (in the least-significant 32 bits) and
     nwaiters.  */
# if __BYTE_ORDER == __LITTLE_ENDIAN
#  define SEM_VALUE_OFFSET 0
# elif __BYTE_ORDER == __BIG_ENDIAN
#  define SEM_VALUE_OFFSET 1
# else
#  error Unsupported byte order.
# endif
# define SEM_NWAITERS_SHIFT 32
# define SEM_VALUE_MASK (~(unsigned int)0)
  uint64_t data;
  int pshared;
#define __SEMAPHORE_INITIALIZER(value, pshared) \
  { (value), (pshared) }
#else
# define SEM_VALUE_SHIFT 1
# define SEM_NWAITERS_MASK ((unsigned int)1)
  unsigned int value;
  unsigned int nwaiters;
  int pshared;
#define __SEMAPHORE_INITIALIZER(value, pshared) \
  { (value) << SEM_VALUE_SHIFT, 0, (pshared) }
#endif
};

extern int __sem_waitfast (struct new_sem *isem, int definitive_result);

#endif /* pt-internal.h */
