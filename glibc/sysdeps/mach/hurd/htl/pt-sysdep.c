/* System dependent pthreads code.  Hurd version.
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

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <mach.h>
#include <mach/mig_support.h>

#include <pt-internal.h>
#include <pthreadP.h>

__thread struct __pthread *___pthread_self;

static void
reset_pthread_total (void)
{
  /* Only current thread remains */
  __pthread_total = 1;
}

/* This function is called from the Hurd-specific startup code.  It
   should return a new stack pointer for the main thread.  The caller
   will switch to this new stack before doing anything serious.  */
static void
_init_routine (void *stack)
{
  struct __pthread *thread;
  int err;
  pthread_attr_t attr, *attrp = 0;

  if (__pthread_threads != NULL)
    /* Already initialized */
    return;

  /* Initialize the library.  */
  ___pthread_init ();

  if (stack != NULL)
    {
      /* We are getting initialized due to dlopening a library using libpthread
         while the main program was not linked against libpthread.  */
      /* Avoid allocating another stack */
      attrp = &attr;
      __pthread_attr_init (attrp);
      __pthread_attr_setstack (attrp, stack, __vm_page_size);
    }

  /* Create the pthread structure for the main thread (i.e. us).  */
  err = __pthread_create_internal (&thread, attrp, 0, 0);
  assert_perror (err);

  /* XXX The caller copies the command line arguments and the environment
     to the new stack.  Pretend it wasn't allocated so that it remains
     valid if the main thread terminates.  */
  thread->stack = 0;
  thread->tcb = THREAD_SELF;

#ifndef PAGESIZE
  __pthread_default_attr.__guardsize = __vm_page_size;
#endif

  ___pthread_self = thread;

  /* Decrease the number of threads, to take into account that the
     signal thread (which will be created by the glibc startup code
     when we return from here) shouldn't be seen as a user thread.  */
  __pthread_total--;

  __pthread_atfork (NULL, NULL, reset_pthread_total);

  GL(dl_init_static_tls) = &__pthread_init_static_tls;

  /* Make MiG code thread aware.  */
  __mig_init (thread->stackaddr);
}

void
__pthread_initialize_minimal (void)
{
  _init_routine (__libc_stack_end);
}

#ifdef SHARED
__attribute__ ((constructor))
static void
dynamic_init_routine (void)
{
  _init_routine (__libc_stack_end);
}
#endif
