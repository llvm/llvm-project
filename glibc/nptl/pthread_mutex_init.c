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

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <string.h>
#include <kernel-features.h>
#include "pthreadP.h"
#include <atomic.h>
#include <pthread-offsets.h>
#include <futex-internal.h>
#include <shlib-compat.h>
#include <sys/mman.h>

#include <stap-probe.h>

static const struct pthread_mutexattr default_mutexattr =
  {
    /* Default is a normal mutex, not shared between processes.  */
    .mutexkind = PTHREAD_MUTEX_NORMAL
  };


static bool
prio_inherit_missing (void)
{
  static int tpi_supported;
  if (__glibc_unlikely (atomic_load_relaxed (&tpi_supported) == 0))
    {
      int e = futex_unlock_pi (&(unsigned int){0}, 0);
      atomic_store_relaxed (&tpi_supported, e == ENOSYS ? -1 : 1);
    }
  return __glibc_unlikely (tpi_supported < 0);
}

int
___pthread_mutex_init (pthread_mutex_t *mutex,
		      const pthread_mutexattr_t *mutexattr)
{
  const struct pthread_mutexattr *imutexattr;

  ASSERT_TYPE_SIZE (pthread_mutex_t, __SIZEOF_PTHREAD_MUTEX_T);

  /* __kind is the only field where its offset should be checked to
     avoid ABI breakage with static initializers.  */
  ASSERT_PTHREAD_INTERNAL_OFFSET (pthread_mutex_t, __data.__kind,
				  __PTHREAD_MUTEX_KIND_OFFSET);
  ASSERT_PTHREAD_INTERNAL_MEMBER_SIZE (pthread_mutex_t, __data.__kind, int);

  imutexattr = ((const struct pthread_mutexattr *) mutexattr
		?: &default_mutexattr);

  /* Sanity checks.  */
  switch (__builtin_expect (imutexattr->mutexkind
			    & PTHREAD_MUTEXATTR_PROTOCOL_MASK,
			    PTHREAD_PRIO_NONE
			    << PTHREAD_MUTEXATTR_PROTOCOL_SHIFT))
    {
    case PTHREAD_PRIO_NONE << PTHREAD_MUTEXATTR_PROTOCOL_SHIFT:
      break;

    case PTHREAD_PRIO_INHERIT << PTHREAD_MUTEXATTR_PROTOCOL_SHIFT:
      if (__glibc_unlikely (prio_inherit_missing ()))
	return ENOTSUP;
      break;

    default:
      /* XXX: For now we don't support robust priority protected mutexes.  */
      if (imutexattr->mutexkind & PTHREAD_MUTEXATTR_FLAG_ROBUST)
	return ENOTSUP;
      break;
    }

  /* Clear the whole variable.  */
  memset (mutex, '\0', __SIZEOF_PTHREAD_MUTEX_T);

  /* Copy the values from the attribute.  */
  int mutex_kind = imutexattr->mutexkind & ~PTHREAD_MUTEXATTR_FLAG_BITS;

  if ((imutexattr->mutexkind & PTHREAD_MUTEXATTR_FLAG_ROBUST) != 0)
    {
#ifndef __ASSUME_SET_ROBUST_LIST
      if ((imutexattr->mutexkind & PTHREAD_MUTEXATTR_FLAG_PSHARED) != 0
	  && !__nptl_set_robust_list_avail)
	return ENOTSUP;
#endif

      mutex_kind |= PTHREAD_MUTEX_ROBUST_NORMAL_NP;
    }

  switch (imutexattr->mutexkind & PTHREAD_MUTEXATTR_PROTOCOL_MASK)
    {
    case PTHREAD_PRIO_INHERIT << PTHREAD_MUTEXATTR_PROTOCOL_SHIFT:
      mutex_kind |= PTHREAD_MUTEX_PRIO_INHERIT_NP;
      break;

    case PTHREAD_PRIO_PROTECT << PTHREAD_MUTEXATTR_PROTOCOL_SHIFT:
      mutex_kind |= PTHREAD_MUTEX_PRIO_PROTECT_NP;

      int ceiling = (imutexattr->mutexkind
		     & PTHREAD_MUTEXATTR_PRIO_CEILING_MASK)
		    >> PTHREAD_MUTEXATTR_PRIO_CEILING_SHIFT;
      if (! ceiling)
	{
	  /* See __init_sched_fifo_prio.  */
	  if (atomic_load_relaxed (&__sched_fifo_min_prio) == -1)
	    __init_sched_fifo_prio ();
	  if (ceiling < atomic_load_relaxed (&__sched_fifo_min_prio))
	    ceiling = atomic_load_relaxed (&__sched_fifo_min_prio);
	}
      mutex->__data.__lock = ceiling << PTHREAD_MUTEX_PRIO_CEILING_SHIFT;
      break;

    default:
      break;
    }

  /* The kernel when waking robust mutexes on exit never uses
     FUTEX_PRIVATE_FLAG FUTEX_WAKE.  */
  if ((imutexattr->mutexkind & (PTHREAD_MUTEXATTR_FLAG_PSHARED
				| PTHREAD_MUTEXATTR_FLAG_ROBUST)) != 0)
    mutex_kind |= PTHREAD_MUTEX_PSHARED_BIT;

  /* See concurrency notes regarding __kind in struct __pthread_mutex_s
     in sysdeps/nptl/bits/thread-shared-types.h.  */
  atomic_store_relaxed (&(mutex->__data.__kind), mutex_kind);

  /* Default values: mutex not used yet.  */
  // mutex->__count = 0;	already done by memset
  // mutex->__owner = 0;	already done by memset
  // mutex->__nusers = 0;	already done by memset
  // mutex->__spins = 0;	already done by memset
  // mutex->__next = NULL;	already done by memset

  LIBC_PROBE (mutex_init, 1, mutex);

  __try_to_mark_as_unmigratable(mutex);

  return 0;
}
versioned_symbol (libpthread, ___pthread_mutex_init, pthread_mutex_init,
		  GLIBC_2_0);
libc_hidden_ver (___pthread_mutex_init, __pthread_mutex_init)
#ifndef SHARED
strong_alias (___pthread_mutex_init, __pthread_mutex_init)
#endif

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, ___pthread_mutex_init, __pthread_mutex_init,
	       GLIBC_2_0);
#endif
