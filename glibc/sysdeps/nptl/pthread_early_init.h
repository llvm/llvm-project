/* pthread initialization called from __libc_early_init.  NPTL version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _PTHREAD_EARLY_INIT_H
#define _PTHREAD_EARLY_INIT_H 1

#include <nptl/nptl-stack.h>
#include <pthreadP.h>
#include <pthread_mutex_conf.h>
#include <sys/resource.h>

static inline void
__pthread_early_init (void)
{
  /* Determine the default allowed stack size.  This is the size used
     in case the user does not specify one.  */
  struct rlimit limit;
  if (__getrlimit (RLIMIT_STACK, &limit) != 0
      || limit.rlim_cur == RLIM_INFINITY)
    /* The system limit is not usable.  Use an architecture-specific
       default.  */
    limit.rlim_cur = ARCH_STACK_DEFAULT_SIZE;
  else if (limit.rlim_cur < PTHREAD_STACK_MIN)
    /* The system limit is unusably small.
       Use the minimal size acceptable.  */
    limit.rlim_cur = PTHREAD_STACK_MIN;

  /* Make sure it meets the minimum size that allocate_stack
     (allocatestack.c) will demand, which depends on the page size.  */
  const uintptr_t pagesz = GLRO(dl_pagesize);
  const size_t minstack = (pagesz + __nptl_tls_static_size_for_stack ()
                           + MINIMAL_REST_STACK);
  if (limit.rlim_cur < minstack)
    limit.rlim_cur = minstack;

  /* Round the resource limit up to page size.  */
  limit.rlim_cur = ALIGN_UP (limit.rlim_cur, pagesz);
  __default_pthread_attr.internal.stacksize = limit.rlim_cur;
  __default_pthread_attr.internal.guardsize = GLRO (dl_pagesize);

#if HAVE_TUNABLES
  __pthread_tunables_init ();
#endif
}

#endif  /* _PTHREAD_EARLY_INIT_H */
