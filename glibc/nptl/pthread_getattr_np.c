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
#include <inttypes.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include "pthreadP.h"
#include <lowlevellock.h>
#include <ldsodefs.h>


int
__pthread_getattr_np (pthread_t thread_id, pthread_attr_t *attr)
{
  struct pthread *thread = (struct pthread *) thread_id;

  /* Prepare the new thread attribute.  */
  int ret = __pthread_attr_init (attr);
  if (ret != 0)
    return ret;

  struct pthread_attr *iattr = (struct pthread_attr *) attr;

  lll_lock (thread->lock, LLL_PRIVATE);

  /* The thread library is responsible for keeping the values in the
     thread desriptor up-to-date in case the user changes them.  */
  memcpy (&iattr->schedparam, &thread->schedparam,
	  sizeof (struct sched_param));
  iattr->schedpolicy = thread->schedpolicy;

  /* Clear the flags work.  */
  iattr->flags = thread->flags;

  /* The thread might be detached by now.  */
  if (IS_DETACHED (thread))
    iattr->flags |= ATTR_FLAG_DETACHSTATE;

  /* This is the guardsize after adjusting it.  */
  iattr->guardsize = thread->reported_guardsize;

  /* The sizes are subject to alignment.  */
  if (__glibc_likely (thread->stackblock != NULL))
    {
      /* The stack size reported to the user should not include the
	 guard size.  */
      iattr->stacksize = thread->stackblock_size - thread->guardsize;
#if _STACK_GROWS_DOWN
      iattr->stackaddr = (char *) thread->stackblock
			 + thread->stackblock_size;
#else
      iattr->stackaddr = (char *) thread->stackblock;
#endif
    }
  else
    {
      /* No stack information available.  This must be for the initial
	 thread.  Get the info in some magical way.  */

      /* Stack size limit.  */
      struct rlimit rl;

      /* The safest way to get the top of the stack is to read
	 /proc/self/maps and locate the line into which
	 __libc_stack_end falls.  */
      FILE *fp = fopen ("/proc/self/maps", "rce");
      if (fp == NULL)
	ret = errno;
      /* We need the limit of the stack in any case.  */
      else
	{
	  if (__getrlimit (RLIMIT_STACK, &rl) != 0)
	    ret = errno;
	  else
	    {
	      /* We consider the main process stack to have ended with
	         the page containing __libc_stack_end.  There is stuff below
		 it in the stack too, like the program arguments, environment
		 variables and auxv info, but we ignore those pages when
		 returning size so that the output is consistent when the
		 stack is marked executable due to a loaded DSO requiring
		 it.  */
	      void *stack_end = (void *) ((uintptr_t) __libc_stack_end
					  & -(uintptr_t) GLRO(dl_pagesize));
#if _STACK_GROWS_DOWN
	      stack_end += GLRO(dl_pagesize);
#endif
	      /* We need no locking.  */
	      __fsetlocking (fp, FSETLOCKING_BYCALLER);

	      /* Until we found an entry (which should always be the case)
		 mark the result as a failure.  */
	      ret = ENOENT;

	      char *line = NULL;
	      size_t linelen = 0;
#if _STACK_GROWS_DOWN
	      uintptr_t last_to = 0;
#endif

	      while (! feof_unlocked (fp))
		{
		  if (__getline (&line, &linelen, fp) <= 0)
		    break;

		  uintptr_t from;
		  uintptr_t to;
		  if (sscanf (line, "%" SCNxPTR "-%" SCNxPTR, &from, &to) != 2)
		    continue;
		  if (from <= (uintptr_t) __libc_stack_end
		      && (uintptr_t) __libc_stack_end < to)
		    {
		      /* Found the entry.  Now we have the info we need.  */
		      iattr->stackaddr = stack_end;
		      iattr->stacksize =
		        rl.rlim_cur - (size_t) (to - (uintptr_t) stack_end);

		      /* Cut it down to align it to page size since otherwise we
		         risk going beyond rlimit when the kernel rounds up the
		         stack extension request.  */
		      iattr->stacksize = (iattr->stacksize
					  & -(intptr_t) GLRO(dl_pagesize));
#if _STACK_GROWS_DOWN
		      /* The limit might be too high.  */
		      if ((size_t) iattr->stacksize
			  > (size_t) iattr->stackaddr - last_to)
			iattr->stacksize = (size_t) iattr->stackaddr - last_to;
#else
		      /* The limit might be too high.  */
		      if ((size_t) iattr->stacksize
			  > to - (size_t) iattr->stackaddr)
			iattr->stacksize = to - (size_t) iattr->stackaddr;
#endif
		      /* We succeed and no need to look further.  */
		      ret = 0;
		      break;
		    }
#if _STACK_GROWS_DOWN
		  last_to = to;
#endif
		}

	      free (line);
	    }

	  fclose (fp);
	}
    }

  iattr->flags |= ATTR_FLAG_STACKADDR;

  if (ret == 0)
    {
      size_t size = 16;
      cpu_set_t *cpuset = NULL;

      do
	{
	  size <<= 1;

	  void *newp = realloc (cpuset, size);
	  if (newp == NULL)
	    {
	      ret = ENOMEM;
	      break;
	    }
	  cpuset = (cpu_set_t *) newp;

	  ret = __pthread_getaffinity_np (thread_id, size, cpuset);
	}
      /* Pick some ridiculous upper limit.  Is 8 million CPUs enough?  */
      while (ret == EINVAL && size < 1024 * 1024);

      if (ret == 0)
	ret = __pthread_attr_setaffinity_np (attr, size, cpuset);
      else if (ret == ENOSYS)
	/* There is no such functionality.  */
	ret = 0;
      free (cpuset);
    }

  lll_unlock (thread->lock, LLL_PRIVATE);

  if (ret != 0)
    __pthread_attr_destroy (attr);

  return ret;
}
versioned_symbol (libc, __pthread_getattr_np, pthread_getattr_np, GLIBC_2_32);

#if SHLIB_COMPAT (libc, GLIBC_2_2_3, GLIBC_2_32)
strong_alias (__pthread_getattr_np, __pthread_getattr_np_alias)
compat_symbol (libc, __pthread_getattr_np_alias,
	       pthread_getattr_np, GLIBC_2_2_3);
#endif
