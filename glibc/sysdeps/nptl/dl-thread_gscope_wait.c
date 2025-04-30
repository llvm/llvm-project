/* Out-of-line notification function for the GSCOPE locking mechanism.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#include <nptl/descr.h>
#include <futex-internal.h>
#include <ldsodefs.h>
#include <list.h>
#include <lowlevellock.h>

void
__thread_gscope_wait (void)
{
  lll_lock (GL (dl_stack_cache_lock), LLL_PRIVATE);

  struct pthread *self = THREAD_SELF;

  /* Iterate over the list with system-allocated threads first.  */
  list_t *runp;
  list_for_each (runp, &GL (dl_stack_used))
    {
      struct pthread *t = list_entry (runp, struct pthread, list);
      if (t == self || t->header.gscope_flag == THREAD_GSCOPE_FLAG_UNUSED)
        continue;

      int *const gscope_flagp = &t->header.gscope_flag;

      /* We have to wait until this thread is done with the global
         scope.  First tell the thread that we are waiting and
         possibly have to be woken.  */
      if (atomic_compare_and_exchange_bool_acq (gscope_flagp,
                                                THREAD_GSCOPE_FLAG_WAIT,
                                                THREAD_GSCOPE_FLAG_USED))
        continue;

      do
        futex_wait_simple ((unsigned int *) gscope_flagp,
                           THREAD_GSCOPE_FLAG_WAIT, FUTEX_PRIVATE);
      while (*gscope_flagp == THREAD_GSCOPE_FLAG_WAIT);
    }

  /* Now the list with threads using user-allocated stacks.  */
  list_for_each (runp, &GL (dl_stack_user))
    {
      struct pthread *t = list_entry (runp, struct pthread, list);
      if (t == self || t->header.gscope_flag == THREAD_GSCOPE_FLAG_UNUSED)
        continue;

      int *const gscope_flagp = &t->header.gscope_flag;

      /* We have to wait until this thread is done with the global
         scope.  First tell the thread that we are waiting and
         possibly have to be woken.  */
      if (atomic_compare_and_exchange_bool_acq (gscope_flagp,
                                                THREAD_GSCOPE_FLAG_WAIT,
                                                THREAD_GSCOPE_FLAG_USED))
        continue;

      do
        futex_wait_simple ((unsigned int *) gscope_flagp,
                           THREAD_GSCOPE_FLAG_WAIT, FUTEX_PRIVATE);
      while (*gscope_flagp == THREAD_GSCOPE_FLAG_WAIT);
    }

  lll_unlock (GL (dl_stack_cache_lock), LLL_PRIVATE);
}
