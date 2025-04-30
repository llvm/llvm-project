/* _Fork implementation.  Linux version.
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

#include <arch-fork.h>
#include <pthreadP.h>

/* Pointer to the fork generation counter in the thread library.  */
extern unsigned long int *__fork_generation_pointer attribute_hidden;

pid_t
_Fork (void)
{
  pid_t pid = arch_fork (&THREAD_SELF->tid);
  if (pid == 0)
    {
      struct pthread *self = THREAD_SELF;

      /* Initialize the robust mutex list setting in the kernel which has
	 been reset during the fork.  We do not check for errors because if
	 it fails here, it must have failed at process startup as well and
	 nobody could have used robust mutexes.
	 Before we do that, we have to clear the list of robust mutexes
	 because we do not inherit ownership of mutexes from the parent.
	 We do not have to set self->robust_head.futex_offset since we do
	 inherit the correct value from the parent.  We do not need to clear
	 the pending operation because it must have been zero when fork was
	 called.  */
#if __PTHREAD_MUTEX_HAVE_PREV
      self->robust_prev = &self->robust_head;
#endif
      self->robust_head.list = &self->robust_head;
      INTERNAL_SYSCALL_CALL (set_robust_list, &self->robust_head,
			     sizeof (struct robust_list_head));
    }
  return pid;
}
libc_hidden_def (_Fork)
