/* _longjmp_unwind -- Clean up stack frames unwound by longjmp.  Hurd version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <jmpbuf-unwind.h>
#include <hurd/userlink.h>
#include <hurd/signal.h>
#include <hurd/sigpreempt.h>
#include <assert.h>
#include <stdint.h>


#ifndef _JMPBUF_UNWINDS
#error "<jmpbuf-unwind.h> fails to define _JMPBUF_UNWINDS"
#endif

static inline uintptr_t
demangle_ptr (uintptr_t x)
{
# ifdef PTR_DEMANGLE
  PTR_DEMANGLE (x);
# endif
  return x;
}

/* This function is called by `longjmp' (with its arguments) to restore
   active resources to a sane state before the frames code using them are
   jumped out of.  */

void
_longjmp_unwind (jmp_buf env, int val)
{
  struct hurd_sigstate *ss = _hurd_self_sigstate ();
  struct hurd_userlink *link;

  /* All access to SS->active_resources must take place inside a critical
     section where signal handlers cannot run.  */
  __spin_lock (&ss->lock);
  assert (! __spin_lock_locked (&ss->critical_section_lock));
  __spin_lock (&ss->critical_section_lock);

  /* Remove local signal preemptors being unwound past.  */
  while (ss->preemptors
	 && _JMPBUF_UNWINDS (env[0].__jmpbuf, ss->preemptors, demangle_ptr))
    ss->preemptors = ss->preemptors->next;

  __spin_unlock (&ss->lock);

  /* Iterate over the current thread's list of active resources.
     Process the head portion of the list whose links reside
     in stack frames being unwound by this jump.  */

  for (link = ss->active_resources;
       link && _JMPBUF_UNWINDS (env[0].__jmpbuf, link, demangle_ptr);
       link = link->thread.next)
    /* Remove this link from the resource's users list,
       since the frame using the resource is being unwound.
       This call returns nonzero if that was the last user.  */
    if (_hurd_userlink_unlink (link))
      /* One of the frames being unwound by the longjmp was the last user
	 of its resource.  Call the cleanup function to deallocate it.  */
      (*link->cleanup) (link->cleanup_data, env, val);

  _hurd_critical_section_unlock (ss);
}
