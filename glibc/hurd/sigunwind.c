/* longjmp cleanup function for unwinding past signal handlers.
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

#include <hurd.h>
#include <thread_state.h>
#include <hurd/threadvar.h>
#include <jmpbuf-unwind.h>
#include <assert.h>
#include <stdint.h>


/* _hurd_setup_sighandler puts a link on the `active resources' chain so that
   _longjmp_unwind will call this function with the `struct sigcontext *'
   describing the context interrupted by the signal, when `longjmp' is jumping
   to an environment that unwinds past the interrupted frame.  */

void
_hurdsig_longjmp_from_handler (void *data, jmp_buf env, int val)
{
  struct sigcontext *scp = data;
  struct hurd_sigstate *ss = _hurd_self_sigstate ();
  int onstack;
  inline void cleanup (void)
    {
      /* Destroy the MiG reply port used by the signal handler, and restore
	 the reply port in use by the thread when interrupted.  */
      mach_port_t *reply_port = &__hurd_local_reply_port;
      if (*reply_port)
	{
	  mach_port_t port = *reply_port;
	  /* Assigning MACH_PORT_DEAD here tells libc's mig_get_reply_port
	     not to get another reply port, but avoids mig_dealloc_reply_port
	     trying to deallocate it after the receive fails (which it will,
	     because the reply port will be bogus, regardless).  */
	  *reply_port = MACH_PORT_DEAD;
	  __mach_port_destroy (__mach_task_self (), port);
	}
      if (scp->sc_reply_port)
	__mach_port_destroy (__mach_task_self (), scp->sc_reply_port);
    }

  __spin_lock (&ss->lock);
  /* We should only ever be called from _longjmp_unwind (in jmp-unwind.c),
     which calls us inside a critical section.  */
  assert (__spin_lock_locked (&ss->critical_section_lock));
  /* Are we on the alternate signal stack now?  */
  onstack = (ss->sigaltstack.ss_flags & SS_ONSTACK);
  __spin_unlock (&ss->lock);

  if (onstack && ! scp->sc_onstack)
    {
      /* We are unwinding off the signal stack.  We must use sigreturn to
	 do it robustly.  Mutate the sigcontext so that when sigreturn
	 resumes from that context, it will be as if `__longjmp (ENV, VAL)'
	 were done.  */

      struct hurd_userlink *link;

      inline uintptr_t demangle_ptr (uintptr_t x)
	{
# ifdef PTR_DEMANGLE
	  PTR_DEMANGLE (x);
# endif
	  return x;
	}

      /* Continue _longjmp_unwind's job of running the unwind
	 forms for frames being unwound, since we will not
	 return to its loop like this one, which called us.  */
      for (link = ss->active_resources;
	   link && _JMPBUF_UNWINDS (env[0].__jmpbuf, link, demangle_ptr);
	   link = link->thread.next)
	if (_hurd_userlink_unlink (link))
	  {
	    if (link->cleanup == &_hurdsig_longjmp_from_handler)
	      {
		/* We are unwinding past another signal handler invocation.
		   Just finish the cleanup for this (inner) one, and then
		   swap SCP to restore to the outer context.  */
		cleanup ();
		scp = link->cleanup_data;
	      }
	    else
	      (*link->cleanup) (link->cleanup_data, env, val);
	  }

#define sc_machine_thread_state paste(sc_,machine_thread_state)
#define paste(a,b)	paste1(a,b)
#define paste1(a,b)	a##b

      /* There are no more unwind forms to be run!
	 Now we can just have the sigreturn do the longjmp for us.  */
      _hurd_longjmp_thread_state
	((struct machine_thread_state *) &scp->sc_machine_thread_state,
	 env, val);

      /* Restore to the same current signal mask.  If sigsetjmp saved the
	 mask, longjmp has already restored it as desired; if not, we
	 should leave it as it is.  */
      scp->sc_mask = ss->blocked;

      /* sigreturn expects the link added by _hurd_setup_sighandler
	 to still be there, but _longjmp_unwind removed it just before
	 calling us.  Put it back now so sigreturn can find it.  */
      link = (void *) &scp[1];
      assert (! link->resource.next && ! link->resource.prevp);
      assert (link->thread.next == ss->active_resources);
      assert (link->thread.prevp == &ss->active_resources);
      if (link->thread.next)
	link->thread.next->thread.prevp = &link->thread.next;
      ss->active_resources = link;

      /* We must momentarily exit the critical section so that sigreturn
	 does not get upset with us.  But we don't want signal handlers
	 running right now, because we are presently in the bogus state of
	 having run all the unwind forms back to ENV's frame, but our SP is
	 still inside those unwound frames.  */
      __spin_lock (&ss->lock);
      __spin_unlock (&ss->critical_section_lock);
      ss->blocked = ~(sigset_t) 0 & ~_SIG_CANT_MASK;
      __spin_unlock (&ss->lock);

      /* Restore to the modified signal context that now
	 performs `longjmp (ENV, VAL)'.  */
      __sigreturn (scp);
      assert (! "sigreturn returned!");
    }

  /* We are not unwinding off the alternate signal stack.  So nothing
     really funny is going on here.  We can just clean up this handler
     frame and let _longjmp_unwind continue unwinding.  */
  cleanup ();
  ss->intr_port = scp->sc_intr_port;
}
