/* Convenience function to catch expected signals during an operation.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <hurd/signal.h>
#include <hurd/sigpreempt.h>
#include <string.h>
#include <assert.h>

error_t
__hurd_catch_signal (sigset_t sigset,
		     unsigned long int first, unsigned long int last,
		     error_t (*operate) (struct hurd_signal_preemptor *),
		     sighandler_t handler)
{
  /* We need to restore the signal mask, because otherwise the
     signal-handling code will have blocked the caught signal and for
     instance calling hurd_catch_signal again would then dump core.  */
  sigjmp_buf buf;
  void throw (int signo, long int sigcode, struct sigcontext *scp)
    { siglongjmp (buf, scp->sc_error ?: EGRATUITOUS); }

  struct hurd_signal_preemptor preemptor =
    {
      sigset, first, last,
      NULL, handler == SIG_ERR ? (sighandler_t) &throw : handler,
    };

  struct hurd_sigstate *const ss = _hurd_self_sigstate ();
  error_t error;

  if (handler != SIG_ERR)
    /* Not our handler; don't bother saving state.  */
    error = 0;
  else
    /* This returns again with nonzero value when we preempt a signal.  */
    error = sigsetjmp (buf, 1);

  if (error == 0)
    {
      /* Install a signal preemptor for the thread.  */
      __spin_lock (&ss->lock);
      preemptor.next = ss->preemptors;
      ss->preemptors = &preemptor;
      __spin_unlock (&ss->lock);

      /* Try the operation that might crash.  */
      (*operate) (&preemptor);
    }

  /* Either FUNCTION completed happily and ERROR is still zero, or it hit
     an expected signal and `throw' made setjmp return the signal error
     code in ERROR.  Now we can remove the preemptor and return.  */

  __spin_lock (&ss->lock);
  assert (ss->preemptors == &preemptor);
  ss->preemptors = preemptor.next;
  __spin_unlock (&ss->lock);

  return error;
}
strong_alias (__hurd_catch_signal, hurd_catch_signal)


error_t
hurd_safe_memset (void *dest, int byte, size_t nbytes)
{
  error_t operate (struct hurd_signal_preemptor *preemptor)
    {
      memset (dest, byte, nbytes);
      return 0;
    }
  return __hurd_catch_signal (__sigmask (SIGBUS) | __sigmask (SIGSEGV),
			      (vm_address_t) dest, (vm_address_t) dest + nbytes,
			      &operate, SIG_ERR);
}


error_t
hurd_safe_copyout (void *dest, const void *src, size_t nbytes)
{
  error_t operate (struct hurd_signal_preemptor *preemptor)
    {
      memcpy (dest, src, nbytes);
      return 0;
    }
  return __hurd_catch_signal (__sigmask (SIGBUS) | __sigmask (SIGSEGV),
			      (vm_address_t) dest, (vm_address_t) dest + nbytes,
			      &operate, SIG_ERR);
}

error_t
hurd_safe_copyin (void *dest, const void *src, size_t nbytes)
{
  error_t operate (struct hurd_signal_preemptor *preemptor)
    {
      memcpy (dest, src, nbytes);
      return 0;
    }
  return __hurd_catch_signal (__sigmask (SIGBUS) | __sigmask (SIGSEGV),
			      (vm_address_t) src, (vm_address_t) src + nbytes,
			      &operate, SIG_ERR);
}

error_t
hurd_safe_memmove (void *dest, const void *src, size_t nbytes)
{
  jmp_buf buf;
  void throw (int signo, long int sigcode, struct sigcontext *scp)
    { longjmp (buf, scp->sc_error ?: EGRATUITOUS); }

  struct hurd_signal_preemptor src_preemptor =
    {
      __sigmask (SIGBUS) | __sigmask (SIGSEGV),
      (vm_address_t) src, (vm_address_t) src + nbytes,
      NULL, (sighandler_t) &throw,
    };
  struct hurd_signal_preemptor dest_preemptor =
    {
      __sigmask (SIGBUS) | __sigmask (SIGSEGV),
      (vm_address_t) dest, (vm_address_t) dest + nbytes,
      NULL, (sighandler_t) &throw,
      &src_preemptor
    };

  struct hurd_sigstate *const ss = _hurd_self_sigstate ();
  error_t error;

  /* This returns again with nonzero value when we preempt a signal.  */
  error = setjmp (buf);

  if (error == 0)
    {
      /* Install a signal preemptor for the thread.  */
      __spin_lock (&ss->lock);
      src_preemptor.next = ss->preemptors;
      ss->preemptors = &dest_preemptor;
      __spin_unlock (&ss->lock);

      /* Do the copy; it might fault.  */
      memmove (dest, src, nbytes);
    }

  /* Either memmove completed happily and ERROR is still zero, or it hit
     an expected signal and `throw' made setjmp return the signal error
     code in ERROR.  Now we can remove the preemptor and return.  */

  __spin_lock (&ss->lock);
  assert (ss->preemptors == &dest_preemptor);
  ss->preemptors = src_preemptor.next;
  __spin_unlock (&ss->lock);

  return error;
}
