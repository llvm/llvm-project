/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

#include <hurd/sigpreempt.h>
#include <hurd/signal.h>
#include <assert.h>

void
hurd_preempt_signals (struct hurd_signal_preemptor *preemptor)
{
  __mutex_lock (&_hurd_siglock);
  preemptor->next = _hurdsig_preemptors;
  _hurdsig_preemptors = preemptor;
  _hurdsig_preempted_set |= preemptor->signals;
  __mutex_unlock (&_hurd_siglock);
}

void
hurd_unpreempt_signals (struct hurd_signal_preemptor *preemptor)
{
  struct hurd_signal_preemptor **p;
  sigset_t preempted = 0;

  __mutex_lock (&_hurd_siglock);

  p = &_hurdsig_preemptors;
  while (*p)
    if (*p == preemptor)
      {
	/* Found it; take it off the chain.  */
	*p = (*p)->next;
	if ((preemptor->signals & preempted) != preemptor->signals)
	  {
	    /* This might have been the only preemptor for some
	       of those signals, so we must collect the full mask
	       from the others.  */
	    struct hurd_signal_preemptor *pp;
	    for (pp = *p; pp; pp = pp->next)
	      preempted |= pp->signals;
	    _hurdsig_preempted_set = preempted;
	  }
	__mutex_unlock (&_hurd_siglock);
	return;
      }
    else
      {
	preempted |= (*p)->signals;
	p = &(*p)->next;
      }

  __mutex_unlock (&_hurd_siglock); /* Avoid deadlock during death rattle.  */
  assert (! "removing absent preemptor");
}
