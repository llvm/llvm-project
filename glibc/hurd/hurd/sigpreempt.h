/* Preemption of Hurd signals before POSIX.1 semantics take over.
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

#ifndef	_HURD_SIGPREEMPT_H

#define	_HURD_SIGPREEMPT_H	1
#define __need_size_t
#include <stddef.h>
#include <errno.h>
#include <bits/types/error_t.h>
#include <signal.h>		/* For sighandler_t, SIG_ERR.  */
#include <bits/types/sigset_t.h>
struct hurd_sigstate;		/* <hurd/signal.h> */
struct hurd_signal_detail;	/* <hurd/signal.h> */

struct hurd_signal_preemptor
  {
    /* These members select which signals this structure will apply to.
       The rest of the structure is only consulted if these match.  */
    sigset_t signals;		/* Signals preempted.  */
    unsigned long int first, last; /* Range of sigcode values preempted.  */

    /* This function will be called (with SS->lock held) to decide what to
       do with the signal described.  It may modify the codes of the signal
       passed.  If the return value is SIG_ERR, the next matching preemptor
       is tried, or the normal handling is done for the signal (which may
       have been changed by the preemptor function).  Otherwise, the signal
       is processed as if the return value were its handler setting.  */
    __sighandler_t (*preemptor) (struct hurd_signal_preemptor *preemptor,
			         struct hurd_sigstate *ss,
			         int *signo, struct hurd_signal_detail *detail);
    /* If PREEMPTOR is null, act as if it returned HANDLER.  */
    __sighandler_t handler;

    struct hurd_signal_preemptor *next;	/* List structure.  */
  };

/* The caller must initialize all members of *PREEMPTOR except `next'.
   The preemptor is registered on the global list.  */
void hurd_preempt_signals (struct hurd_signal_preemptor *preemptor);

/* Remove a preemptor registered with hurd_preempt_signals.  */
void hurd_unpreempt_signals (struct hurd_signal_preemptor *preemptor);


/* Call *OPERATE and return its value.  If a signal in SIGSET with a sigcode
   in the range [FIRST,LAST] arrives during the call, catch it.  If HANDLER
   is a function, it handles the signal in the normal way (i.e. it should
   longjmp unless it can restart the insn on return).  If it is SIG_ERR,
   hurd_catch_signal returns the sc_error value from the signal (or
   EGRATUITOUS if that is zero).

   The preemptor structure is passed to *OPERATE, which may modify its
   sigcode range or functions at any time during which it is guaranteed no
   signal in SIGSET will arrive.  */

error_t hurd_catch_signal (sigset_t sigset,
			   unsigned long int first, unsigned long int last,
			   error_t (*operate) (struct hurd_signal_preemptor *),
			   __sighandler_t handler);


/* Convenience functions using `hurd_catch_signal'.  */


/* Like `memset', but catch faults in DEST.  */
error_t hurd_safe_memset (void *dest, int byte, size_t nbytes);

/* Like `memcpy', but catch faults in SRC.  */
error_t hurd_safe_copyin (void *dest, const void *src, size_t nbytes);

/* Like `memcpy', but catch faults in DEST.  */
error_t hurd_safe_copyout (void *dest, const void *src, size_t nbytes);

/* Like `memmove', but catch faults in SRC or DEST.
   If only one region is expected to fault, it is more efficient
   to use `hurd_safe_copyin' or `hurd_safe_copyout' as appropriate.  */
error_t hurd_safe_memmove (void *dest, const void *src, size_t nbytes);


#endif	/* hurd/sigpreempt.h */
