/* Preemption of Hurd signals before POSIX.1 semantics take over.  Wrapper.
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

#ifndef _HURD_SIGPREEMPT_H
# include <hurd/hurd/sigpreempt.h>

# ifndef _ISOMAC
#  define HURD_PREEMPT_SIGNAL_P(preemptor, signo, sigcode) \
  (((preemptor)->signals & __sigmask (signo)) \
   && (sigcode) >= (preemptor)->first && (sigcode) <= (preemptor)->last)

/* Signal preemptors applying to all threads; locked by _hurd_siglock.  */
extern struct hurd_signal_preemptor *_hurdsig_preemptors;
extern sigset_t _hurdsig_preempted_set;

# endif /* _ISOMAC */
#endif /* _HURD_SIGPREEMPT_H */
