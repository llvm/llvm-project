/* Get or set a thread's signal mask.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>
#include <signal.h>

#include <pt-internal.h>

int
pthread_sigmask (int how, const sigset_t *set, sigset_t *oset)
{
  struct __pthread *self = _pthread_self ();

  /* Do not clear SELF's pending signals.  */
  return __pthread_sigstate (self, how, set, oset, 0);
}
