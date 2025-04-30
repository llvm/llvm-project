/* __get_pthread_stack_min ().  Linux version.
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

/* Return sysconf (_SC_THREAD_STACK_MIN).  */

static inline long int
__get_pthread_stack_min (void)
{
  /* sysconf (_SC_THREAD_STACK_MIN) >= sysconf (_SC_MINSIGSTKSZ).  */
  long int pthread_stack_min = GLRO(dl_minsigstacksize);
  assert (pthread_stack_min != 0);
  _Static_assert (__builtin_constant_p (PTHREAD_STACK_MIN),
		  "PTHREAD_STACK_MIN is constant");
  /* Return MAX (PTHREAD_STACK_MIN, pthread_stack_min).  */
  if (pthread_stack_min < PTHREAD_STACK_MIN)
    pthread_stack_min = PTHREAD_STACK_MIN;
  /* We have a private interface, __pthread_get_minstack@GLIBC_PRIVATE
     which returns a larger size that includes the required TLS variable
     space which has been determined at startup.  For sysconf here we are
     conservative and don't include the space required for TLS access.
     Eventually the TLS variable space will not be part of the stack
     (Bug 11787).  */
  return pthread_stack_min;
}
