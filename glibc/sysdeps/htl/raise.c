/* raise.c - Generic raise implementation.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.
   Written by Neal H. Walfield <neal@gnu.org>.

   This file is part of the GNU Hurd.

   The GNU Hurd is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License
   as published by the Free Software Foundation; either version 3 of
   the License, or (at your option) any later version.

   The GNU Hurd is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this program.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthreadP.h>
#include <signal.h>
#include <unistd.h>

#pragma weak __pthread_kill
#pragma weak __pthread_self
#pragma weak __pthread_threads
int
raise (int signo)
{
  /* According to POSIX, if we implement threads (and we do), then
     "the effect of the raise() function shall be equivalent to
     calling: pthread_kill(pthread_self(), sig);"  */

  if (__pthread_kill != NULL && __pthread_threads != NULL)
    {
      int err;
      err = __pthread_kill (__pthread_self (), signo);
      if (err)
	{
	  errno = err;
	  return -1;
	}
      return 0;
    }
  else
    return __kill (__getpid (), signo);
}

libc_hidden_def (raise)
weak_alias (raise, gsignal)
