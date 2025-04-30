/* Sleep for a given number of seconds.  POSIX.1 version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <sys/param.h>


/* Make the process sleep for SECONDS seconds, or until a signal arrives
   and is not ignored.  The function returns the number of seconds less
   than SECONDS which it actually slept (zero if it slept the full time).
   If a signal handler does a `longjmp' or modifies the handling of the
   SIGALRM signal while inside `sleep' call, the handling of the SIGALRM
   signal afterwards is undefined.  There is no return value to indicate
   error, but if `sleep' returns SECONDS, it probably didn't work.  */
unsigned int
__sleep (unsigned int seconds)
{
  int save_errno = errno;

  const unsigned int max
    = (unsigned int) (((unsigned long int) (~((time_t) 0))) >> 1);
  struct timespec ts = { 0, 0 };
  do
    {
      if (sizeof (ts.tv_sec) <= sizeof (seconds))
        {
          /* Since SECONDS is unsigned assigning the value to .tv_sec can
             overflow it.  In this case we have to wait in steps.  */
          ts.tv_sec += MIN (seconds, max);
          seconds -= (unsigned int) ts.tv_sec;
        }
      else
        {
          ts.tv_sec = (time_t) seconds;
          seconds = 0;
        }

      if (__nanosleep (&ts, &ts) < 0)
        /* We were interrupted.
           Return the number of (whole) seconds we have not yet slept.  */
        return seconds + ts.tv_sec;
    }
  while (seconds > 0);

  __set_errno (save_errno);

  return 0;
}
weak_alias (__sleep, sleep)
