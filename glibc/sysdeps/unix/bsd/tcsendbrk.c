/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stddef.h>
#include <signal.h>
#include <termios.h>
#include <unistd.h>
#include "bsdtty.h"
#include <sys/file.h>
#include <sys/time.h>
#include <sys/types.h>

/* Send zero bits on FD.  */
int
tcsendbreak (int fd, int duration)
{
  struct timeval delay;

  /* The break lasts 0.25 to 0.5 seconds if DURATION is zero,
     and an implementation-defined period if DURATION is nonzero.
     We define a positive DURATION to be number of microseconds to break.  */
  if (duration <= 0)
    duration = 400000;

  delay.tv_sec = 0;
  delay.tv_usec = duration;

  /* Starting sending break.  */
  if (__ioctl (fd, TIOCSBRK, (void *) NULL) < 0)
    return -1;

  /* Wait DURATION microseconds.  */
  (void) __select (0, (fd_set *) NULL, (fd_set *) NULL, (fd_set *) NULL,
		   &delay);

  /* Turn off the break.  */
  return __ioctl (fd, TIOCCBRK, (void *) NULL);
}
