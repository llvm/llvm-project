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
#include <termios.h>
#include <unistd.h>

#include "bsdtty.h"

/* Suspend or restart transmission on FD.  */
int
tcflow (int fd, int action)
{
  switch (action)
    {
    case TCOOFF:
      return __ioctl (fd, TIOCSTOP, (void *) NULL);
    case TCOON:
      return __ioctl (fd, TIOCSTART, (void *) NULL);

    case TCIOFF:
    case TCION:
      {
	/* This just writes the START or STOP character with
	   `write'.  Is there another way to do this?  */
	struct termios attr;
	unsigned char c;
	if (__tcgetattr (fd, &attr) < 0)
	  return -1;
	c = attr.c_cc[action == TCIOFF ? VSTOP : VSTART];
	if (c != _POSIX_VDISABLE && write (fd, &c, 1) < 1)
	  return -1;
	return 0;
      }

    default:
      __set_errno (EINVAL);
      return -1;
    }
}
