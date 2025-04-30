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

static int bad_speed (speed_t speed);

/* Set the state of FD to *TERMIOS_P.  */
int
tcsetattr (int fd, int optional_actions, const struct termios *termios_p)
{
  if (fd < 0)
    {
      __set_errno (EBADF);
      return -1;
    }
  if (termios_p == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }
  switch (optional_actions)
    {
    case TCSANOW:
    case TCSADRAIN:
    case TCSAFLUSH:
      break;
    default:
      __set_errno (EINVAL);
      return -1;
    }

  if (bad_speed(termios_p->__ospeed)
      || bad_speed(termios_p->__ispeed == 0
		   ? termios_p->__ospeed : termios_p->__ispeed))
    {
      __set_errno (EINVAL);
      return -1;
    }

  __set_errno (ENOSYS);
  return -1;
}
libc_hidden_def (tcsetattr)

/* Strychnine checking.  */
static int
bad_speed (speed_t speed)
{
  switch (speed)
    {
    case B0:
    case B50:
    case B75:
    case B110:
    case B134:
    case B150:
    case B200:
    case B300:
    case B600:
    case B1200:
    case B1800:
    case B2400:
    case B4800:
    case B9600:
    case B19200:
    case B38400:
    case B57600:
    case B115200:
    case B230400:
    case B460800:
    case B500000:
    case B576000:
    case B921600:
    case B1000000:
    case B1152000:
    case B1500000:
    case B2000000:
    case B2500000:
    case B3000000:
    case B3500000:
    case B4000000:
      return 0;
    default:
      return 1;
    }
}


stub_warning (tcsetattr)
