/* `struct termios' speed frobnication functions.  4.4 BSD/generic GNU version.
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

#include <stddef.h>
#include <errno.h>
#include <termios.h>

/* Return the output baud rate stored in *TERMIOS_P.  */
speed_t
cfgetospeed (const struct termios *termios_p)
{
  return termios_p->__ospeed;
}

/* Return the input baud rate stored in *TERMIOS_P.  */
speed_t
cfgetispeed (const struct termios *termios_p)
{
  return termios_p->__ispeed;
}

/* Set the output baud rate stored in *TERMIOS_P to SPEED.  */
int
cfsetospeed (struct termios *termios_p, speed_t speed)
{
  if (termios_p == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  termios_p->__ospeed = speed;
  return 0;
}
libc_hidden_def (cfsetospeed)

/* Set the input baud rate stored in *TERMIOS_P to SPEED.  */
int
cfsetispeed (struct termios *termios_p, speed_t speed)
{
  if (termios_p == NULL)
    {
      __set_errno (EINVAL);
      return -1;
    }

  termios_p->__ispeed = speed;
  return 0;
}
libc_hidden_def (cfsetispeed)
