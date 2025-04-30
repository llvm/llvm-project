/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Zack Weinberg <zack@rabi.phys.columbia.edu>, 1998.

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
#include <stdlib.h>
#include <sys/ioctl.h>
#include <termios.h>


/* Unlock the slave pseudo terminal associated with the master pseudo
   terminal specified by FD.  */
int
unlockpt (int fd)
{
  int unlock = 0;

  int ret = __ioctl (fd, TIOCSPTLCK, &unlock);
  if (ret != 0 && errno == ENOTTY)
    /* POSIX mandates EINVAL for non-ptmx descriptors.  */
    __set_errno (EINVAL);
  return ret;
}
libc_hidden_def (unlockpt)
