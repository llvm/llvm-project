/* Send break to terminal.
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

#include <errno.h>
#include <stddef.h>
#include <termios.h>
#include <sys/ioctl.h>

/* Send zero bits on FD.  */
int
tcsendbreak (int fd, int duration)
{
  /* The break lasts 0.25 to 0.5 seconds if DURATION is zero,
     and an implementation-defined period if DURATION is nonzero.
     We define a positive DURATION to be number of milliseconds to break.  */
  if (duration <= 0)
    return __ioctl (fd, TCSBRK, 0);

#ifdef TCSBRKP
  /* Probably Linux-specific: a positive third TCSBRKP ioctl argument is
     defined to be the number of 100ms units to break.  */
  return __ioctl (fd, TCSBRKP, (duration + 99) / 100);
#else
  /* ioctl can't send a break of any other duration for us.
     This could be changed to use trickery (e.g. lower speed and
     send a '\0') to send the break, but for now just return an error.  */
  return INLINE_SYSCALL_ERROR_RETURN_VALUE (EINVAL);
#endif
}
