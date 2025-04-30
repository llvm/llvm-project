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
#include <paths.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <_itoa.h>

/* Directory where we can find the slave pty nodes.  */
#define _PATH_DEVPTS "/dev/pts/"

/* Static buffer for `ptsname'.  */
static char buffer[sizeof (_PATH_DEVPTS) + 20];


/* Return the pathname of the pseudo terminal slave associated with
   the master FD is open on, or NULL on errors.
   The returned storage is good until the next call to this function.  */
char *
ptsname (int fd)
{
  return __ptsname_r (fd, buffer, sizeof (buffer)) != 0 ? NULL : buffer;
}


/* Store at most BUFLEN characters of the pathname of the slave pseudo
   terminal associated with the master FD is open on in BUF.
   Return 0 on success, otherwise an error number.  */
int
__ptsname_r (int fd, char *buf, size_t buflen)
{
  int save_errno = errno;
  unsigned int ptyno;

  if (__ioctl (fd, TIOCGPTN, &ptyno) == 0)
    {
      /* Buffer we use to print the number in.  For a maximum size for
	 `int' of 8 bytes we never need more than 20 digits.  */
      char numbuf[21];
      const char *devpts = _PATH_DEVPTS;
      const size_t devptslen = strlen (_PATH_DEVPTS);
      char *p;

      numbuf[sizeof (numbuf) - 1] = '\0';
      p = _itoa_word (ptyno, &numbuf[sizeof (numbuf) - 1], 10, 0);

      if (buflen < devptslen + (&numbuf[sizeof (numbuf)] - p))
	{
	  __set_errno (ERANGE);
	  return ERANGE;
	}

      memcpy (__stpcpy (buf, devpts), p, &numbuf[sizeof (numbuf)] - p);
    }
  else
    /* Bad file descriptor, or not a ptmx descriptor.  */
    return errno;

  __set_errno (save_errno);
  return 0;
}
libc_hidden_def (__ptsname_r)
weak_alias (__ptsname_r, ptsname_r)
