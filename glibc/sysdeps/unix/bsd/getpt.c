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
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

/* Prefix for master pseudo terminal nodes.  */
#define _PATH_PTY "/dev/pty"


/* Letters indicating a series of pseudo terminals.  */
#ifndef PTYNAME1
#define PTYNAME1 "pqrsPQRS"
#endif
const char __libc_ptyname1[] attribute_hidden = PTYNAME1;

/* Letters indicating the position within a series.  */
#ifndef PTYNAME2
#define PTYNAME2 "0123456789abcdefghijklmnopqrstuv";
#endif
const char __libc_ptyname2[] attribute_hidden = PTYNAME2;


/* Open a master pseudo terminal and return its file descriptor.  */
int
__bsd_openpt (int oflag)
{
  char buf[sizeof (_PATH_PTY) + 2];
  const char *p, *q;
  char *s;

  s = __mempcpy (buf, _PATH_PTY, sizeof (_PATH_PTY) - 1);
  /* s[0] and s[1] will be filled in the loop.  */
  s[2] = '\0';

  for (p = __libc_ptyname1; *p != '\0'; ++p)
    {
      s[0] = *p;

      for (q = __libc_ptyname2; *q != '\0'; ++q)
	{
	  int fd;

	  s[1] = *q;

	  fd = __open (buf, oflag);
	  if (fd != -1)
	    return fd;

	  if (errno == ENOENT)
	    return -1;
	}
    }

  __set_errno (ENOENT);
  return -1;
}

int
__getpt (void)
{
  return __bsd_openpt (O_RDWR);
}
libc_hidden_def (__getpt)
weak_alias (__getpt, getpt)

int
__posix_openpt (int oflag)
{
  return __bsd_openpt (oflag);
}
weak_alias (__posix_openpt, posix_openpt)
