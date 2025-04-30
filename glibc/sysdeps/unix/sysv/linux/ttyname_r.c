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
#include <limits.h>
#include <stddef.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#include <fd_to_filename.h>

#include "ttyname.h"

static int getttyname_r (char *buf, size_t buflen,
			 const struct __stat64_t64 *mytty, int save,
			 int *dostat);

static int
attribute_compat_text_section
getttyname_r (char *buf, size_t buflen, const struct __stat64_t64 *mytty,
	      int save, int *dostat)
{
  struct __stat64_t64 st;
  DIR *dirstream;
  struct dirent64 *d;
  size_t devlen = strlen (buf);

  dirstream = __opendir (buf);
  if (dirstream == NULL)
    {
      *dostat = -1;
      return errno;
    }

  while ((d = __readdir64 (dirstream)) != NULL)
    if ((d->d_fileno == mytty->st_ino || *dostat)
	&& strcmp (d->d_name, "stdin")
	&& strcmp (d->d_name, "stdout")
	&& strcmp (d->d_name, "stderr"))
      {
	char *cp;
	size_t needed = _D_EXACT_NAMLEN (d) + 1;

	if (needed > buflen)
	  {
	    *dostat = -1;
	    (void) __closedir (dirstream);
	    __set_errno (ERANGE);
	    return ERANGE;
	  }

	cp = __stpncpy (buf + devlen, d->d_name, needed);
	cp[0] = '\0';

	if (__stat64_time64 (buf, &st) == 0
	    && is_mytty (mytty, &st))
	  {
	    (void) __closedir (dirstream);
	    __set_errno (save);
	    return 0;
	  }
      }

  (void) __closedir (dirstream);
  __set_errno (save);
  /* It is not clear what to return in this case.  `isatty' says FD
     refers to a TTY but no entry in /dev has this inode.  */
  return ENOTTY;
}

/* Store at most BUFLEN character of the pathname of the terminal FD is
   open on in BUF.  Return 0 on success,  otherwise an error number.  */
int
__ttyname_r (int fd, char *buf, size_t buflen)
{
  struct fd_to_filename filename;
  struct __stat64_t64 st, st1;
  int dostat = 0;
  int doispty = 0;
  int save = errno;

  /* Test for the absolute minimal size.  This makes life easier inside
     the loop.  */
  if (!buf)
    {
      __set_errno (EINVAL);
      return EINVAL;
    }

  if (buflen < sizeof ("/dev/pts/"))
    {
      __set_errno (ERANGE);
      return ERANGE;
    }

  /* isatty check, tcgetattr is used because it sets the correct
     errno (EBADF resp. ENOTTY) on error.  */
  struct termios term;
  if (__glibc_unlikely (__tcgetattr (fd, &term) < 0))
    return errno;

  if (__fstat64_time64 (fd, &st) < 0)
    return errno;

  /* We try using the /proc filesystem.  */
  ssize_t ret = __readlink (__fd_to_filename (fd, &filename), buf, buflen - 1);
  if (__glibc_unlikely (ret == -1 && errno == ENAMETOOLONG))
    {
      __set_errno (ERANGE);
      return ERANGE;
    }

  if (__glibc_likely (ret != -1))
    {
#define UNREACHABLE_LEN strlen ("(unreachable)")
      if (ret > UNREACHABLE_LEN
	  && memcmp (buf, "(unreachable)", UNREACHABLE_LEN) == 0)
	{
	  memmove (buf, buf + UNREACHABLE_LEN, ret - UNREACHABLE_LEN);
	  ret -= UNREACHABLE_LEN;
	}

      /* readlink need not terminate the string.  */
      buf[ret] = '\0';

      /* Verify readlink result, fall back on iterating through devices.  */
      if (buf[0] == '/'
	  && __stat64_time64 (buf, &st1) == 0
	  && is_mytty (&st, &st1))
	return 0;

      doispty = 1;
    }

  /* Prepare the result buffer.  */
  memcpy (buf, "/dev/pts/", sizeof ("/dev/pts/"));
  buflen -= sizeof ("/dev/pts/") - 1;

  if (__stat64_time64 (buf, &st1) == 0 && S_ISDIR (st1.st_mode))
    {
      ret = getttyname_r (buf, buflen, &st, save,
			  &dostat);
    }
  else
    {
      __set_errno (save);
      ret = ENOENT;
    }

  if (ret && dostat != -1)
    {
      buf[sizeof ("/dev/") - 1] = '\0';
      buflen += sizeof ("pts/") - 1;
      ret = getttyname_r (buf, buflen, &st, save,
			  &dostat);
    }

  if (ret && dostat != -1)
    {
      buf[sizeof ("/dev/") - 1] = '\0';
      dostat = 1;
      ret = getttyname_r (buf, buflen, &st,
			  save, &dostat);
    }

  if (ret && doispty && is_pty (&st))
    {
      /* We failed to figure out the TTY's name, but we can at least
         signal that we did verify that it really is a PTY slave.
         This happens when we have inherited the file descriptor from
         a different mount namespace.  */
      __set_errno (ENODEV);
      return ENODEV;
    }

  return ret;
}
libc_hidden_def (__ttyname_r)
weak_alias (__ttyname_r, ttyname_r)
