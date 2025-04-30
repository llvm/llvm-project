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
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <fcntl.h>

#include <utmp.h>

static char name[UT_NAMESIZE + 1];

/* Return the login name of the user, or NULL if it can't be determined.
   The returned pointer, if not NULL, is good only until the next call.  */

#ifdef STATIC
STATIC
#endif
char *
getlogin (void)
{
  char tty_pathname[2 + 2 * NAME_MAX];
  char *real_tty_path = tty_pathname;
  int err;
  char *result = NULL;
  struct utmp *ut, line, buffer;

  /* Get name of tty connected to fd 0.  Return NULL if not a tty or
     if fd 0 isn't open.  Note that a lot of documentation says that
     getlogin() is based on the controlling terminal---what they
     really mean is "the terminal connected to standard input".  The
     getlogin() implementation of DEC Unix, SunOS, Solaris, HP-UX all
     return NULL if fd 0 has been closed, so this is the compatible
     thing to do.  Note that ttyname(open("/dev/tty")) on those
     systems returns /dev/tty, so that is not a possible solution for
     getlogin().  */
  err = __ttyname_r (0, real_tty_path, sizeof (tty_pathname));
  if (err != 0)
    {
      __set_errno (err);
      return NULL;
    }

  real_tty_path += 5;		/* Remove "/dev/".  */

  __setutent ();
  strncpy (line.ut_line, real_tty_path, sizeof line.ut_line);
  if (__getutline_r (&line, &buffer, &ut) < 0)
    {
      if (errno == ESRCH)
	/* The caller expects ENOENT if nothing is found.  */
	__set_errno (ENOENT);
      result = NULL;
    }
  else
    {
      strncpy (name, ut->ut_user, UT_NAMESIZE);
      name[UT_NAMESIZE] = '\0';
      result = name;
    }

  __endutent ();

  return result;
}
