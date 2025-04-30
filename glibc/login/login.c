/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <utmp.h>
#include <shlib-compat.h>

/* Return the result of ttyname in the buffer pointed to by TTY, which should
   be of length BUF_LEN.  If it is too long to fit in this buffer, a
   sufficiently long buffer is allocated using malloc, and returned in TTY.
   0 is returned upon success, -1 otherwise.  */
static int
tty_name (int fd, char **tty, size_t buf_len)
{
  int rv;			/* Return value.  0 = success.  */
  char *buf = *tty;		/* Buffer for ttyname, initially the user's. */

  for (;;)
    {
      char *new_buf;

      if (buf_len)
	{
	  rv = __ttyname_r (fd, buf, buf_len);

	  if (rv != 0 || memchr (buf, '\0', buf_len))
	    /* We either got an error, or we succeeded and the
	       returned name fit in the buffer.  */
	    break;

	  /* Try again with a longer buffer.  */
	  buf_len += buf_len;	/* Double it */
	}
      else
	/* No initial buffer; start out by mallocing one.  */
	buf_len = 128;		/* First time guess.  */

      if (buf != *tty)
	/* We've already malloced another buffer at least once.  */
	new_buf = realloc (buf, buf_len);
      else
	new_buf = malloc (buf_len);
      if (! new_buf)
	{
	  rv = -1;
	  __set_errno (ENOMEM);
	  break;
	}
      buf = new_buf;
    }

  if (rv == 0)
    *tty = buf;		/* Return buffer to the user.  */
  else if (buf != *tty)
    free (buf);		/* Free what we malloced when returning an error.  */

  return rv;
}

void
__login (const struct utmp *ut)
{
#ifdef PATH_MAX
  char _tty[PATH_MAX + UT_LINESIZE];
#else
  char _tty[512 + UT_LINESIZE];
#endif
  char *tty = _tty;
  int found_tty;
  const char *ttyp;
  struct utmp copy = *ut;

  /* Fill in those fields we supply.  */
  copy.ut_type = USER_PROCESS;
  copy.ut_pid = getpid ();

  /* Seek tty.  */
  found_tty = tty_name (STDIN_FILENO, &tty, sizeof (_tty));
  if (found_tty < 0)
    found_tty = tty_name (STDOUT_FILENO, &tty, sizeof (_tty));
  if (found_tty < 0)
    found_tty = tty_name (STDERR_FILENO, &tty, sizeof (_tty));

  if (found_tty >= 0)
    {
      /* We only want to insert the name of the tty without path.
	 But take care of name like /dev/pts/3.  */
      if (strncmp (tty, "/dev/", 5) == 0)
	ttyp = tty + 5;		/* Skip the "/dev/".  */
      else
	ttyp = basename (tty);

      /* Position to record for this tty.  */
      strncpy (copy.ut_line, ttyp, UT_LINESIZE);

      /* Tell that we want to use the UTMP file.  */
      if (__utmpname (_PATH_UTMP) == 0)
	{
	  /* Open UTMP file.  */
	  __setutent ();

	  /* Write the entry.  */
	  __pututline (&copy);

	  /* Close UTMP file.  */
	  __endutent ();
	}

      if (tty != _tty)
	free (tty);		/* Free buffer malloced by tty_name.  */
    }
  else
    /* We provide a default value so that the output does not contain
       an random bytes in this field.  */
    strncpy (copy.ut_line, "???", UT_LINESIZE);

  /* Update the WTMP file.  Here we have to add a new entry.  */
  __updwtmp (_PATH_WTMP, &copy);
}
versioned_symbol (libc, __login, login, GLIBC_2_34);
libc_hidden_ver (__login, login)

#if OTHER_SHLIB_COMPAT (libutil, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libutil, __login, login, GLIBC_2_0);
#endif
