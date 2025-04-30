/* Open a pseudoterminal.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <support/tty.h>
#include <support/check.h>
#include <support/support.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* As ptsname, but allocates space for an appropriately-sized string
   using malloc.  */
static char *
xptsname (int fd)
{
  int rv;
  size_t buf_len = 128;
  char *buf = xmalloc (buf_len);
  for (;;)
    {
      rv = ptsname_r (fd, buf, buf_len);
      if (rv)
        FAIL_EXIT1 ("ptsname_r: %s", strerror (errno));

      if (memchr (buf, '\0', buf_len))
        return buf; /* ptsname succeeded and the buffer was not truncated */

      buf_len *= 2;
      buf = xrealloc (buf, buf_len);
    }
}

void
support_openpty (int *a_outer, int *a_inner, char **a_name,
                 const struct termios *termp,
                 const struct winsize *winp)
{
  int outer = -1, inner = -1;
  char *namebuf = 0;

  outer = posix_openpt (O_RDWR | O_NOCTTY);
  if (outer == -1)
    FAIL_EXIT1 ("posix_openpt: %s", strerror (errno));

  if (grantpt (outer))
    FAIL_EXIT1 ("grantpt: %s", strerror (errno));

  if (unlockpt (outer))
    FAIL_EXIT1 ("unlockpt: %s", strerror (errno));


#ifdef TIOCGPTPEER
  inner = ioctl (outer, TIOCGPTPEER, O_RDWR | O_NOCTTY);
#endif
  if (inner == -1)
    {
      /* The kernel might not support TIOCGPTPEER, fall back to open
         by name.  */
      namebuf = xptsname (outer);
      inner = open (namebuf, O_RDWR | O_NOCTTY);
      if (inner == -1)
        FAIL_EXIT1 ("%s: %s", namebuf, strerror (errno));
    }

  if (termp)
    {
      if (tcsetattr (inner, TCSAFLUSH, termp))
        FAIL_EXIT1 ("tcsetattr: %s", strerror (errno));
    }
#ifdef TIOCSWINSZ
  if (winp)
    {
      if (ioctl (inner, TIOCSWINSZ, winp))
        FAIL_EXIT1 ("TIOCSWINSZ: %s", strerror (errno));
    }
#endif

  if (a_name)
    {
      if (!namebuf)
        namebuf = xptsname (outer);
      *a_name = namebuf;
    }
  else
    free (namebuf);
  *a_outer = outer;
  *a_inner = inner;
}
