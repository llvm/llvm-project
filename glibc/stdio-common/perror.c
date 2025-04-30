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
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>
#include "libioP.h"

static void
perror_internal (FILE *fp, const char *s, int errnum)
{
  char buf[1024];
  const char *colon;
  const char *errstring;

  if (s == NULL || *s == '\0')
    s = colon = "";
  else
    colon = ": ";

  errstring = __strerror_r (errnum, buf, sizeof buf);

  (void) __fxprintf (fp, "%s%s%s\n", s, colon, errstring);
}


/* Print a line on stderr consisting of the text in S, a colon, a space,
   a message describing the meaning of the contents of `errno' and a newline.
   If S is NULL or "", the colon and space are omitted.  */
void
perror (const char *s)
{
  int errnum = errno;
  FILE *fp;
  int fd = -1;


  /* The standard says that 'perror' must not change the orientation
     of the stream.  What is supposed to happen when the stream isn't
     oriented yet?  In this case we'll create a new stream which is
     using the same underlying file descriptor.  */
  if (__builtin_expect (_IO_fwide (stderr, 0) != 0, 1)
      || (fd = __fileno (stderr)) == -1
      || (fd = __dup (fd)) == -1
      || (fp = fdopen (fd, "w+")) == NULL)
    {
      if (__glibc_unlikely (fd != -1))
	__close (fd);

      /* Use standard error as is.  */
      perror_internal (stderr, s, errnum);
    }
  else
    {
      /* We don't have to do any special hacks regarding the file
	 position.  Since the stderr stream wasn't used so far we just
	 write to the descriptor.  */
      perror_internal (fp, s, errnum);

      if (_IO_ferror_unlocked (fp))
	stderr->_flags |= _IO_ERR_SEEN;

      /* Close the stream.  */
      fclose (fp);
    }
}
libc_hidden_def (perror)
