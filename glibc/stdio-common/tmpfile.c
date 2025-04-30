/* Open a stdio stream on an anonymous temporary file.  Generic/POSIX version.
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

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <iolibio.h>
#define __fdopen _IO_fdopen
#ifndef tmpfile
# define tmpfile __new_tmpfile
#endif


/* This returns a new stream opened on a temporary file (generated
   by tmpnam).  The file is opened with mode "w+b" (binary read/write).
   If we couldn't generate a unique filename or the file couldn't
   be opened, NULL is returned.  */
FILE *
tmpfile (void)
{
  int fd;
  FILE *f;
  int flags = 0;
#ifdef FLAGS
  flags = FLAGS;
#endif

  /* First try a system specific method.  */
  fd = __gen_tempfd (flags);

  if (fd < 0)
    {
      char buf[FILENAME_MAX];

      if (__path_search (buf, sizeof buf, NULL, "tmpf", 0))
	return NULL;

      fd = __gen_tempname (buf, 0, flags, __GT_FILE);
      if (fd < 0)
	return NULL;

      /* Note that this relies on the Unix semantics that
	 a file is not really removed until it is closed.  */
      (void) __unlink (buf);
    }

  if ((f = __fdopen (fd, "w+b")) == NULL)
    __close (fd);

  return f;
}

#if !defined O_LARGEFILE || O_LARGEFILE == 0
weak_alias (__new_tmpfile, tmpfile64)
#endif

#ifndef FLAGS /* Not for tmpfile64.  */
# undef tmpfile
# include <shlib-compat.h>
versioned_symbol (libc, __new_tmpfile, tmpfile, GLIBC_2_1);
#endif
