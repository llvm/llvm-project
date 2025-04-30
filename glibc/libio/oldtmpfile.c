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

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)

#define _IO_USE_OLD_IO_FILE
#include <stdio.h>
#include <unistd.h>
#include <iolibio.h>

/* This returns a new stream opened on a temporary file (generated
   by tmpnam).  The file is opened with mode "w+b" (binary read/write).
   If we couldn't generate a unique filename or the file couldn't
   be opened, NULL is returned.  */
FILE *
attribute_compat_text_section
__old_tmpfile (void)
{
  char buf[FILENAME_MAX];
  int fd;
  FILE *f;

  if (__path_search (buf, FILENAME_MAX, NULL, "tmpf", 0))
    return NULL;
  fd = __gen_tempname (buf, 0, 0, __GT_FILE);
  if (fd < 0)
    return NULL;

  /* Note that this relies on the Unix semantics that
     a file is not really removed until it is closed.  */
  (void) __unlink (buf);

  if ((f = _IO_old_fdopen (fd, "w+b")) == NULL)
    __close (fd);

  return f;
}

compat_symbol (libc, __old_tmpfile, tmpfile, GLIBC_2_0);
#endif
