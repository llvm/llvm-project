/* Open an nss_files database file.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <nss_files.h>

#include <errno.h>
#include <stdio_ext.h>

FILE *
__nss_files_fopen (const char *path)
{
  FILE *fp = fopen (path, "rce");
  if (fp == NULL)
    return NULL;

  /* The stream is not shared across threads.  */
  __fsetlocking (fp, FSETLOCKING_BYCALLER);

  /* This tells libio that the file is seekable, and that fp->_offset
     is correct, ensuring that __ftello64 is efficient (bug 26257).  */
  if (__fseeko64 (fp, 0, SEEK_SET) < 0)
    {
      /* nss_files requires seekable files, to deal with repeated
         reads of the same line after reporting ERANGE.  */
      fclose (fp);
      __set_errno (ESPIPE);
      return NULL;
    }

  return fp;
}
libc_hidden_def (__nss_files_fopen)
