/* Write a string to a file.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <support/check.h>
#include <support/xunistd.h>

void
support_write_file_string (const char *path, const char *contents)
{
  int fd = xopen (path, O_CREAT | O_TRUNC | O_WRONLY, 0666);
  const char *end = contents + strlen (contents);
  for (const char *p = contents; p < end; )
    {
      ssize_t ret = write (fd, p, end - p);
      if (ret < 0)
        FAIL_EXIT1 ("cannot write to \"%s\": %m", path);
      if (ret == 0)
        FAIL_EXIT1 ("zero-length write to \"%s\"", path);
      p += ret;
    }
  xclose (fd);
}
