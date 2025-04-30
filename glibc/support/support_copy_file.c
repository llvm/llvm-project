/* Copy a file from one path to another.
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

#include <fcntl.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xunistd.h>

void
support_copy_file (const char *from, const char *to)
{
  struct stat64 st;
  xstat (from, &st);
  int fd_from = xopen (from, O_RDONLY, 0);
  mode_t mode = st.st_mode & 0777;
  int fd_to = xopen (to, O_WRONLY | O_TRUNC | O_CREAT, mode);
  ssize_t ret = support_copy_file_range (fd_from, NULL, fd_to, NULL,
                                         st.st_size, 0);
  if (ret < 0)
    FAIL_EXIT1 ("copying from \"%s\" to \"%s\": %m", from, to);
  if (ret != st.st_size)
    FAIL_EXIT1 ("copying from \"%s\" to \"%s\": only %zd of %llu bytes copied",
                from, to, ret, (unsigned long long int) st.st_size);
  if (fchmod (fd_to, mode) < 0)
    FAIL_EXIT1 ("fchmod on %s to 0%o: %m", to, mode);
  xclose (fd_to);
  xclose (fd_from);
}
