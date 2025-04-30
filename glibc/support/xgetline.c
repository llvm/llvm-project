/* fopen with error checking.
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

#include <support/xstdio.h>
#include <support/check.h>

size_t
xgetline (char **lineptr, size_t *n, FILE *stream)
{
  TEST_VERIFY (!ferror (stream));
  ssize_t ret = getline (lineptr, n, stream);
  if (ferror (stream))
    {
      TEST_VERIFY (ret < 0);
      FAIL_EXIT1 ("getline: %m");
    }
  if (feof (stream))
    {
      TEST_VERIFY (ret <= 0);
      return 0;
    }
  TEST_VERIFY (ret > 0);
  return ret;
}
