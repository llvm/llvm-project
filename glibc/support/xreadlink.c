/* Error-checking, allocating wrapper for readlink.
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

#include <scratch_buffer.h>
#include <support/check.h>
#include <support/support.h>
#include <xunistd.h>

char *
xreadlink (const char *path)
{
  struct scratch_buffer buf;
  scratch_buffer_init (&buf);

  while (true)
    {
      ssize_t count = readlink (path, buf.data, buf.length);
      if (count < 0)
        FAIL_EXIT1 ("readlink (\"%s\"): %m", path);
      if (count < buf.length)
        {
          char *result = xstrndup (buf.data, count);
          scratch_buffer_free (&buf);
          return result;
        }
      if (!scratch_buffer_grow (&buf))
        FAIL_EXIT1 ("scratch_buffer_grow in xreadlink");
    }
}
