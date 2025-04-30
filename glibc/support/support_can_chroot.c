/* Return true if the process can perform a chroot operation.
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

#include <errno.h>
#include <stdio.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/xunistd.h>
#include <sys/stat.h>
#include <unistd.h>

static void
callback (void *closure)
{
  int *result = closure;
  struct stat64 before;
  xstat ("/dev", &before);
  if (chroot ("/dev") != 0)
    {
      *result = errno;
      return;
    }
  struct stat64 after;
  xstat ("/", &after);
  TEST_VERIFY (before.st_dev == after.st_dev);
  TEST_VERIFY (before.st_ino == after.st_ino);
  *result = 0;
}

bool
support_can_chroot (void)
{
  int *result = support_shared_allocate (sizeof (*result));
  *result = 0;
  support_isolate_in_subprocess (callback, result);
  bool ok = *result == 0;
  if (!ok)
    {
      static bool already_warned;
      if (!already_warned)
        {
          already_warned = true;
          errno = *result;
          printf ("warning: this process does not support chroot: %m\n");
        }
    }
  support_shared_free (result);
  return ok;
}
