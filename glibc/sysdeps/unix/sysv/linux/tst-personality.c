/* BZ #19408 linux personality syscall wrapper test.

   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <sys/personality.h>

static int
do_test (void)
{
  int rc = 0;
  unsigned int test_persona = -EINVAL;
  unsigned int saved_persona;

  errno = 0xdefaced;
  saved_persona = personality (0xffffffff);

  if (personality (test_persona) != saved_persona
      || personality (0xffffffff) == -1
      || personality (PER_LINUX) == -1
      || personality (0xffffffff) != PER_LINUX
      || 0xdefaced != errno)
    rc = 1;

  (void) personality (saved_persona);
  return rc;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
