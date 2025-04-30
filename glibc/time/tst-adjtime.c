/* Basic tests for adjtime.
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

#include <sys/time.h>
#include <stdlib.h>

#include <errno.h>
#include <support/check.h>


static int
do_test (void)
{
  /* Check if the interface allows getting the amount of time remaining
     from any previous adjustment that has not yet been completed.  This
     is a non-privileged function of adjtime.  */
  struct timeval tv;
  int r = adjtime (NULL, &tv);
  if (r == -1)
    {
      if (errno == ENOSYS)
	FAIL_UNSUPPORTED ("adjtime unsupported");
      FAIL_EXIT1 ("adjtime (NULL, ...) failed: %m");
    }

  return 0;
}

#include <support/test-driver.c>
