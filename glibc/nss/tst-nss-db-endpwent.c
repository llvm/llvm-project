/* Test for endpwent->getpwent crash for BZ #24695
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <pwd.h>

#include <support/support.h>
#include <support/check.h>

/* It is entirely allowed to start with a getpwent call without
   resetting the state of the service via a call to setpwent.
   You can also call getpwent more times than you have entries in
   the service, and it should not fail.  This test iteratates the
   database once, gets to the end, and then attempts a second
   iteration to look for crashes.  */

static void
try_it (void)
{
  struct passwd *pw;

  /* setpwent is intentionally omitted here.  The first call to
     getpwent detects that it's first and initializes.  The second
     time try_it is called, this "first call" was not detected before
     the fix, and getpwent would crash.  */

  while ((pw = getpwent ()) != NULL)
    ;

  /* We only care if this segfaults or not.  */
  endpwent ();
}

static int
do_test (void)
{
  char *cmd;

  cmd = xasprintf ("%s/makedb -o /var/db/passwd.db /var/db/passwd.in",
		   support_bindir_prefix);
  system (cmd);
  free (cmd);

  try_it ();
  try_it ();

  return 0;
}
#include <support/test-driver.c>
