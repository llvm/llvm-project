/* Check that dlopen'ing the executable itself fails (bug 24900).
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>

/* Call dlopen and check that fails with an error message indicating
   an attempt to open an ET_EXEC or PIE object.  */
static void
check_dlopen_failure (const char *path)
{
  void *handle = dlopen (path, RTLD_LAZY);
  if (handle != NULL)
    FAIL_EXIT1 ("dlopen succeeded unexpectedly: %s", path);

  const char *message = dlerror ();
  TEST_VERIFY_EXIT (message != NULL);
  if ((strstr (message,
	       "cannot dynamically load position-independent executable")
       == NULL)
      && strstr (message, "cannot dynamically load executable") == NULL)
    FAIL_EXIT1 ("invalid dlopen error message: \"%s\"", message);
}

static int
do_test (int argc, char *argv[])
{
  check_dlopen_failure (argv[0]);

  char *full_path = realpath (argv[0], NULL);
  check_dlopen_failure  (full_path);
  free (full_path);

  return 0;
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
