/* Test error reporting for dlsym, dlvsym failures.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
#include <gnu/lib-names.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Used to disambiguate symbol names.  */
static int counter;

static void
test_one (void *handle, const char *name, void *(func) (void *, const char *),
          const char *suffix)
{
  ++counter;
  char symbol[32];
  snprintf (symbol, sizeof (symbol), "no_such_symbol_%d", counter);
  char *expected_message;
  if (asprintf (&expected_message, ": undefined symbol: %s%s",
                symbol, suffix) < 0)
    {
      printf ("error: asprintf: %m\n");
      abort ();
    }

  void *addr = func (handle, symbol);
  if (addr != NULL)
    {
      printf ("error: %s: found symbol \"no_such_symbol\"\n", name);
      abort ();
    }
  const char *message = dlerror ();
  if (message == NULL)
    {
      printf ("error: %s: missing error message\n", name);
      abort ();
    }
  const char *message_without_path = strchrnul (message, ':');
  if (strcmp (message_without_path, expected_message) != 0)
    {
      printf ("error: %s: unexpected error message: %s\n", name, message);
      abort ();
    }
  free (expected_message);

  message = dlerror ();
  if (message != NULL)
    {
      printf ("error: %s: unexpected error message: %s\n", name, message);
      abort ();
    }
}

static void
test_handles (const char *name, void *(func) (void *, const char *),
              const char *suffix)
{
  test_one (RTLD_DEFAULT, name, func, suffix);
  test_one (RTLD_NEXT, name, func, suffix);

  void *handle = dlopen (LIBC_SO, RTLD_LAZY);
  if (handle == NULL)
    {
      printf ("error: cannot dlopen %s: %s\n", LIBC_SO, dlerror ());
      abort ();
    }
  test_one (handle, name, func, suffix);
  dlclose (handle);
}

static void *
dlvsym_no_such_version (void *handle, const char *name)
{
  return dlvsym (handle, name, "NO_SUCH_VERSION");
}

static void *
dlvsym_glibc_private (void *handle, const char *name)
{
  return dlvsym (handle, name, "GLIBC_PRIVATE");
}

static int
do_test (void)
{
  test_handles ("dlsym", dlsym, "");
  test_handles ("dlvsym", dlvsym_no_such_version,
                ", version NO_SUCH_VERSION");
  test_handles ("dlvsym", dlvsym_glibc_private,
                ", version GLIBC_PRIVATE");

  return 0;
}


#include <support/test-driver.c>
