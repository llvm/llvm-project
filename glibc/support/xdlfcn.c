/* Support functionality for using dlopen/dlclose/dlsym.
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

#include <stddef.h>
#include <support/check.h>
#include <support/xdlfcn.h>

void *
xdlopen (const char *filename, int flags)
{
  void *dso = dlopen (filename, flags);

  if (dso == NULL)
    FAIL_EXIT1 ("error: dlopen: %s\n", dlerror ());

  return dso;
}

void *
xdlsym (void *handle, const char *symbol)
{
  /* Clear any pending errors.  */
  dlerror ();

  void *sym = dlsym (handle, symbol);

  if (sym == NULL)
    {
      const char *error = dlerror ();
      if (error != NULL)
        FAIL_EXIT1 ("error: dlsym: %s\n", error);
      /* If there was no error, we found a NULL symbol.  Return the
         NULL value in this case.  */
    }

  return sym;
}

void *
xdlvsym (void *handle, const char *symbol, const char *version)
{
  /* Clear any pending errors.  */
  dlerror ();

  void *sym = dlvsym (handle, symbol, version);

  if (sym == NULL)
    {
      const char *error = dlerror ();
      if (error != NULL)
        FAIL_EXIT1 ("error: dlvsym: %s\n", error);
      /* If there was no error, we found a NULL symbol.  Return the
         NULL value in this case.  */
    }

  return sym;
}

void
xdlclose (void *handle)
{
  if (dlclose (handle) != 0)
    FAIL_EXIT1 ("error: dlclose: %s\n", dlerror ());
}
