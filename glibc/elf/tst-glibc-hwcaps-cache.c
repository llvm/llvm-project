/* Wrapper to invoke tst-glibc-hwcaps in a container, to test ld.so.cache.
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

/* This program is just a wrapper that runs ldconfig followed by
   tst-glibc-hwcaps.  The actual test is provided via an
   implementation in a sysdeps subdirectory.  */

#include <stdio.h>
#include <stdlib.h>
#include <support/support.h>
#include <unistd.h>

int
main (int argc, char **argv)
{
  /* Run ldconfig to populate the cache.  */
  {
    char *command = xasprintf ("%s/ldconfig", support_install_rootsbindir);
    if (system (command) != 0)
      return 1;
    free (command);
  }

  /* Reuse tst-glibc-hwcaps.  Since this code is running in a
     container, we can launch it directly.  */
  char *path = xasprintf ("%s/elf/tst-glibc-hwcaps", support_objdir_root);
  execv (path, argv);
  printf ("error: execv of %s failed: %m\n", path);
  return 1;
}
