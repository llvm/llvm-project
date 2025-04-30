/* Wrapper to invoke tst-glibc-hwcaps-2 in a container to test ldconfig.
   Copyright (C) 2021 Free Software Foundation, Inc.
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
   tst-glibc-hwcaps-2.  The actual test is provided via an
   implementation in a sysdeps subdirectory.  */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <support/support.h>
#include <support/capture_subprocess.h>

int
main (int argc, char **argv)
{
  /* Run ldconfig to populate the cache.  */
  char *command = xasprintf ("%s/ldconfig", support_install_rootsbindir);
  struct support_capture_subprocess result =
    support_capture_subprogram (command,  &((char *) { NULL }));
  support_capture_subprocess_check (&result, "ldconfig", 0, sc_allow_none);
  free (command);

  /* Reuse tst-glibc-hwcaps.  Since this code is running in a
     container, we can launch it directly.  */
  char *path = xasprintf ("%s/elf/tst-glibc-hwcaps-2", support_objdir_root);
  execv (path, argv);
  printf ("error: execv of %s failed: %m\n", path);
  return 1;
}
