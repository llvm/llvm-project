/* Test relocation ordering if the main executable is libc.so.6 (bug 20972).
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

#include <gnu/lib-names.h>
#include <stdio.h>
#include <support/support.h>
#include <unistd.h>

int
main (void)
{
  char *libc = xasprintf ("%s/%s", support_slibdir_prefix, LIBC_SO);
  char *argv[] = { libc, NULL };
  char *envp[] = { (char *) "LD_PRELOAD=" LIBPTHREAD_SO,
    /* Relocation ordering matters most without lazy binding.  */
    (char *) "LD_BIND_NOW=1",
    NULL };
  execve (libc, argv, envp);
  printf ("execve of %s failed: %m\n", libc);
  return 1;
}
