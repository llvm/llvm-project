/* dlopen test for PIE objects.
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

/* This test attempts to open the (otherwise unrelated) PIE test
   program elf/tst-pie1 and expects the attempt to fail.  */

#include <dlfcn.h>
#include <stddef.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>

static void
test_mode (int mode)
{
  char *pie_path = xasprintf ("%s/elf/tst-pie1", support_objdir_root);
  if (dlopen (pie_path, mode) != NULL)
    FAIL_EXIT1 ("dlopen succeeded unexpectedly (%d)", mode);
  const char *message = dlerror ();
  const char *expected
    = "cannot dynamically load position-independent executable";
  if (strstr (message, expected) == NULL)
    FAIL_EXIT1 ("unexpected error message (mode %d): %s", mode, message);
}

static int
do_test (void)
{
  test_mode (RTLD_LAZY);
  test_mode (RTLD_NOW);
  return 0;
}

#include <support/test-driver.c>
