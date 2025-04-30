/* Test sysconf with an empty chroot.
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

#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xunistd.h>
#include <unistd.h>

/* Check for an SMP system in a forked process, so that the parent
   process does not cache the value.  */
static void
is_smp_callback (void *closure)
{
  bool *result = closure;

  long cpus = sysconf (_SC_NPROCESSORS_ONLN);
  TEST_VERIFY_EXIT (cpus > 0);
  *result = cpus != 1;
}

static bool
is_smp (void)
{
  bool *result = support_shared_allocate (sizeof (*result));
  support_isolate_in_subprocess (is_smp_callback, result);
  bool result_copy = *result;
  support_shared_free (result);
  return result_copy;
}

static char *path_chroot;

/* Prepare an empty directory, to be used as a chroot.  */
static void
prepare (int argc, char **argv)
{
  path_chroot = xasprintf ("%s/tst-resolv-res_init-XXXXXX", test_dir);
  if (mkdtemp (path_chroot) == NULL)
    FAIL_EXIT1 ("mkdtemp (\"%s\"): %m", path_chroot);
  add_temp_file (path_chroot);
}

/* The actual test.  Run it in a subprocess, so that the test harness
   can remove the temporary directory in --direct mode.  */
static void
chroot_callback (void *closure)
{
  xchroot (path_chroot);
  long cpus = sysconf (_SC_NPROCESSORS_ONLN);
  printf ("info: sysconf (_SC_NPROCESSORS_ONLN) in chroot: %ld\n", cpus);
  TEST_VERIFY (cpus > 0);
  TEST_VERIFY (cpus != 1);
  _exit (0);
}

static int
do_test (void)
{
  if (!is_smp ())
    {
      printf ("warning: test not supported on uniprocessor system\n");
      return EXIT_UNSUPPORTED;
    }

  support_become_root ();
  if (!support_can_chroot ())
    return EXIT_UNSUPPORTED;

  support_isolate_in_subprocess (chroot_callback, NULL);

  return 0;
}

#define PREPARE prepare
#include <support/test-driver.c>
