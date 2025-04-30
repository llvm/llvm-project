/* Test ldconfig after /etc/ld.so.conf update and verify that a running process
   observes changes to /etc/ld.so.cache.

   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xdlfcn.h>
#include <support/xstdio.h>
#include <support/xunistd.h>


#define DSO "libldconfig-ld-mod.so"
#define DSO_DIR "/tmp/tst-ldconfig"
#define CONF "/etc/ld.so.conf"


static void
run_ldconfig (void *x __attribute__((unused)))
{
  char *prog = xasprintf ("%s/ldconfig", support_install_rootsbindir);
  char *args[] = { prog, NULL };

  execv (args[0], args);
  FAIL_EXIT1 ("execv: %m");
}


/* Create a new directory.
   Copy a test shared object there.
   Try to dlopen it by soname.  This should fail.
   (Directory is not searched.)
   Run ldconfig.
   Try to dlopen it again.  It should still fail.
   (Directory is still not searched.)
   Add the directory to /etc/ld.so.conf.
   Try to dlopen it again.  It should still fail.
   (The loader does not read /etc/ld.so.conf, only /etc/ld.so.cache.)
   Run ldconfig.
   Try to dlopen it again.  This should finally succeed.  */
static int
do_test (void)
{
  struct support_capture_subprocess result;

  /* Create the needed directories.  */
  xmkdirp ("/var/cache/ldconfig", 0777);
  xmkdirp (DSO_DIR, 0777);

  /* Rename the DSO to start with "lib" because there's an undocumented
     filter in ldconfig where it ignores any file that doesn't start with
     "lib" (for regular shared libraries) or "ld-" (for ld-linux-*).  */
  char *mod_src_path = xasprintf ("%s/tst-ldconfig-ld-mod.so",
				  support_libdir_prefix);
  if (rename (mod_src_path, "/tmp/tst-ldconfig/libldconfig-ld-mod.so"))
    FAIL_EXIT1 ("Renaming/moving the DSO failed: %m");
  free (mod_src_path);


  /* Open the DSO.  We expect this to fail - tst-ldconfig directory
     is not searched.  */
  TEST_VERIFY_EXIT (dlopen (DSO, RTLD_NOW | RTLD_GLOBAL) == NULL);

  FILE *fp = xfopen (CONF, "a+");
  if (!fp)
    FAIL_EXIT1 ("creating /etc/ld.so.conf failed: %m");
  xfclose (fp);

  /* Run ldconfig.  */
  result = support_capture_subprocess (run_ldconfig, NULL);
  support_capture_subprocess_check (&result, "execv", 0, sc_allow_none);

  /* Try to dlopen the same DSO again, we expect this to fail again.  */
  TEST_VERIFY_EXIT (dlopen (DSO, RTLD_NOW | RTLD_GLOBAL) == NULL);

  /* Add tst-ldconfig directory to /etc/ld.so.conf.  */
  fp = xfopen (CONF, "w");
  if (!(fwrite (DSO_DIR, 1, sizeof (DSO_DIR), fp)))
    FAIL_EXIT1 ("updating /etc/ld.so.conf failed: %m");
  xfclose (fp);

  /* Try to dlopen the same DSO again, we expect this to still fail.  */
  TEST_VERIFY_EXIT (dlopen (DSO, RTLD_NOW | RTLD_GLOBAL) == NULL);

  /* Run ldconfig again.  */
  result = support_capture_subprocess (run_ldconfig, NULL);
  support_capture_subprocess_check (&result, "execv", 0, sc_allow_none);
  support_capture_subprocess_free (&result);

  /* Finally, we expect dlopen to pass now.  */
  TEST_VERIFY_EXIT (dlopen (DSO, RTLD_NOW | RTLD_GLOBAL) != NULL);

  return 0;
}

#include <support/test-driver.c>
