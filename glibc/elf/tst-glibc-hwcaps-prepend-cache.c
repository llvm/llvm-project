/* Test that --glibc-hwcaps-prepend works, using dlopen and /etc/ld.so.cache.
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

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xdlfcn.h>
#include <support/xunistd.h>

/* Invoke /sbin/ldconfig with some error checking.  */
static void
run_ldconfig (void)
{
  char *command = xasprintf ("%s/ldconfig", support_install_rootsbindir);
  TEST_COMPARE (system (command), 0);
  free (command);
}

/* The library under test.  */
#define SONAME "libmarkermod1.so"

static int
do_test (void)
{
  if (dlopen (SONAME, RTLD_NOW) != NULL)
    FAIL_EXIT1 (SONAME " is already on the search path");

  /* Install the default implementation of libmarkermod1.so.  */
  xmkdirp ("/etc", 0777);
  support_write_file_string ("/etc/ld.so.conf", "/glibc-test/lib\n");
  xmkdirp ("/glibc-test/lib/glibc-hwcaps/prepend2", 0777);
  xmkdirp ("/glibc-test/lib/glibc-hwcaps/prepend3", 0777);
  {
    char *src = xasprintf ("%s/elf/libmarkermod1-1.so", support_objdir_root);
    support_copy_file (src, "/glibc-test/lib/" SONAME);
    free (src);
  }
  run_ldconfig ();
  {
    /* The default implementation can now be loaded.  */
    void *handle = xdlopen (SONAME, RTLD_NOW);
    int (*marker1) (void) = xdlsym (handle, "marker1");
    TEST_COMPARE (marker1 (), 1);
    xdlclose (handle);
  }

  /* Add the first override to the directory that is searched last.  */
  {
    char *src = xasprintf ("%s/elf/libmarkermod1-2.so", support_objdir_root);
    support_copy_file (src, "/glibc-test/lib/glibc-hwcaps/prepend2/"
                       SONAME);
    free (src);
  }
  {
    /* This is still the first implementation.  The cache has not been
       updated.  */
    void *handle = xdlopen (SONAME, RTLD_NOW);
    int (*marker1) (void) = xdlsym (handle, "marker1");
    TEST_COMPARE (marker1 (), 1);
    xdlclose (handle);
  }
  run_ldconfig ();
  {
    /* After running ldconfig, it is the second implementation.  */
    void *handle = xdlopen (SONAME, RTLD_NOW);
    int (*marker1) (void) = xdlsym (handle, "marker1");
    TEST_COMPARE (marker1 (), 2);
    xdlclose (handle);
  }

  /* Add the second override to the directory that is searched first.  */
  {
    char *src = xasprintf ("%s/elf/libmarkermod1-3.so", support_objdir_root);
    support_copy_file (src, "/glibc-test/lib/glibc-hwcaps/prepend3/"
                       SONAME);
    free (src);
  }
  {
    /* This is still the second implementation.  */
    void *handle = xdlopen (SONAME, RTLD_NOW);
    int (*marker1) (void) = xdlsym (handle, "marker1");
    TEST_COMPARE (marker1 (), 2);
    xdlclose (handle);
  }
  run_ldconfig ();
  {
    /* After running ldconfig, it is the third implementation.  */
    void *handle = xdlopen (SONAME, RTLD_NOW);
    int (*marker1) (void) = xdlsym (handle, "marker1");
    TEST_COMPARE (marker1 (), 3);
    xdlclose (handle);
  }

  /* Remove the second override again, without running ldconfig.
     Ideally, this would revert to implementation 2.  However, in the
     current implementation, the cache returns exactly one file name
     which does not exist after unlinking, so the dlopen fails.  */
  xunlink ("/glibc-test/lib/glibc-hwcaps/prepend3/" SONAME);
  TEST_VERIFY (dlopen (SONAME, RTLD_NOW) == NULL);
  run_ldconfig ();
  {
    /* After running ldconfig, the second implementation is available
       once more.  */
    void *handle = xdlopen (SONAME, RTLD_NOW);
    int (*marker1) (void) = xdlsym (handle, "marker1");
    TEST_COMPARE (marker1 (), 2);
    xdlclose (handle);
  }

  return 0;
}

static void
prepare (int argc, char **argv)
{
  const char *no_restart = "no-restart";
  if (argc == 2 && strcmp (argv[1], no_restart) == 0)
    return;
  /* Re-execute the test with an explicit loader invocation.  */
  execl (support_objdir_elf_ldso,
         support_objdir_elf_ldso,
         "--glibc-hwcaps-prepend", "prepend3:prepend2",
         argv[0], no_restart,
         NULL);
  printf ("error: execv of %s failed: %m\n", argv[0]);
  _exit (1);
}

#define PREPARE prepare
#include <support/test-driver.c>
