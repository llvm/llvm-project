/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>


extern int foo (void);

static const char testsubdir[] = PFX "test-subdir";


static int
do_test (void)
{
  struct stat64 st;
  int result = 1;

  if (mkdir (testsubdir, 0777) != 0
      && (errno != EEXIST
	  || stat64 (testsubdir, &st) != 0
	  || !S_ISDIR (st.st_mode)))
    {
      printf ("cannot create directory %s\n", testsubdir);
      return 1;
    }

  if (system ("cp " PFX "firstobj.so " PFX "test-subdir/in-subdir.so") != 0)
    {
      puts ("cannot copy DSO");
      return 1;
    }

  void *p = dlopen ("in-subdir.so", RTLD_LAZY|RTLD_LOCAL);
  if (p != NULL)
    {
      puts ("succeeded in opening in-subdir.so from do_test");
      dlclose (p);
      goto out;
    }

  result = foo ();

 out:
  unlink (PFX "test-subdir/in-subdir.so");
  rmdir (testsubdir);

  return result;
}

#include <support/test-driver.c>
