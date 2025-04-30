/* Test getauxval from a dynamic library after static dlopen.
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

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <support/check.h>
#include <support/xdlfcn.h>
#include <sys/auxv.h>

unsigned long getauxval_wrapper (unsigned long type, int *errnop);

static int
do_test (void)
{
  unsigned long outer_random = getauxval (AT_RANDOM);
  if (outer_random == 0)
    FAIL_UNSUPPORTED ("getauxval does not support AT_RANDOM");

  unsigned long missing_auxv_type;
  for (missing_auxv_type = AT_RANDOM + 1; ; ++missing_auxv_type)
    {
      errno = 0;
      if (getauxval (missing_auxv_type) == 0 && errno != 0)
        {
          TEST_COMPARE (errno, ENOENT);
          break;
        }
    }
  printf ("info: first missing type: %lu\n", missing_auxv_type);

  void *handle = xdlopen ("tst-auxvalmod.so", RTLD_LAZY);
  void *ptr = xdlsym (handle, "getauxval_wrapper");

  __typeof__ (getauxval_wrapper) *wrapper = ptr;
  int inner_errno = 0;
  unsigned long inner_random = wrapper (AT_RANDOM, &inner_errno);
  TEST_COMPARE (outer_random, inner_random);

  inner_errno = 0;
  TEST_COMPARE (wrapper (missing_auxv_type, &inner_errno), 0);
  TEST_COMPARE (inner_errno, ENOENT);

  TEST_COMPARE (getauxval (AT_HWCAP), wrapper (AT_HWCAP, &inner_errno));
  TEST_COMPARE (getauxval (AT_HWCAP2), wrapper (AT_HWCAP2, &inner_errno));

  xdlclose (handle);
  return 0;
}

#include <support/test-driver.c>
