/* Test fegetenv preserves exception mask (bug 16198).
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <fenv.h>
#include <stdio.h>

static int
do_test (void)
{
#if FE_ALL_EXCEPT
  fenv_t env;

  if (feenableexcept (FE_INVALID) != 0)
    {
      puts ("feenableexcept (FE_INVALID) failed, cannot test");
      return 0;
    }

  if (fegetenv (&env) != 0)
    {
      puts ("fegetenv failed, cannot test");
      return 0;
    }

  int ret = fegetexcept ();
  if (ret == FE_INVALID)
    {
      puts ("fegetenv preserved exception mask, OK");
      return 0;
    }
  else
    {
      printf ("fegetexcept returned %d, expected %d\n", ret, FE_INVALID);
      return 1;
    }
#else
  puts ("No exceptions defined, cannot test");
  return 0;
#endif
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
