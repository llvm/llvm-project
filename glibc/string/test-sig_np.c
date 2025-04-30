/* Test and sigabbrev_np and sigdescr_np.
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

#include <string.h>
#include <signal.h>
#include <array_length.h>

#include <support/support.h>
#include <support/check.h>

static const struct test_t
{
  int errno;
  const char *abbrev;
  const char *descr;
} tests[] =
{
#define N_(name)                      name
#define init_sig(sig, abbrev, desc)   { sig, abbrev, desc },
#include <siglist.h>
#undef init_sig
};

static int
do_test (void)
{
  for (size_t i = 0; i < array_length (tests); i++)
    {
      TEST_COMPARE_STRING (sigabbrev_np (tests[i].errno), tests[i].abbrev);
      TEST_COMPARE_STRING (sigdescr_np (tests[i].errno), tests[i].descr);
    }

  return 0;
}

#include <support/test-driver.c>
