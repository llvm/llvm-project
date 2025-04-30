/* Test heuristic for recognizing DSO file names.
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

#include <dl-is_dso.h>
#include <gnu/lib-names.h>
#include <support/check.h>

static int
do_test (void)
{
  /* Official ABI names.  */
  TEST_VERIFY (_dl_is_dso (LIBC_SO));
  TEST_VERIFY (_dl_is_dso (LD_SO));
  /* Version-based names.  The version number does not matter.  */
  TEST_VERIFY (_dl_is_dso ("libc-2.12.so"));
  TEST_VERIFY (_dl_is_dso ("ld-2.12.so"));
  return 0;
}

#include <support/test-driver.c>
