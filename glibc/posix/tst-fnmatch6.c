/* Test for fnmatch handling of collating symbols (bug 26620)
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

#include <fnmatch.h>
#include <locale.h>
#include <support/check.h>
#include <support/support.h>

static int
do_test (void)
{
  xsetlocale (LC_ALL, "en_US.UTF-8");
  /* From iso14651_t1_common:
     collating-element <U004C_00B7> from "<U004C><U00B7>"
     % decomposition of LATIN CAPITAL LETTER L WITH MIDDLE DOT */
  TEST_VERIFY (fnmatch ("[[.L\xc2\xb7.]]", ".", 0) != 0);
  TEST_VERIFY (fnmatch ("[[.L\xc2\xb7.]]", "L\xc2\xb7", 0) == 0);

  return 0;
}

#include <support/test-driver.c>
