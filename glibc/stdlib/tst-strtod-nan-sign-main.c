/* Test strtod functions handle signs of NaNs (bug 23007).
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <stdlib.h>
#include <wchar.h>

#include <stdlib/tst-strtod.h>
#include <support/check.h>

#define CONCAT_(X, Y) X ## Y
#define CONCAT(X, Y) CONCAT_ (X, Y)
#define FNX(FN) CONCAT (FNPFX, FN)

#define TEST_STRTOD(FSUF, FTYPE, FTOSTR, LSUF, CSUF)	\
static int						\
test_strto ## FSUF (void)				\
{							\
  FTYPE val_pos = FNX (FSUF) (L_("nan"), NULL);		\
  FTYPE copy_pos = copysign ## CSUF (1, val_pos);	\
  TEST_VERIFY (isnan (val_pos) && copy_pos == 1);	\
  FTYPE val_neg = FNX (FSUF) (L_("-nan"), NULL);	\
  FTYPE copy_neg = copysign ## CSUF (1, val_neg);	\
  TEST_VERIFY (isnan (val_neg) && copy_neg == -1);	\
  return 0;						\
}
GEN_TEST_STRTOD_FOREACH (TEST_STRTOD)

static int
do_test (void)
{
  return STRTOD_TEST_FOREACH (test_strto);
}

#include <support/test-driver.c>
