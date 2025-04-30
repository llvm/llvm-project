/* Test totalordermag compat symbol.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>
#include <first-versions.h>
#define COMPAT_TEST
#include "libm-test-totalordermag.c"

#define CONCATX(x, y) x ## y
#define CONCAT(x, y) CONCATX (x, y)
#define COMPAT_VER CONCAT (FIRST_VERSION_libm_, FUNC_TEST (totalordermag))

#if TEST_COMPAT (libm, COMPAT_VER, GLIBC_2_31)

int FUNC_TEST (compat_totalordermag) (FLOAT, FLOAT);
compat_symbol_reference (libm,
			 FUNC_TEST (compat_totalordermag),
			 FUNC_TEST (totalordermag),
			 COMPAT_VER);

static void
compat_totalordermag_test (void)
{
  ALL_RM_TEST (compat_totalordermag, 1, totalordermag_test_data, RUN_TEST_LOOP_ff_b, END);
}

#endif

static void
do_test (void)
{
#if TEST_COMPAT (libm, COMPAT_VER, GLIBC_2_31)
  compat_totalordermag_test ();
#endif
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
