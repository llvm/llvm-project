/* Test cabs.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include "libm-test-driver.c"

static const struct test_c_f_data cabs_test_data[] =
  {
    /* cabs (x + iy) is specified as hypot (x,y) */

    /* cabs (+inf + i x) == +inf.  */
    TEST_c_f (cabs, plus_infty, 1.0, plus_infty),
    /* cabs (-inf + i x) == +inf.  */
    TEST_c_f (cabs, minus_infty, 1.0, plus_infty),

    TEST_c_f (cabs, minus_infty, qnan_value, plus_infty),
    TEST_c_f (cabs, minus_infty, qnan_value, plus_infty),

    TEST_c_f (cabs, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_f (cabs),
  };

static void
cabs_test (void)
{
  ALL_RM_TEST (cabs, 0, cabs_test_data, RUN_TEST_LOOP_c_f, END);
}

static void
do_test (void)
{
  cabs_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
