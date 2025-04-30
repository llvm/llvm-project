/* Unit tests for dl-hwcaps.c.
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

#include <array_length.h>
#include <dl-hwcaps.h>
#include <string.h>
#include <support/check.h>

static void
check_split_masked (const char *input, int32_t bitmask, const char *mask,
                    const char *expected[], size_t expected_length)
{
  struct dl_hwcaps_split_masked split;
  _dl_hwcaps_split_masked_init (&split, input, bitmask, mask);
  size_t index = 0;
  while (_dl_hwcaps_split_masked (&split))
    {
      TEST_VERIFY_EXIT (index < expected_length);
      TEST_COMPARE_BLOB (expected[index], strlen (expected[index]),
                         split.split.segment, split.split.length);
      ++index;
    }
  TEST_COMPARE (index, expected_length);
}

static void
check_split (const char *input,
             const char *expected[], size_t expected_length)
{
  struct dl_hwcaps_split split;
  _dl_hwcaps_split_init (&split, input);
  size_t index = 0;
  while (_dl_hwcaps_split (&split))
    {
      TEST_VERIFY_EXIT (index < expected_length);
      TEST_COMPARE_BLOB (expected[index], strlen (expected[index]),
                         split.segment, split.length);
      ++index;
    }
  TEST_COMPARE (index, expected_length);

  /* Reuse the test cases with masking that does not actually remove
     anything.  */
  check_split_masked (input, -1, NULL, expected, expected_length);
  check_split_masked (input, -1, input, expected, expected_length);
}

static int
do_test (void)
{
  /* Splitting tests, without masking.  */
  check_split (NULL, NULL, 0);
  check_split ("", NULL, 0);
  check_split (":", NULL, 0);
  check_split ("::", NULL, 0);

  {
    const char *expected[] = { "first" };
    check_split ("first", expected, array_length (expected));
    check_split (":first", expected, array_length (expected));
    check_split ("first:", expected, array_length (expected));
    check_split (":first:", expected, array_length (expected));
  }

  {
    const char *expected[] = { "first", "second" };
    check_split ("first:second", expected, array_length (expected));
    check_split ("first::second", expected, array_length (expected));
    check_split (":first:second", expected, array_length (expected));
    check_split ("first:second:", expected, array_length (expected));
    check_split (":first:second:", expected, array_length (expected));
  }

  /* Splitting tests with masking.  */
  {
    const char *expected[] = { "first" };
    check_split_masked ("first", 3, "first:second",
                        expected, array_length (expected));
    check_split_masked ("first:second", 3, "first:",
                        expected, array_length (expected));
    check_split_masked ("first:second", 1, NULL,
                        expected, array_length (expected));
  }
  {
    const char *expected[] = { "second" };
    check_split_masked ("first:second", 3, "second",
                        expected, array_length (expected));
    check_split_masked ("first:second:third", -1, "second:",
                        expected, array_length (expected));
    check_split_masked ("first:second", 2, NULL,
                        expected, array_length (expected));
    check_split_masked ("first:second:third", 2, "first:second",
                        expected, array_length (expected));
  }

  /* Tests for _dl_hwcaps_contains.  */
  TEST_VERIFY (_dl_hwcaps_contains (NULL, "first", strlen ("first")));
  TEST_VERIFY (_dl_hwcaps_contains (NULL, "", 0));
  TEST_VERIFY (! _dl_hwcaps_contains ("", "first", strlen ("first")));
  TEST_VERIFY (! _dl_hwcaps_contains ("firs", "first", strlen ("first")));
  TEST_VERIFY (_dl_hwcaps_contains ("firs", "first", strlen ("first") - 1));
  for (int i = 0; i < strlen ("first"); ++i)
    TEST_VERIFY (! _dl_hwcaps_contains ("first", "first", i));
  TEST_VERIFY (_dl_hwcaps_contains ("first", "first", strlen ("first")));
  TEST_VERIFY (_dl_hwcaps_contains ("first:", "first", strlen ("first")));
  TEST_VERIFY (_dl_hwcaps_contains ("first:second",
                                    "first", strlen ("first")));
  TEST_VERIFY (_dl_hwcaps_contains (":first:second", "first",
                                    strlen ("first")));
  TEST_VERIFY (_dl_hwcaps_contains ("first:second", "second",
                                    strlen ("second")));
  TEST_VERIFY (_dl_hwcaps_contains ("first:second:", "second",
                                    strlen ("second")));
  TEST_VERIFY (_dl_hwcaps_contains ("first::second:", "second",
                                    strlen ("second")));
  TEST_VERIFY (_dl_hwcaps_contains ("first:second::", "second",
                                    strlen ("second")));
  for (int i = 0; i < strlen ("second"); ++i)
    {
      TEST_VERIFY (!_dl_hwcaps_contains ("first:second", "second", i));
      TEST_VERIFY (!_dl_hwcaps_contains ("first:second:", "second", i));
      TEST_VERIFY (!_dl_hwcaps_contains ("first:second::", "second", i));
      TEST_VERIFY (!_dl_hwcaps_contains ("first::second", "second", i));
    }

  return 0;
}

#include <support/test-driver.c>

/* Rebuild the sources here because the object file is built for
   inclusion into the dynamic loader.  */
#include "dl-hwcaps_split.c"
