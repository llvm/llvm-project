/* Test for invalid field handling in file-style NSS databases.  [BZ #18724]
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* This test needs to be statically linked because it access hidden
   functions.  */

#include <nss.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <support/support.h>

static bool errors;

static void
check (const char *what, bool expr)
{
  if (!expr)
    {
      printf ("FAIL: %s\n", what);
      errors = true;
    }
}

#define CHECK(expr) check (#expr, (expr))

static void
check_rewrite (const char *input, const char *expected)
{
  char *to_free;
  const char *result = __nss_rewrite_field (input, &to_free);
  CHECK (result != NULL);
  if (result != NULL && strcmp (result, expected) != 0)
    {
      printf ("FAIL: rewrite \"%s\" -> \"%s\", expected \"%s\"\n",
	      (input == NULL) ? "(null)" : input, result,
	      (expected == NULL) ? "(null)" : expected);
      errors = true;
    }
  free (to_free);
}

static int
do_test (void)
{
  CHECK (__nss_valid_field (NULL));
  CHECK (__nss_valid_field (""));
  CHECK (__nss_valid_field ("+"));
  CHECK (__nss_valid_field ("-"));
  CHECK (__nss_valid_field (" "));
  CHECK (__nss_valid_field ("abcdef"));
  CHECK (__nss_valid_field ("abc def"));
  CHECK (__nss_valid_field ("abc\tdef"));
  CHECK (!__nss_valid_field ("abcdef:"));
  CHECK (!__nss_valid_field ("abcde:f"));
  CHECK (!__nss_valid_field (":abcdef"));
  CHECK (!__nss_valid_field ("abcdef\n"));
  CHECK (!__nss_valid_field ("\nabcdef"));
  CHECK (!__nss_valid_field (":"));
  CHECK (!__nss_valid_field ("\n"));

  CHECK (__nss_valid_list_field (NULL));
  CHECK (__nss_valid_list_field ((char *[]) {(char *) "good", NULL}));
  CHECK (!__nss_valid_list_field ((char *[]) {(char *) "g,ood", NULL}));
  CHECK (!__nss_valid_list_field ((char *[]) {(char *) "g\nood", NULL}));
  CHECK (!__nss_valid_list_field ((char *[]) {(char *) "g:ood", NULL}));

  check_rewrite (NULL, "");
  check_rewrite ("", "");
  check_rewrite ("abc", "abc");
  check_rewrite ("abc\n", "abc ");
  check_rewrite ("abc:", "abc ");
  check_rewrite ("\nabc", " abc");
  check_rewrite (":abc", " abc");
  check_rewrite (":", " ");
  check_rewrite ("\n", " ");
  check_rewrite ("a:b:c", "a b c");
  check_rewrite ("a\nb\nc", "a b c");
  check_rewrite ("a\nb:c", "a b c");
  check_rewrite ("aa\nbb\ncc", "aa bb cc");
  check_rewrite ("aa\nbb:cc", "aa bb cc");

  return errors;
}

#include <support/test-driver.c>
