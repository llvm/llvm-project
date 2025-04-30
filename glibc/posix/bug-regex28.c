/* Test RE_HAT_LISTS_NOT_NEWLINE and RE_DOT_NEWLINE.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2007.

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

#include <regex.h>
#include <stdio.h>
#include <string.h>

#include <support/test-driver.h>
#include <support/check.h>

struct tests
{
  const char *regex;
  const char *string;
  reg_syntax_t syntax;
  int retval;
};
static const struct tests tests[] = {
#define EGREP RE_SYNTAX_EGREP
#define EGREP_NL (RE_SYNTAX_EGREP | RE_DOT_NEWLINE) & ~RE_HAT_LISTS_NOT_NEWLINE
  { "a.b", "a\nb", EGREP, 0 },
  { "a.b", "a\nb", EGREP_NL, 0 },
  { "a[^x]b", "a\nb", EGREP, 0 },
  { "a[^x]b", "a\nb", EGREP_NL, 0 },
  /* While \S and \W are internally handled as [^[:space:]] and [^[:alnum:]_],
     RE_HAT_LISTS_NOT_NEWLINE did not make any difference, so ensure
     it doesn't change.  */
  { "a\\Sb", "a\nb", EGREP, -1 },
  { "a\\Sb", "a\nb", EGREP_NL, -1 },
  { "a\\Wb", "a\nb", EGREP, 0 },
  { "a\\Wb", "a\nb", EGREP_NL, 0 }
};
static const size_t tests_size = sizeof (tests) / sizeof (tests[0]);

static int
do_test (void)
{
  struct re_pattern_buffer r;

  for (size_t i = 0; i < tests_size; i++)
    {
      re_set_syntax (tests[i].syntax);
      memset (&r, 0, sizeof (r));
      const char *re = re_compile_pattern (tests[i].regex,
					   strlen (tests[i].regex), &r);
      TEST_VERIFY (re == NULL);
      if (re != NULL)
        continue;

      size_t len = strlen (tests[i].string);
      int rv = re_search (&r, tests[i].string, len, 0, len, NULL);
      TEST_VERIFY (rv == tests[i].retval);
      if (test_verbose > 0)
        printf ("info: i=%zu rv=%d expected=%d\n", i, rv, tests[i].retval);

      regfree (&r);
    }

  return 0;
}

#include <support/test-driver.c>
