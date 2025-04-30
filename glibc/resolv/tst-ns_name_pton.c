/* Tests for ns_name_pton.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <arpa/nameser.h>
#include <array_length.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>

/* Bits which indicate which functions are supposed to report
   success.  */
enum
  {
    hnok = 1,
    dnok = 2,
    mailok = 4,
    ownok = 8,
    allnomailok = hnok | dnok | ownok,
    allok = hnok | dnok | mailok | ownok
  };

/* A string of 60 characters.  */
#define STRING60 "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"

/* A string of 63 characters (maximum label length).  */
#define STRING63 STRING60 "zzz"

/* A string of 60 bytes (non-ASCII).  */
#define STRING60OCT \
  "\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377" \
  "\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377" \
  "\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377" \
  "\377\377\377\377\377\377\377\377\377"

/* A string of 63 bytes (non-ASCII).  */
#define STRING63OCT STRING60OCT "\377\377\377"

/* A string of 60 bytes (non-ASCII, quoted decimal).  */
#define STRING60DEC \
  "\\255\\255\\255\\255\\255\\255\\255\\255\\255\\255" \
  "\\255\\255\\255\\255\\255\\255\\255\\255\\255\\255" \
  "\\255\\255\\255\\255\\255\\255\\255\\255\\255\\255" \
  "\\255\\255\\255\\255\\255\\255\\255\\255\\255\\255" \
  "\\255\\255\\255\\255\\255\\255\\255\\255\\255\\255" \
  "\\255\\255\\255\\255\\255\\255\\255\\255\\255\\255"

/* A string of 63 bytes (non-ASCII, quoted decimal).  */
#define STRING63DEC STRING60DEC "\\255\\255\\255"

/* Combines a test name with the expected results.  */
struct test_case
{
  const char *dn;
  const char *back; /* Expected test result converted using ns_name_ntop.  */
  bool fully_qualified; /* True if the domain name has a trailing dot.  */
};

static const struct test_case tests[] =
  {
    { "", ".", false },
    { ".", ".", true },
    { "..", NULL, },
    { "www", "www", false },
    { "www.", "www", true },
    { "www\\.", "www\\.", false },
    { ".www", NULL, },
    { ".www\\.", NULL, },
    { "example.com", "example.com", false },
    { "example.com.", "example.com", true },
    { ".example.com", NULL, },
    { ".example.com.", NULL, },
    { "example\\.com", "example\\.com", false },
    { "example\\.com.", "example\\.com", true },
    { "example..", NULL, },
    { "example..com", NULL, },
    { "example..com", NULL, },
    { "\\0", NULL, },
    { "\\00", NULL, },
    { "\\000", "\\000", false },
    { "\\1", NULL, },
    { "\\01", NULL, },
    { "\\001", "\\001", false },
    { "\\1x", NULL, },
    { "\\01x", NULL, },
    { "\\001x", "\\001x", false },
    { "\\256", NULL, },
    { "\\0641", "\\@1", false },
    { "\\0011", "\\0011", false },
    { STRING63, STRING63, false },
    { STRING63 ".", STRING63, true },
    { STRING63 "z", NULL, },
    { STRING63 "\\.", NULL, },
    { STRING60 "zz\\.", STRING60 "zz\\.", false },
    { STRING60 "zz\\..", STRING60 "zz\\.", true },
    { STRING63 "." STRING63 "." STRING63 "." STRING60 "z",
      STRING63 "." STRING63 "." STRING63 "." STRING60 "z", false },
    { STRING63 "." STRING63 "." STRING63 "." STRING60 "z.",
      STRING63 "." STRING63 "." STRING63 "." STRING60 "z", true },
    { STRING63 "." STRING63 "." STRING63 "." STRING60 "zz", NULL, },
    { STRING63 "." STRING63 "." STRING63 "." STRING60 "zzz", NULL, },
    { STRING63OCT "." STRING63OCT "." STRING63OCT "." STRING60OCT "\377",
      STRING63DEC "." STRING63DEC "." STRING63DEC "." STRING60DEC "\\255",
      false },
    { STRING63OCT "." STRING63OCT "." STRING63OCT "." STRING60OCT "\377.",
      STRING63DEC "." STRING63DEC "." STRING63DEC "." STRING60DEC "\\255",
      true },
    { STRING63OCT "." STRING63OCT "." STRING63OCT "." STRING60OCT
      "\377\377", NULL, },
    { STRING63OCT "." STRING63OCT "." STRING63OCT "." STRING60OCT
      "\377\377\377", NULL, },
    { "\\", NULL, },
    { "\\\\", "\\\\", false },
    { "\\\\.", "\\\\", true },
    { "\\\\\\", NULL, },
    { "a\\", NULL, },
    { "a.\\", NULL, },
    { "a.b\\", NULL, },
  };

static int
do_test (void)
{
  unsigned char *wire = xmalloc (NS_MAXCDNAME);
  char *text = xmalloc (NS_MAXDNAME);
  for (const struct test_case *test = tests; test < array_end (tests); ++test)
    {
      if (test_verbose)
        printf ("info: testing domain name [[[%s]]]\n", test->dn);
      int ret = ns_name_pton (test->dn, wire, NS_MAXCDNAME);
      if (ret == -1)
        {
          if (test->back != NULL)
            {
              support_record_failure ();
              printf ("error: unexpected decoding failure for [[%s]]\n",
                      test->dn);
            }
          /* Otherwise, we have an expected decoding failure.  */
          continue;
        }

      if (ret < -1 || ret > 1)
        {
          support_record_failure ();
          printf ("error: invalid return value %d for [[%s]]\n",
                  ret, test->dn);
          continue;
        }

      int ret2 = ns_name_ntop (wire, text, NS_MAXDNAME);

      if (ret2 < 0)
        {
          support_record_failure ();
          printf ("error: failure to convert back [[%s]]\n", test->dn);
        }

      if (test->back == NULL)
        {
          support_record_failure ();
          printf ("error: unexpected success converting [[%s]]\n", test->dn);
          if (ret2 >= 1)
            printf ("error:   result converts back to [[%s]]\n", test->dn);
          continue;
        }

      if (strcmp (text, test->back) != 0)
        {
          support_record_failure ();
          printf ("error: back-conversion of [[%s]] did not match\n",
                  test->dn);
          printf ("error:   expected: [[%s]]\n", test->back);
          printf ("error:     actual: [[%s]]\n", text);
        }

      if (ret != test->fully_qualified)
        {
          support_record_failure ();
          printf ("error: invalid fully-qualified status for [[%s]]\n",
                  test->dn);
          printf ("error:   expected: %d\n", (int) test->fully_qualified);
          printf ("error:     actual: %d\n", ret);
        }
    }

  free (text);
  free (wire);
  return 0;
}

#include <support/test-driver.c>
