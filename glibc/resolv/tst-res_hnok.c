/* Tests for res_hnok and related functions.
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

#include <array_length.h>
#include <resolv.h>
#include <string.h>
#include <support/check.h>
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

/* Combines a test name with the expected results.  */
struct test_case
{
  const char *dn;
  unsigned int result;          /* Combination of the *ok flags.  */
};

static const struct test_case tests[] =
  {
    { "", allok },
    { ".", allok },
    { "..", 0 },
    { "www", allnomailok },
    { "www.", allnomailok },
    { "example", allnomailok },
    { "example.com", allok },
    { "www.example.com", allok },
    { "www.example.com.", allok },
    { "www-.example.com.", allok },
    { "www.-example.com.", allok },
    { "*.example.com", dnok | mailok | ownok },
    { "-v", dnok },
    { "-v.example.com", mailok | dnok },
    { "**.example.com", dnok | mailok },
    { "www.example.com\\", 0 },
    { STRING63, allnomailok },
    { STRING63 ".", allnomailok },
    { STRING63 "\\.", 0 },
    { STRING63 "z", 0 },
    { STRING63 ".example.com", allok },
    { STRING63 "." STRING63 "." STRING63 "." STRING60 "z", allok },
    { STRING63 "." STRING63 "." STRING63 "." STRING60 "z.", allok },
    { STRING63 "." STRING63 "." STRING63 "." STRING60 "zz", 0 },
    { STRING63 "." STRING63 "." STRING63 "." STRING60 "zzz", 0 },
    { "hostmaster@mail.example.com", dnok | mailok },
    { "hostmaster\\@mail.example.com", dnok | mailok },
    { "with whitespace", 0 },
    { "with\twhitespace", 0 },
    { "with\nwhitespace", 0 },
    { "with.whitespace ", 0 },
    { "with.whitespace\t", 0 },
    { "with.whitespace\n", 0 },
    { "with\\ whitespace", 0 },
    { "with\\\twhitespace", 0 },
    { "with\\\nwhitespace", 0 },
    { "with.whitespace\\ ", 0 },
    { "with.whitespace\\\t", 0 },
    { "with.whitespace\\\n", 0 },
  };

/* Run test case *TEST with FUNC (named FUNCNAME) and report an error
   if the result does not match the result flag at BIT.  */
static void
one_test (const struct test_case *test, const char *funcname,
          int (*func) (const char *), unsigned int bit)
{
  int expected = (test->result & bit) != 0;
  int actual = func (test->dn);
  if (actual != expected)
    {
      support_record_failure ();
      printf ("error: %s (\"%s\"): expected=%d, actual=%d\n",
              funcname, test->dn, expected, actual);
    }
}

/* Run 255 tests using all the bytes from 1 to 255, surround the byte
   with the strings PREFIX and SUFFIX, and check that FUNC (named
   FUNCNAME) accepts only those bytes listed in ACCEPTED.  */
static void
one_char (const char *prefix, const char *accepted, const char *suffix,
          const char *funcname, int (*func) (const char *))
{
  for (int ch = 1; ch <= 255; ++ch)
    {
      char dn[1024];
      snprintf (dn, sizeof (dn), "%s%c%s", prefix, ch, suffix);
      int expected = strchr (accepted, ch) != NULL;
      int actual = func (dn);
      if (actual != expected)
        {
          support_record_failure ();
          printf ("error: %s (\"%s\"): expected=%d, actual=%d\n",
                  funcname, dn, expected, actual);
        }
    }
}

#define LETTERSDIGITS \
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

#define PRINTABLE \
  "!\"#$%&'()*+,/:;<=>?@[\\]^`{|}~"

static int
do_test (void)
{
  for (const struct test_case *test = tests; test < array_end (tests); ++test)
    {
      if (test_verbose)
        printf ("info: testing domain name [[[%s]]] (0x%x)\n",
                test->dn, test->result);
      one_test (test, "res_hnok", res_hnok, hnok);
      one_test (test, "res_dnok", res_dnok, dnok);
      one_test (test, "res_mailok", res_mailok, mailok);
      one_test (test, "res_ownok", res_ownok, ownok);
    }

  one_char
    ("", LETTERSDIGITS "._", "", "res_hnok", res_hnok);
  one_char
    ("middle",
     LETTERSDIGITS ".-_\\", /* "middle\\suffix" == "middlesuffix", so good.  */
     "suffix", "res_hnok", res_hnok);
  one_char
    ("middle",
     LETTERSDIGITS ".-_" PRINTABLE,
     "suffix.example", "res_mailok", res_mailok);
  one_char
    ("mailbox.middle",
     LETTERSDIGITS ".-_\\",
     "suffix.example", "res_mailok", res_mailok);

  return 0;
}

#include <support/test-driver.c>
