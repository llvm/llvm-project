/* Regular expression tests.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

#include <sys/types.h>
#include <mcheck.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <libc-diag.h>

#define BRE RE_SYNTAX_POSIX_BASIC
#define ERE RE_SYNTAX_POSIX_EXTENDED

static struct test_s
{
  int syntax;
  const char *pattern;
  const char *string;
  int start, res;
} tests[] = {
  {BRE, "\\<A", "CBAA", 0, -1},
  {BRE, "\\<A", "CBAA", 2, -1},
  {BRE, "A\\>", "CAAB", 1, -1},
  {BRE, "\\bA", "CBAA", 0, -1},
  {BRE, "\\bA", "CBAA", 2, -1},
  {BRE, "A\\b", "CAAB", 1, -1},
  {BRE, "\\<A", "AA", 0, 0},
  {BRE, "\\<A", "C-AA", 2, 2},
  {BRE, "A\\>", "CAA-", 1, 2},
  {BRE, "A\\>", "CAA", 1, 2},
  {BRE, "\\bA", "AA", 0, 0},
  {BRE, "\\bA", "C-AA", 2, 2},
  {BRE, "A\\b", "CAA-", 1, 2},
  {BRE, "A\\b", "CAA", 1, 2},
  {BRE, "\\<[A]", "CBAA", 0, -1},
  {BRE, "\\<[A]", "CBAA", 2, -1},
  {BRE, "[A]\\>", "CAAB", 1, -1},
  {BRE, "\\b[A]", "CBAA", 0, -1},
  {BRE, "\\b[A]", "CBAA", 2, -1},
  {BRE, "[A]\\b", "CAAB", 1, -1},
  {BRE, "\\<[A]", "AA", 0, 0},
  {BRE, "\\<[A]", "C-AA", 2, 2},
  {BRE, "[A]\\>", "CAA-", 1, 2},
  {BRE, "[A]\\>", "CAA", 1, 2},
  {BRE, "\\b[A]", "AA", 0, 0},
  {BRE, "\\b[A]", "C-AA", 2, 2},
  {BRE, "[A]\\b", "CAA-", 1, 2},
  {BRE, "[A]\\b", "CAA", 1, 2},
  {ERE, "\\b(A|!|.B)", "A=AC", 0, 0},
  {ERE, "\\b(A|!|.B)", "=AC", 0, 1},
  {ERE, "\\b(A|!|.B)", "!AC", 0, 1},
  {ERE, "\\b(A|!|.B)", "=AB", 0, 1},
  {ERE, "\\b(A|!|.B)", "DA!C", 0, 2},
  {ERE, "\\b(A|!|.B)", "=CB", 0, 1},
  {ERE, "\\b(A|!|.B)", "!CB", 0, 1},
  {ERE, "\\b(A|!|.B)", "D,B", 0, 1},
  {ERE, "\\b(A|!|.B)", "!.C", 0, -1},
  {ERE, "\\b(A|!|.B)", "BCB", 0, -1},
  {ERE, "(A|\\b)(A|B|C)", "DAAD", 0, 1},
  {ERE, "(A|\\b)(A|B|C)", "DABD", 0, 1},
  {ERE, "(A|\\b)(A|B|C)", "AD", 0, 0},
  {ERE, "(A|\\b)(A|B|C)", "C!", 0, 0},
  {ERE, "(A|\\b)(A|B|C)", "D,B", 0, 2},
  {ERE, "(A|\\b)(A|B|C)", "DA?A", 0, 3},
  {ERE, "(A|\\b)(A|B|C)", "BBC", 0, 0},
  {ERE, "(A|\\b)(A|B|C)", "DA", 0, -1},
  {ERE, "(!|\\b)(!|=|~)", "A!=\\", 0, 1},
  {ERE, "(!|\\b)(!|=|~)", "/!=A", 0, 1},
  {ERE, "(!|\\b)(!|=|~)", "A=A", 0, 1},
  {ERE, "(!|\\b)(!|=|~)", "==!=", 0, 2},
  {ERE, "(!|\\b)(!|=|~)", "==C~", 0, 3},
  {ERE, "(!|\\b)(!|=|~)", "=~=", 0, -1},
  {ERE, "(!|\\b)(!|=|~)", "~!", 0, -1},
  {ERE, "(!|\\b)(!|=|~)", "~=~", 0, -1},
  {ERE, "(\\b|A.)[ABC]", "AC", 0, 0},
  {ERE, "(\\b|A.)[ABC]", "=A", 0, 1},
  {ERE, "(\\b|A.)[ABC]", "DACC", 0, 1},
  {ERE, "(\\b|A.)[A~C]", "AC", 0, 0},
  {ERE, "(\\b|A.)[A~C]", "=A", 0, 1},
  {ERE, "(\\b|A.)[A~C]", "DACC", 0, 1},
  {ERE, "(\\b|A.)[A~C]", "B!A=", 0, 2},
  {ERE, "(\\b|A.)[A~C]", "B~C", 0, 1},
  {ERE, ".\\b.", "AA~", 0, 1},
  {ERE, ".\\b.", "=A=", 0, 0},
  {ERE, ".\\b.", "==", 0, -1},
  {ERE, ".\\b.", "ABA", 0, -1},
  {ERE, "[^k]\\b[^k]", "AA~", 0, 1},
  {ERE, "[^k]\\b[^k]", "=A=", 0, 0},
  {ERE, "[^k]\\b[^k]", "Ak~kA~", 0, 4},
  {ERE, "[^k]\\b[^k]", "==", 0, -1},
  {ERE, "[^k]\\b[^k]", "ABA", 0, -1},
  {ERE, "[^k]\\b[^k]", "Ak~", 0, -1},
  {ERE, "[^k]\\b[^k]", "k=k", 0, -1},
  {ERE, "[^C]\\b[^C]", "AA~", 0, 1},
  {ERE, "[^C]\\b[^C]", "=A=", 0, 0},
  {ERE, "[^C]\\b[^C]", "AC~CA~", 0, 4},
  {ERE, "[^C]\\b[^C]", "==", 0, -1},
  {ERE, "[^C]\\b[^C]", "ABA", 0, -1},
  {ERE, "[^C]\\b[^C]", "AC~", 0, -1},
  {ERE, "[^C]\\b[^C]", "C=C", 0, -1},
  {ERE, "\\<(A|!|.B)", "A=AC", 0, 0},
  {ERE, "\\<(A|!|.B)", "=AC", 0, 1},
  {ERE, "\\<(A|!|.B)", "!AC", 0, 1},
  {ERE, "\\<(A|!|.B)", "=AB", 0, 1},
  {ERE, "\\<(A|!|.B)", "=CB", 0, 1},
  {ERE, "\\<(A|!|.B)", "!CB", 0, 1},
  {ERE, "\\<(A|!|.B)", "DA!C", 0, -1},
  {ERE, "\\<(A|!|.B)", "D,B", 0, -1},
  {ERE, "\\<(A|!|.B)", "!.C", 0, -1},
  {ERE, "\\<(A|!|.B)", "BCB", 0, -1},
  {ERE, "(A|\\<)(A|B|C)", "DAAD", 0, 1},
  {ERE, "(A|\\<)(A|B|C)", "DABD", 0, 1},
  {ERE, "(A|\\<)(A|B|C)", "AD", 0, 0},
  {ERE, "(A|\\<)(A|B|C)", "C!", 0, 0},
  {ERE, "(A|\\<)(A|B|C)", "D,B", 0, 2},
  {ERE, "(A|\\<)(A|B|C)", "DA?A", 0, 3},
  {ERE, "(A|\\<)(A|B|C)", "BBC", 0, 0},
  {ERE, "(A|\\<)(A|B|C)", "DA", 0, -1},
  {ERE, "(!|\\<)(!|=|~)", "A!=\\", 0, 1},
  {ERE, "(!|\\<)(!|=|~)", "/!=A", 0, 1},
  {ERE, "(!|\\<)(!|=|~)", "==!=", 0, 2},
  {ERE, "(!|\\<)(!|=|~)", "==C~", 0, -1},
  {ERE, "(!|\\<)(!|=|~)", "A=A", 0, -1},
  {ERE, "(!|\\<)(!|=|~)", "=~=", 0, -1},
  {ERE, "(!|\\<)(!|=|~)", "~!", 0, -1},
  {ERE, "(!|\\<)(!|=|~)", "~=~", 0, -1},
  {ERE, "(\\<|A.)[ABC]", "AC", 0, 0},
  {ERE, "(\\<|A.)[ABC]", "=A", 0, 1},
  {ERE, "(\\<|A.)[ABC]", "DACC", 0, 1},
  {ERE, "(\\<|A.)[A~C]", "AC", 0, 0},
  {ERE, "(\\<|A.)[A~C]", "=A", 0, 1},
  {ERE, "(\\<|A.)[A~C]", "DACC", 0, 1},
  {ERE, "(\\<|A.)[A~C]", "B!A=", 0, 2},
  {ERE, "(\\<|A.)[A~C]", "B~C", 0, 2},
  {ERE, ".\\<.", "=A=", 0, 0},
  {ERE, ".\\<.", "AA~", 0, -1},
  {ERE, ".\\<.", "==", 0, -1},
  {ERE, ".\\<.", "ABA", 0, -1},
  {ERE, "[^k]\\<[^k]", "=k=A=", 0, 2},
  {ERE, "[^k]\\<[^k]", "kk~", 0, -1},
  {ERE, "[^k]\\<[^k]", "==", 0, -1},
  {ERE, "[^k]\\<[^k]", "ABA", 0, -1},
  {ERE, "[^k]\\<[^k]", "=k=", 0, -1},
  {ERE, "[^C]\\<[^C]", "=C=A=", 0, 2},
  {ERE, "[^C]\\<[^C]", "CC~", 0, -1},
  {ERE, "[^C]\\<[^C]", "==", 0, -1},
  {ERE, "[^C]\\<[^C]", "ABA", 0, -1},
  {ERE, "[^C]\\<[^C]", "=C=", 0, -1},
  {ERE, ".\\B.", "ABA", 0, 0},
  {ERE, ".\\B.", "=BDC", 0, 1},
  {ERE, "[^k]\\B[^k]", "kkkABA", 0, 3},
  {ERE, "[^k]\\B[^k]", "kBk", 0, -1},
  {ERE, "[^C]\\B[^C]", "CCCABA", 0, 3},
  {ERE, "[^C]\\B[^C]", "CBC", 0, -1},
  {ERE, ".(\\b|\\B).", "=~AB", 0, 0},
  {ERE, ".(\\b|\\B).", "A=C", 0, 0},
  {ERE, ".(\\b|\\B).", "ABC", 0, 0},
  {ERE, ".(\\b|\\B).", "=~\\!", 0, 0},
  {ERE, "[^k](\\b|\\B)[^k]", "=~AB", 0, 0},
  {ERE, "[^k](\\b|\\B)[^k]", "A=C", 0, 0},
  {ERE, "[^k](\\b|\\B)[^k]", "ABC", 0, 0},
  {ERE, "[^k](\\b|\\B)[^k]", "=~kBD", 0, 0},
  {ERE, "[^k](\\b|\\B)[^k]", "=~\\!", 0, 0},
  {ERE, "[^k](\\b|\\B)[^k]", "=~kB", 0, 0},
  {ERE, "[^C](\\b|\\B)[^C]", "=~AB", 0, 0},
  {ERE, "[^C](\\b|\\B)[^C]", "A=C", 0, 0},
  {ERE, "[^C](\\b|\\B)[^C]", "ABC", 0, 0},
  {ERE, "[^C](\\b|\\B)[^C]", "=~CBD", 0, 0},
  {ERE, "[^C](\\b|\\B)[^C]", "=~\\!", 0, 0},
  {ERE, "[^C](\\b|\\B)[^C]", "=~CB", 0, 0},
  {ERE, "\\b([A]|[!]|.B)", "A=AC", 0, 0},
  {ERE, "\\b([A]|[!]|.B)", "=AC", 0, 1},
  {ERE, "\\b([A]|[!]|.B)", "!AC", 0, 1},
  {ERE, "\\b([A]|[!]|.B)", "=AB", 0, 1},
  {ERE, "\\b([A]|[!]|.B)", "DA!C", 0, 2},
  {ERE, "\\b([A]|[!]|.B)", "=CB", 0, 1},
  {ERE, "\\b([A]|[!]|.B)", "!CB", 0, 1},
  {ERE, "\\b([A]|[!]|.B)", "D,B", 0, 1},
  {ERE, "\\b([A]|[!]|.B)", "!.C", 0, -1},
  {ERE, "\\b([A]|[!]|.B)", "BCB", 0, -1},
  {ERE, "([A]|\\b)([A]|[B]|[C])", "DAAD", 0, 1},
  {ERE, "([A]|\\b)([A]|[B]|[C])", "DABD", 0, 1},
  {ERE, "([A]|\\b)([A]|[B]|[C])", "AD", 0, 0},
  {ERE, "([A]|\\b)([A]|[B]|[C])", "C!", 0, 0},
  {ERE, "([A]|\\b)([A]|[B]|[C])", "D,B", 0, 2},
  {ERE, "([A]|\\b)([A]|[B]|[C])", "DA?A", 0, 3},
  {ERE, "([A]|\\b)([A]|[B]|[C])", "BBC", 0, 0},
  {ERE, "([A]|\\b)([A]|[B]|[C])", "DA", 0, -1},
  {ERE, "([!]|\\b)([!]|[=]|[~])", "A!=\\", 0, 1},
  {ERE, "([!]|\\b)([!]|[=]|[~])", "/!=A", 0, 1},
  {ERE, "([!]|\\b)([!]|[=]|[~])", "A=A", 0, 1},
  {ERE, "([!]|\\b)([!]|[=]|[~])", "==!=", 0, 2},
  {ERE, "([!]|\\b)([!]|[=]|[~])", "==C~", 0, 3},
  {ERE, "([!]|\\b)([!]|[=]|[~])", "=~=", 0, -1},
  {ERE, "([!]|\\b)([!]|[=]|[~])", "~!", 0, -1},
  {ERE, "([!]|\\b)([!]|[=]|[~])", "~=~", 0, -1},
  {ERE, "\\<([A]|[!]|.B)", "A=AC", 0, 0},
  {ERE, "\\<([A]|[!]|.B)", "=AC", 0, 1},
  {ERE, "\\<([A]|[!]|.B)", "!AC", 0, 1},
  {ERE, "\\<([A]|[!]|.B)", "=AB", 0, 1},
  {ERE, "\\<([A]|[!]|.B)", "=CB", 0, 1},
  {ERE, "\\<([A]|[!]|.B)", "!CB", 0, 1},
  {ERE, "\\<([A]|[!]|.B)", "DA!C", 0, -1},
  {ERE, "\\<([A]|[!]|.B)", "D,B", 0, -1},
  {ERE, "\\<([A]|[!]|.B)", "!.C", 0, -1},
  {ERE, "\\<([A]|[!]|.B)", "BCB", 0, -1},
  {ERE, "([A]|\\<)([A]|[B]|[C])", "DAAD", 0, 1},
  {ERE, "([A]|\\<)([A]|[B]|[C])", "DABD", 0, 1},
  {ERE, "([A]|\\<)([A]|[B]|[C])", "AD", 0, 0},
  {ERE, "([A]|\\<)([A]|[B]|[C])", "C!", 0, 0},
  {ERE, "([A]|\\<)([A]|[B]|[C])", "D,B", 0, 2},
  {ERE, "([A]|\\<)([A]|[B]|[C])", "DA?A", 0, 3},
  {ERE, "([A]|\\<)([A]|[B]|[C])", "BBC", 0, 0},
  {ERE, "([A]|\\<)([A]|[B]|[C])", "DA", 0, -1},
  {ERE, "([!]|\\<)([!=]|[~])", "A!=\\", 0, 1},
  {ERE, "([!]|\\<)([!=]|[~])", "/!=A", 0, 1},
  {ERE, "([!]|\\<)([!=]|[~])", "==!=", 0, 2},
  {ERE, "([!]|\\<)([!=]|[~])", "==C~", 0, -1},
  {ERE, "([!]|\\<)([!=]|[~])", "A=A", 0, -1},
  {ERE, "([!]|\\<)([!=]|[~])", "=~=", 0, -1},
  {ERE, "([!]|\\<)([!=]|[~])", "~!", 0, -1},
  {ERE, "([!]|\\<)([!=]|[~])", "~=~", 0, -1},
  {ERE, "(\\<|[A].)[ABC]", "AC", 0, 0},
  {ERE, "(\\<|[A].)[ABC]", "=A", 0, 1},
  {ERE, "(\\<|[A].)[ABC]", "DACC", 0, 1},
  {ERE, "(\\<|[A].)[A~C]", "AC", 0, 0},
  {ERE, "(\\<|[A].)[A~C]", "=A", 0, 1},
  {ERE, "(\\<|[A].)[A~C]", "DACC", 0, 1},
  {ERE, "(\\<|[A].)[A~C]", "B!A=", 0, 2},
  {ERE, "(\\<|[A].)[A~C]", "B~C", 0, 2},
  {ERE, "^[^A]*\\bB", "==B", 0, 0},
  {ERE, "^[^A]*\\bB", "CBD!=B", 0, 0},
  {ERE, "[^A]*\\bB", "==B", 2, 2}
};

int
do_one_test (const struct test_s *test, const char *fail)
{
  int res;
  const char *err;
  struct re_pattern_buffer regbuf;

  re_set_syntax (test->syntax);
  memset (&regbuf, '\0', sizeof (regbuf));
  err = re_compile_pattern (test->pattern, strlen (test->pattern),
			    &regbuf);
  if (err != NULL)
    {
      printf ("%sre_compile_pattern \"%s\" failed: %s\n", fail, test->pattern,
	      err);
      return 1;
    }

#if __GNUC_PREREQ (10, 0) && !__GNUC_PREREQ (11, 0)
  DIAG_PUSH_NEEDS_COMMENT;
  /* Avoid GCC 10 false positive warning: specified size exceeds maximum
     object size.  */
  DIAG_IGNORE_NEEDS_COMMENT (10, "-Wstringop-overflow");
#endif
  res = re_search (&regbuf, test->string, strlen (test->string),
		   test->start, strlen (test->string) - test->start, NULL);
#if __GNUC_PREREQ (10, 0) && !__GNUC_PREREQ (11, 0)
  DIAG_POP_NEEDS_COMMENT;
#endif
  if (res != test->res)
    {
      printf ("%sre_search \"%s\" \"%s\" failed: %d (expected %d)\n",
	      fail, test->pattern, test->string, res, test->res);
      regfree (&regbuf);
      return 1;
    }

  if (test->res > 0 && test->start == 0)
    {
#if __GNUC_PREREQ (10, 0) && !__GNUC_PREREQ (11, 0)
  DIAG_PUSH_NEEDS_COMMENT;
  /* Avoid GCC 10 false positive warning: specified size exceeds maximum
     object size.  */
  DIAG_IGNORE_NEEDS_COMMENT (10, "-Wstringop-overflow");
#endif
      res = re_search (&regbuf, test->string, strlen (test->string),
		       test->res, strlen (test->string) - test->res, NULL);
#if __GNUC_PREREQ (10, 0) && !__GNUC_PREREQ (11, 0)
  DIAG_POP_NEEDS_COMMENT;
#endif
      if (res != test->res)
	{
	  printf ("%sre_search from expected \"%s\" \"%s\" failed: %d (expected %d)\n",
		  fail, test->pattern, test->string, res, test->res);
	  regfree (&regbuf);
	  return 1;
	}
    }

  regfree (&regbuf);
  return 0;
}

static char *
replace (char *p, char c)
{
  switch (c)
    {
      /* A -> A" */
      case 'A': *p++ = '\xc3'; *p++ = '\x84'; break;
      /* B -> O" */
      case 'B': *p++ = '\xc3'; *p++ = '\x96'; break;
      /* C -> U" */
      case 'C': *p++ = '\xc3'; *p++ = '\x9c'; break;
      /* D -> a" */
      case 'D': *p++ = '\xc3'; *p++ = '\xa4'; break;
      /* ! -> MULTIPLICATION SIGN */
      case '!': *p++ = '\xc3'; *p++ = '\x97'; break;
      /* = -> EM DASH */
      case '=': *p++ = '\xe2'; *p++ = '\x80'; *p++ = '\x94'; break;
      /* ~ -> MUSICAL SYMBOL HALF NOTE */
      case '~': *p++ = '\xf0'; *p++ = '\x9d'; *p++ = '\x85'; *p++ = '\x9e';
      break;
    }
  return p;
}

int
do_mb_tests (const struct test_s *test)
{
  int i, j;
  struct test_s t;
  const char *const chars = "ABCD!=~";
  char repl[8], *p;
  char pattern[strlen (test->pattern) * 4 + 1];
  char string[strlen (test->string) * 4 + 1];
  char fail[8 + sizeof ("UTF-8 ")];

  t = *test;
  t.pattern = pattern;
  t.string = string;
  strcpy (fail, "UTF-8 ");
  for (i = 1; i < 128; ++i)
    {
      p = repl;
      for (j = 0; j < 7; ++j)
	if (i & (1 << j))
	  {
	    if (!strchr (test->pattern, chars[j])
		&& !strchr (test->string, chars[j]))
	      break;
	    *p++ = chars[j];
	  }
      if (j < 7)
	continue;
      *p = '\0';

      for (j = 0, p = pattern; test->pattern[j]; ++j)
	if (strchr (repl, test->pattern[j]))
	  p = replace (p, test->pattern[j]);
	else if (test->pattern[j] == '\\' && test->pattern[j + 1])
	  {
	    *p++ = test->pattern[j++];
	    *p++ = test->pattern[j];
	  }
	else
	  *p++ = test->pattern[j];
      *p = '\0';

      t.start = test->start;
      t.res = test->res;

      for (j = 0, p = string; test->string[j]; ++j)
	if (strchr (repl, test->string[j]))
	  {
	    char *d = replace (p, test->string[j]);
	    if (test->start > j)
	      t.start += d - p - 1;
	    if (test->res > j)
	      t.res += d - p - 1;
	    p = d;
	  }
	else
	  *p++ = test->string[j];
      *p = '\0';

      p = stpcpy (fail + strlen ("UTF-8 "), repl);
      *p++ = ' ';
      *p = '\0';

      if (do_one_test (&t, fail))
	return 1;
    }
  return 0;
}

int
main (void)
{
  size_t i;
  int ret = 0;

  mtrace ();

  for (i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      if (setlocale (LC_ALL, "de_DE.ISO-8859-1") == NULL)
	{
	  puts ("setlocale de_DE.ISO-8859-1 failed");
	  ret = 1;
	}
      ret |= do_one_test (&tests[i], "");
      if (setlocale (LC_ALL, "de_DE.UTF-8") == NULL)
	{
	  puts ("setlocale de_DE.UTF-8 failed");
	  ret = 1;
	}
      ret |= do_one_test (&tests[i], "UTF-8 ");
      ret |= do_mb_tests (&tests[i]);
    }

  return ret;
}
