/* Regular expression tests.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2002.

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

/* Tests supposed to match.  */
struct
{
  const char *pattern;
  const char *string;
  int flags, nmatch;
  regmatch_t rm[5];
} tests[] = {
  /* Test for newline handling in regex.  */
  { "[^~]*~", "\nx~y", 0, 2, { { 0, 3 }, { -1, -1 } } },
  /* Other tests.  */
  { "a(.*)b", "a b", REG_EXTENDED, 2, { { 0, 3 }, { 1, 2 } } },
  { ".*|\\([KIO]\\)\\([^|]*\\).*|?[KIO]", "10~.~|P|K0|I10|O16|?KSb", 0, 3,
    { { 0, 21 }, { 15, 16 }, { 16, 18 } } },
  { ".*|\\([KIO]\\)\\([^|]*\\).*|?\\1", "10~.~|P|K0|I10|O16|?KSb", 0, 3,
    { { 0, 21 }, { 8, 9 }, { 9, 10 } } },
  { "^\\(a*\\)\\1\\{9\\}\\(a\\{0,9\\}\\)\\([0-9]*;.*[^a]\\2\\([0-9]\\)\\)",
    "a1;;0a1aa2aaa3aaaa4aaaaa5aaaaaa6aaaaaaa7aaaaaaaa8aaaaaaaaa9aa2aa1a0", 0,
    5, { { 0, 67 }, { 0, 0 }, { 0, 1 }, { 1, 67 }, { 66, 67 } } },
  /* Test for BRE expression anchoring.  POSIX says just that this may match;
     in glibc regex it always matched, so avoid changing it.  */
  { "\\(^\\|foo\\)bar", "bar", 0, 2, { { 0, 3 }, { -1, -1 } } },
  { "\\(foo\\|^\\)bar", "bar", 0, 2, { { 0, 3 }, { -1, -1 } } },
  /* In ERE this must be treated as an anchor.  */
  { "(^|foo)bar", "bar", REG_EXTENDED, 2, { { 0, 3 }, { -1, -1 } } },
  { "(foo|^)bar", "bar", REG_EXTENDED, 2, { { 0, 3 }, { -1, -1 } } },
  /* Here ^ cannot be treated as an anchor according to POSIX.  */
  { "(^|foo)bar", "(^|foo)bar", 0, 2, { { 0, 10 }, { -1, -1 } } },
  { "(foo|^)bar", "(foo|^)bar", 0, 2, { { 0, 10 }, { -1, -1 } } },
  /* More tests on backreferences.  */
  { "()\\1", "x", REG_EXTENDED, 2, { { 0, 0 }, { 0, 0 } } },
  { "()x\\1", "x", REG_EXTENDED, 2, { { 0, 1 }, { 0, 0 } } },
  { "()\\1*\\1*", "", REG_EXTENDED, 2, { { 0, 0 }, { 0, 0 } } },
  { "([0-9]).*\\1(a*)", "7;7a6", REG_EXTENDED, 3, { { 0, 4 }, { 0, 1 }, { 3, 4 } } },
  { "([0-9]).*\\1(a*)", "7;7a", REG_EXTENDED, 3, { { 0, 4 }, { 0, 1 }, { 3, 4 } } },
  { "(b)()c\\1", "bcb", REG_EXTENDED, 3, { { 0, 3 }, { 0, 1 }, { 1, 1 } } },
  { "()(b)c\\2", "bcb", REG_EXTENDED, 3, { { 0, 3 }, { 0, 0 }, { 0, 1 } } },
  { "a(b)()c\\1", "abcb", REG_EXTENDED, 3, { { 0, 4 }, { 1, 2 }, { 2, 2 } } },
  { "a()(b)c\\2", "abcb", REG_EXTENDED, 3, { { 0, 4 }, { 1, 1 }, { 1, 2 } } },
  { "()(b)\\1c\\2", "bcb", REG_EXTENDED, 3, { { 0, 3 }, { 0, 0 }, { 0, 1 } } },
  { "(b())\\2\\1", "bbbb", REG_EXTENDED, 3, { { 0, 2 }, { 0, 1 }, { 1, 1 } } },
  { "a()(b)\\1c\\2", "abcb", REG_EXTENDED, 3, { { 0, 4 }, { 1, 1 }, { 1, 2 } } },
  { "a()d(b)\\1c\\2", "adbcb", REG_EXTENDED, 3, { { 0, 5 }, { 1, 1 }, { 2, 3 } } },
  { "a(b())\\2\\1", "abbbb", REG_EXTENDED, 3, { { 0, 3 }, { 1, 2 }, { 2, 2 } } },
  { "(bb())\\2\\1", "bbbb", REG_EXTENDED, 3, { { 0, 4 }, { 0, 2 }, { 2, 2 } } },
  { "^([^,]*),\\1,\\1$", "a,a,a", REG_EXTENDED, 2, { { 0, 5 }, { 0, 1 } } },
  { "^([^,]*),\\1,\\1$", "ab,ab,ab", REG_EXTENDED, 2, { { 0, 8 }, { 0, 2 } } },
  { "^([^,]*),\\1,\\1,\\1$", "abc,abc,abc,abc", REG_EXTENDED, 2,
    { { 0, 15 }, { 0, 3 } } },
  { "^(.?)(.?)(.?)(.?)(.?).?\\5\\4\\3\\2\\1$",
    "level", REG_NOSUB | REG_EXTENDED, 0, { { -1, -1 } } },
  { "^(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.).?\\9\\8\\7\\6\\5\\4\\3\\2\\1$|^.?$",
    "level", REG_NOSUB | REG_EXTENDED, 0, { { -1, -1 } } },
  { "^(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.).?\\9\\8\\7\\6\\5\\4\\3\\2\\1$|^.?$",
    "abcdedcba", REG_EXTENDED, 1, { { 0, 9 } } },
#if 0
  /* XXX Not used since they fail so far.  */
  { "^(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.).?\\9\\8\\7\\6\\5\\4\\3\\2\\1$|^.?$",
    "ababababa", REG_EXTENDED, 1, { { 0, 9 } } },
  { "^(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.?).?\\9\\8\\7\\6\\5\\4\\3\\2\\1$",
    "level", REG_NOSUB | REG_EXTENDED, 0, { { -1, -1 } } },
  { "^(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.?)(.?).?\\9\\8\\7\\6\\5\\4\\3\\2\\1$",
    "ababababa", REG_EXTENDED, 1, { { 0, 9 } } },
#endif
};

int
main (void)
{
  regex_t re;
  regmatch_t rm[5];
  size_t i;
  int n, ret = 0;

  mtrace ();

  for (i = 0; i < sizeof (tests) / sizeof (tests[0]); ++i)
    {
      n = regcomp (&re, tests[i].pattern, tests[i].flags);
      if (n != 0)
	{
	  char buf[500];
	  regerror (n, &re, buf, sizeof (buf));
	  printf ("%s: regcomp %zd failed: %s\n", tests[i].pattern, i, buf);
	  ret = 1;
	  continue;
	}

      if (regexec (&re, tests[i].string, tests[i].nmatch, rm, 0))
	{
	  printf ("%s: regexec %zd failed\n", tests[i].pattern, i);
	  ret = 1;
	  regfree (&re);
	  continue;
	}

      for (n = 0; n < tests[i].nmatch; ++n)
	if (rm[n].rm_so != tests[i].rm[n].rm_so
              || rm[n].rm_eo != tests[i].rm[n].rm_eo)
	  {
	    if (tests[i].rm[n].rm_so == -1 && tests[i].rm[n].rm_eo == -1)
	      break;
	    printf ("%s: regexec %zd match failure rm[%d] %d..%d\n",
		    tests[i].pattern, i, n, rm[n].rm_so, rm[n].rm_eo);
	    ret = 1;
	    break;
	  }

      regfree (&re);
    }

  return ret;
}
