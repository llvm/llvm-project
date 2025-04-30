#include <stdlib.h>
#include <stdio.h>
#include "fnmatch.h"

struct {
  const char *name;
  const char *pattern;
  int flags;
  int expected;
} tests[] = {
  { "lib", "*LIB*", FNM_PERIOD, FNM_NOMATCH },
  { "lib", "*LIB*", FNM_CASEFOLD|FNM_PERIOD, 0 },
  { "a/b", "a[/]b", 0, 0 },
  { "a/b", "a[/]b", FNM_PATHNAME, FNM_NOMATCH },
  { "a/b", "[a-z]/[a-z]", 0, 0 },
  { "a/b", "*", FNM_PATHNAME, FNM_NOMATCH },
  { "a/b", "*[/]b", FNM_PATHNAME, FNM_NOMATCH },
  { "a/b", "*[b]", FNM_PATHNAME, FNM_NOMATCH },
  { "a/b", "[*]/b", 0, FNM_NOMATCH },
  { "*/b", "[*]/b", 0, 0 },
  { "a/b", "[?]/b", 0, FNM_NOMATCH },
  { "?/b", "[?]/b", 0, 0 },
  { "a/b", "[[a]/b", 0, 0 },
  { "[/b", "[[a]/b", 0, 0 },
  { "a/b", "\\*/b", 0, FNM_NOMATCH },
  { "*/b", "\\*/b", 0, 0 },
  { "a/b", "\\?/b", 0, FNM_NOMATCH },
  { "?/b", "\\?/b", 0, 0 },
  { "[/b", "[/b", 0, 0 },
  { "[/b", "\\[/b", 0, 0 },
  { "aa/b", "?""?/b", 0, 0 },
  { "aa/b", "?""?""?b", 0, 0 },
  { "aa/b", "?""?""?b", FNM_PATHNAME, FNM_NOMATCH },
  { ".a/b", "?a/b", FNM_PATHNAME|FNM_PERIOD, FNM_NOMATCH },
  { "a/.b", "a/?b", FNM_PATHNAME|FNM_PERIOD, FNM_NOMATCH },
  { ".a/b", "*a/b", FNM_PATHNAME|FNM_PERIOD, FNM_NOMATCH },
  { "a/.b", "a/*b", FNM_PATHNAME|FNM_PERIOD, FNM_NOMATCH },
  { ".a/b", "[.]a/b", FNM_PATHNAME|FNM_PERIOD, FNM_NOMATCH },
  { "a/.b", "a/[.]b", FNM_PATHNAME|FNM_PERIOD, FNM_NOMATCH },
  { "a/b", "*/?", FNM_PATHNAME|FNM_PERIOD, 0 },
  { "a/b", "?/*", FNM_PATHNAME|FNM_PERIOD, 0 },
  { ".a/b", ".*/?", FNM_PATHNAME|FNM_PERIOD, 0 },
  { "a/.b", "*/.?", FNM_PATHNAME|FNM_PERIOD, 0 },
  { "a/.b", "*/*", FNM_PATHNAME|FNM_PERIOD, FNM_NOMATCH },
  { "a/.b", "*?*/*", FNM_PERIOD, 0 },
  { "a./b", "*[.]/b", FNM_PATHNAME|FNM_PERIOD, 0 },
  { "a/b", "*[[:alpha:]]/*[[:alnum:]]", FNM_PATHNAME, 0 },
  { "a/b", "*[![:digit:]]*/[![:d-d]", FNM_PATHNAME, 0 },
  { "a/[", "*[![:digit:]]*/[[:d-d]", FNM_PATHNAME, 0 },
  { "a/[", "*[![:digit:]]*/[![:d-d]", FNM_PATHNAME, FNM_NOMATCH },
  { "a.b", "a?b", FNM_PATHNAME|FNM_PERIOD, 0 },
  { "a.b", "a*b", FNM_PATHNAME|FNM_PERIOD, 0 },
  { "a.b", "a[.]b", FNM_PATHNAME|FNM_PERIOD, 0 },
  { "a/b", "*a*", FNM_PATHNAME|FNM_LEADING_DIR, 0 },
  { "ab/c", "*a?", FNM_PATHNAME|FNM_LEADING_DIR, 0 },
  { "ab/c", "a?", FNM_PATHNAME|FNM_LEADING_DIR, 0 },
  { "a/b", "?*/?", FNM_PATHNAME, 0 },
  { "/b", "*/?", FNM_PATHNAME, 0 },
  { "/b", "**/?", FNM_PATHNAME, 0 },
};

int
main (void)
{
  size_t i;
  int errors = 0;

  for (i = 0; i < sizeof (tests) / sizeof (*tests); i++)
    {
      int match;

      match = fnmatch (tests[i].pattern, tests[i].name, tests[i].flags);

      printf ("[%2zd]  %s %s %s  -> %s\n", i, tests[i].pattern,
	      match == 0 ? "matches" : "does not match",
	      tests[i].name,
	      match != tests[i].expected ? "FAIL" : "OK");

      if (match != tests[i].expected)
	++errors ;
    }

  return errors != 0;
}
