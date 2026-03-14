// RUN: %clangxx -O0 -g %s -o %t && %run %t
/// Check that REG_STARTEND is handled correctly.
/// This is a regression test for https://github.com/google/sanitizers/issues/1371
/// Previously, on GLibc systems, the interceptor was calling __compat_regexec
/// (regexec@GLIBC_2.2.5) insead of the newer __regexec (regexec@GLIBC_2.3.4).
/// The __compat_regexec strips the REG_STARTEND flag but does not report an error
/// if other flags are present. This can result in infinite loops for programs that
/// use REG_STARTEND to find all matches inside a buffer (since ignoring REG_STARTEND
/// means that the search always starts from the first character).

#include <assert.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// REG_STARTEND is a BSD extension not supported everywhere.
#ifdef REG_STARTEND
void test_matched(const regex_t *preg, const char *string, size_t start,
                  size_t end, const char *expected) {
  regmatch_t match[1];
  match[0].rm_so = start;
  match[0].rm_eo = end;
  int rv = regexec(preg, string, 1, match, REG_STARTEND);
  int matchlen = (int)(match[0].rm_eo - match[0].rm_so);
  const char *matchstart = string + match[0].rm_so;
  if (rv == 0) {
    if (expected == nullptr) {
      fprintf(stderr, "ERROR: expected no match but got '%.*s'\n",
              matchlen, matchstart);
      abort();
    } else if ((size_t)matchlen != strlen(expected) ||
               memcmp(matchstart, expected, strlen(expected)) != 0) {
      fprintf(stderr, "ERROR: expected '%s' match but got '%.*s'\n",
              expected, matchlen, matchstart);
      abort();
    }
  } else if (rv == REG_NOMATCH) {
    if (expected != nullptr) {
      fprintf(stderr, "ERROR: expected '%s' match but got no match\n", expected);
      abort();
    }
  } else {
    printf("ERROR: unexpected regexec return value %d\n", rv);
    abort();
  }
}

int main(void) {
  regex_t regex;
  int rv = regcomp(&regex, "[A-Z][A-Z]", 0);
  assert(!rv);
  test_matched(&regex, "ABCD", 0, 4, "AB");
  test_matched(&regex, "ABCD", 0, 1, nullptr); // Not long enough
  test_matched(&regex, "ABCD", 1, 4, "BC");
  test_matched(&regex, "ABCD", 1, 2, nullptr); // Not long enough
  test_matched(&regex, "ABCD", 2, 4, "CD");
  test_matched(&regex, "ABCD", 2, 3, nullptr); // Not long enough
  test_matched(&regex, "ABCD", 3, 4, nullptr); // Not long enough
  regfree(&regex);
  printf("Successful test\n");
  return 0;
}
#else
int main(void) {
  return 0;
}
#endif
