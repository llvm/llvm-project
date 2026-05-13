// RUN: %clang %s -o %t && %run %t 2>&1

// There's no interceptor for strcasestr on Windows
// XFAIL: target={{.*windows-msvc.*}}

// AIX does not define strcasestr
// UNSUPPORTED: target={{.*aix.*}}

#define _GNU_SOURCE
#include <assert.h>
#include <string.h>
int main(int argc, char **argv) {
  char *r = 0;
  char s1[] = "aB";
  char s2[] = "b";
  r = strcasestr(s1, s2);
  assert(r == s1 + 1);
  return 0;
}
