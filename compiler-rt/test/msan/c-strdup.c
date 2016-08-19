// RUN: %clang_msan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clang_msan -O1 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clang_msan -O2 %s -o %t && %run %t >%t.out 2>&1
// RUN: %clang_msan -O3 %s -o %t && %run %t >%t.out 2>&1

// XFAIL: target-is-mips64el

// Test that strdup in C programs is intercepted.
// GLibC headers translate strdup to __strdup at -O1 and higher.

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char buf[] = "abc";
  char *p = strdup(buf);
  if (*p)
    exit(0);
  return 0;
}
