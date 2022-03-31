// RUN: cp %s %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c99 -pedantic -Wall -fixit %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c99 -fsyntax-only -pedantic -Wall -Werror %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -std=c99 -E -o - %t | FileCheck %s

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

int printf(char const *, ...);
int scanf(const char *, ...);

void test(void) {
  typedef signed long int ssize_t;
  printf("%f", (ssize_t) 42);
  ssize_t s;
  scanf("%f",  &s);
}

// CHECK: printf("%zd", (ssize_t) 42);
// CHECK: scanf("%zd", &s)
