// RUN: cp %s %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -pedantic -Wall -fixit %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -pedantic -Wall -Werror %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -E -o - %t | FileCheck %s

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

int printf(char const *, ...);

void test() {
  typedef signed long int ssize_t;
  printf("%f", (ssize_t) 42);
}

// CHECK: printf("%zd", (ssize_t) 42);
