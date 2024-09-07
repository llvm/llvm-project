// RUN: %clang_cc1 -fsyntax-only -verify %s

extern char test1[] __attribute__((used)); // expected-warning {{'used' attribute ignored on a non-definition declaration}}
extern const char test2[] __attribute__((used)); // expected-warning {{'used' attribute ignored on a non-definition declaration}}
extern const char test3[] __attribute__((used)) = "";
