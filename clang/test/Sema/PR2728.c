// RUN: %clang_cc1 -verify -fsyntax-only -std=c90 %s
// RUN: %clang_cc1 -verify -fsyntax-only -std=c99 %s
// expected-no-diagnostics

struct s
{
  int a;
};

int a[__builtin_offsetof(struct s, a) == 0];
