// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

struct S {
  static const int f0 = 0;
  static int f1;
};

int S::f1 = f0;
