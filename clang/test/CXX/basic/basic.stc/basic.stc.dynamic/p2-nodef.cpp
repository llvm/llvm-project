// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

int *use_new(int N) {
  return new int [N];
}

int std = 17;
