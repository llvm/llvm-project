// RUN: %clang_cc1 -fsyntax-only %s -verify -pedantic

int X[] = {
  [4]4,       // expected-warning {{use of GNU 'missing =' extension in designator}}
  [5] = 7
};

struct foo {
  int arr[10];
};

struct foo Y[10] = {
  [4] .arr [2] = 4,

  // This is not the GNU array init designator extension.
  [4] .arr [2] 4  // expected-error {{expected '=' or another designator}}
};
