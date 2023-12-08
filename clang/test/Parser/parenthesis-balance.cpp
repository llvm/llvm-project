// RUN: %clang_cc1 -fsyntax-only -verify %s

int f(int x) {
  if (int foo = f(bar)) {}     // expected-error{{use of undeclared identifier 'bar'}}
  while (int foo = f(bar)) {}  // expected-error{{use of undeclared identifier 'bar'}}
  for (int foo = f(bar);;) {}  // expected-error{{use of undeclared identifier 'bar'}}

  int bar;
  if (int foo = f(bar)) {}
  while (int foo = f(bar)) {}
  for (int foo = f(bar);;) {}

  return 0;
}

