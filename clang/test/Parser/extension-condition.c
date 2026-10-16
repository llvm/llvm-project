// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify -std=c17
// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify -std=c2y
// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify -x c++ -std=c++17

// expected-no-diagnostics

int f(int);

void cond(int a) {
  if (__extension__ ({ int r = f(a); r; })) {}
  while (__extension__ ({ int r = f(a); r; })) { break; }
  switch (__extension__ ({ int r = f(a); r; })) { default: break; }
  do {} while (__extension__ ({ int r = f(a); r; }));
  for (; __extension__ ({ int r = f(a); r; });) { break; }
}
