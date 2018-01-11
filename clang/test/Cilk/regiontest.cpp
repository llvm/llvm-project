// RUN: %clang_cc1 -std=c++1z -verify %s
// expected-no-diagnostics

int bar();
int baz(int);

int foo(int n) {
  int x = _Cilk_spawn bar();
  _Cilk_for(int i = 0; i < n; ++i) {
    baz(i);
  }
  int y = _Cilk_spawn baz(n);
  _Cilk_sync;
  return x+y;
}
