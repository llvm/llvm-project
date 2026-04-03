// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -Wlifetime-safety -Wno-dangling -verify %s

// expected-no-diagnostics

void conditional_throw_branches(bool cond, int *value) {
  (void)(cond ? throw 1 : value);
  (void)(cond ? value : throw 1);
  (void)(cond ? throw 1 : throw 2);
}
