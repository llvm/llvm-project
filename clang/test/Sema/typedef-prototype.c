// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef int unary_int_func(int arg);
unary_int_func add_one;

int add_one(int arg) {
  return arg + 1;
}
