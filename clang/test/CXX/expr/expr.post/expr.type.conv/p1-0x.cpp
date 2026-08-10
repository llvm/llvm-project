// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

struct foo {
  foo();
  foo(int);
};

int func(foo& f) {
  decltype(foo())();
  f = (decltype(foo()))5;
  return decltype(3)(5);
}
