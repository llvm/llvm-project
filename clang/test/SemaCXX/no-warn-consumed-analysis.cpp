// RUN: %clang_cc1 -fsyntax-only -verify -Wconsumed -fcxx-exceptions -std=c++11 %s
// expected-no-diagnostics

struct foo {
  ~foo();
};
struct bar : foo {};
struct baz : bar {};
baz foobar(baz a) { return a; }
