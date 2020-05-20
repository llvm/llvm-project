// RUN: %clang_cc1 %s -fsyntax-only -frecovery-ast -verify -std=c++11

// NOTE: the test can be merged into existing tests once -frecovery-ast is on
// by default.

struct Foo { // expected-note {{candidate constructor (the implicit copy constructor) not viable}}
  Foo(int); // expected-note {{candidate constructor not viable}}
  ~Foo() = delete;
};

void test() {
  // we expect the "attempt to use a deleted function" diagnostic is suppressed.
  Foo foo; // expected-error {{no matching constructor for initialization of}}
}
