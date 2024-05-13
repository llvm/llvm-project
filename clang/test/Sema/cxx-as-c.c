// RUN: %clang_cc1 %s -verify

// PR36157
struct Foo {
  Foo(int n) : n_(n) {} // expected-error 1+{{}}
private:
  int n;
};
int main() { Foo f; } // expected-error 1+{{}}
