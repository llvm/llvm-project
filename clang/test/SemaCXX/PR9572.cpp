// RUN: %clang_cc1 -fsyntax-only -verify %s
class Base {
  virtual ~Base(); // expected-note {{implicitly declared private here}}
};
struct Foo : public Base { // expected-error {{base class 'Base' has private destructor}}
  const int kBlah = 3; // expected-warning {{is a C++11 extension}}
  Foo();
};
struct Bar : public Foo {
  Bar() { } // expected-note {{implicit default destructor for 'Foo' first required here}}
};
struct Baz {
  Foo f;
  Baz() { }
};
