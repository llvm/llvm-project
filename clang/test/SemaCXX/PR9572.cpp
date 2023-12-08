// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

class Base {
  virtual ~Base();
#if __cplusplus <= 199711L
  // expected-note@-2 {{implicitly declared private here}}
#else
  // expected-note@-4 {{overridden virtual function is here}}
#endif
};

struct Foo : public Base {
#if __cplusplus <= 199711L
// expected-error@-2 {{base class 'Base' has private destructor}}
#else
// expected-error@-4 {{deleted function '~Foo' cannot override a non-deleted function}}
// expected-note@-5 3{{destructor of 'Foo' is implicitly deleted because base class 'Base' has an inaccessible destructor}}
#endif

  const int kBlah = 3;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{default member initializer for non-static data member is a C++11 extension}}
#endif

  Foo();
};

struct Bar : public Foo {
  Bar() { }
#if __cplusplus <= 199711L
  // expected-note@-2 {{implicit destructor for 'Foo' first required here}}
#else
  // expected-error@-4 {{attempt to use a deleted function}}
#endif
};

struct Baz {
  Foo f;
  Baz() { }
#if __cplusplus >= 201103L
  // expected-error@-2 {{attempt to use a deleted function}}
#endif
};
