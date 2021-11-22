// RUN: %clang_cc1 -fsyntax-only -verify %s

template <typename T> class Foo {
  struct Base : T {};

  // Test that this code no longer causes a crash in Sema. rdar://23291875
  struct Derived : Base, T {};
};


template <typename T> struct Foo2 {
  struct Base1; // expected-note{{member is declared here}}
  struct Base2; // expected-note{{member is declared here}}
  // Should not crash on an incomplete-type and dependent base specifier.
  struct Derived : Base1, Base2 {}; // expected-error {{implicit instantiation of undefined member 'Foo2<int>::Base1'}} \
                                       expected-error {{implicit instantiation of undefined member 'Foo2<int>::Base2'}}
};

Foo2<int>::Derived a; // expected-note{{in instantiation of member class}}
