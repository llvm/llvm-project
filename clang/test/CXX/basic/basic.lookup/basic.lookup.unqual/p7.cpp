// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR5741
namespace test0 {
  struct A {
    struct B { };
    struct C;
  };

  struct A::C : B { };
}

// Test that successive base specifiers don't screw with each other.
namespace test1 {
  struct Opaque1 {};
  struct Opaque2 {};

  struct A {
    struct B { B(Opaque1); };
  };
  struct B {
    B(Opaque2);
  };

  struct C : A, B {
    // Apparently the base-or-member lookup is actually ambiguous
    // without this qualification.
    C() : A(), test1::B(Opaque2()) {}
  };
}

// Test that we don't find the injected class name when parsing base
// specifiers.
namespace test2 {
  template <class T> struct bar {};
  // expected-note@-1 {{template parameter is declared here}}
  template <class T> struct foo : bar<foo> {}; // expected-error {{use of class template 'foo' requires template arguments}} expected-note {{template is declared here}}
}
