// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { };
A::A() { } // expected-error {{definition of implicitly declared default constructor}}

struct B { };
B::B(const B&) { } // expected-error {{definition of implicitly declared copy constructor}}

struct C { };
C& C::operator=(const C&) { return *this; } // expected-error {{definition of implicitly declared copy assignment operator}}

struct D { };
D::~D() { } // expected-error {{definition of implicitly declared destructor}}

// Make sure that the special member functions are introduced for
// name-lookup purposes and overload with user-declared
// constructors and assignment operators.
namespace PR6570 {
  class A { };

  class B {
  public:
    B() {}

    B(const A& a) {
      operator = (CONST);
      operator = (a);
    }

    B& operator = (const A& a) {
      return *this;
    }

    void f(const A &a) {
      B b(a);
    };

    static const B CONST;
  };

}

namespace PR7594 {
  // If the lazy declaration of special member functions is triggered
  // in an out-of-line initializer, make sure the functions aren't in
  // the initializer scope. This used to crash Clang:
  struct C {
    C();
    static C *c;
  };
  C *C::c = new C();
}
