// RUN: %clang_cc1 -std=c++11 %s -verify

namespace PR15757 {
  struct S {
  };

  template<typename X, typename Y> struct T {
    template<typename A> T(X x, A &&a) {}

    template<typename A> explicit T(A &&a)
        noexcept(noexcept(T(X(), static_cast<A &&>(a))))
      : T(X(), static_cast<A &&>(a)) {}
  };

  template<typename X, typename Y> struct U : T<X, Y> {
    using T<X, Y>::T;
  };

  U<S, char> foo(char ch) { return U<S, char>(ch); }

  int main() {
    U<S, int> a(42);
    U<S, char> b('4');
    return 0;
  }
}

namespace WrongIdent {
  struct A {};
  struct B : A {};
  struct C : B {
    using B::A;
  };
}

namespace DefaultCtorConflict {
  struct A { A(int = 0); };
  struct B : A {
    using A::A;
  } b; // ok, not ambiguous, inherited constructor suppresses implicit default constructor
  struct C {
    B b;
  } c;
}

namespace InvalidConstruction {
  struct A { A(int); };
  struct B { B() = delete; };
  struct C : A, B { using A::A; };
  // Initialization here is performed as if by a defaulted default constructor,
  // which would be ill-formed (in the immediate context) in this case because
  // it would be defined as deleted.
  template<typename T> void f(decltype(T(0))*);
  template<typename T> int &f(...);
  int &r = f<C>(0);
}

namespace ExplicitConv {
  struct B {}; // expected-note 2{{candidate}}
  struct D : B { // expected-note 3{{candidate}}
    using B::B; // expected-note 2{{inherited}}
  };
  struct X { explicit operator B(); } x;
  struct Y { explicit operator D(); } y;

  D dx(x); // expected-error {{no matching constructor}}
  D dy(y);
}

namespace NestedListInit {
  struct B { B(); } b; // expected-note 5{{candidate}}
  struct D : B { // expected-note 3{{candidate}}
    using B::B; // expected-note 2{{inherited}}
  };
  // This is a bit weird. We're allowed one pair of braces for overload
  // resolution, and one more pair of braces due to [over.ics.list]/2.
  B b1 = {b};
  B b2 = {{b}};
  B b3 = {{{b}}}; // expected-error {{no match}}
  // This is the same, but we get one call to D's version of B::B(const B&)
  // before the two permitted calls to D::D(D&&).
  D d1 = {b};
  D d2 = {{b}};
  D d3 = {{{b}}};
  D d4 = {{{{b}}}}; // expected-error {{no match}}
}
