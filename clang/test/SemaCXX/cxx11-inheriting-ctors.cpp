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
  struct B {};
  struct D : B { // expected-note 3{{candidate}}
    using B::B;
  };
  struct X { explicit operator B(); } x;
  struct Y { explicit operator D(); } y;

  D dx(x); // expected-error {{no matching constructor}}
  D dy(y);
}

namespace NestedListInit {
  struct B { B(); } b; // expected-note 3{{candidate}}
  struct D : B { // expected-note 14{{not viable}}
    using B::B;
  };
  // This is a bit weird. We're allowed one pair of braces for overload
  // resolution, and one more pair of braces due to [over.ics.list]/2.
  B b1 = {b};
  B b2 = {{b}};
  B b3 = {{{b}}}; // expected-error {{no match}}
  // Per a proposed defect resolution, we don't get to call
  // D's version of B::B(const B&) here.
  D d0 = b; // expected-error {{no viable conversion}}
  D d1 = {b}; // expected-error {{no match}}
  D d2 = {{b}}; // expected-error {{no match}}
  D d3 = {{{b}}}; // expected-error {{no match}}
  D d4 = {{{{b}}}}; // expected-error {{no match}}
}

namespace PR31606 {
  // PR31606: as part of a proposed defect resolution, do not consider
  // inherited constructors that would be copy constructors for any class
  // between the declaring class and the constructed class (inclusive).
  struct Base {};

  struct A : Base {
    using Base::Base;
    bool operator==(A const &) const; // expected-note {{no known conversion from 'PR31606::B' to 'const PR31606::A' for 1st argument}}
  };

  struct B : Base {
    using Base::Base;
  };

  bool a = A{} == A{};
  // Note, we do *not* allow operator=='s argument to use the inherited A::A(Base&&) constructor to construct from B{}.
  bool b = A{} == B{}; // expected-error {{invalid operands}}
}

namespace implicit_member_srcloc {
  template<class T>
  struct S3 {
  };

  template<class T>
  struct S2 {
    S2(S3<T> &&);
  };

  template<class T>
  struct S1 : S2<T> {
    using S2<T>::S2;
    S1();
  };

  template<class T>
  struct S0 {
    S0();
    S0(S0&&) = default;
    S1<T> m1;
  };

  void foo1() {
    S0<int> s0;
  }
}

namespace PR47555 {
  struct A { constexpr A(int) {} };
  struct B : A { using A::A; };
  template<typename> void f() {
    constexpr B b = 0;
  };
  template void f<int>();
}
