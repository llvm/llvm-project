// RUN: %clang_cc1 -std=c++11 %s -verify

// expected-no-diagnostics

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
