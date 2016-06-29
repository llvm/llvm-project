// RUN: %clang_cc1 -std=c++11 -verify %s

namespace std_example {
  struct A { A(int); };
  struct B : A { using A::A; };

  struct C1 : B { using B::B; };
  struct C2 : B { using B::B; };

  struct D1 : C1, C2 {
    using C1::C1; // expected-note {{inherited from base class 'C1' here}}
    using C2::C2; // expected-note {{inherited from base class 'C2' here}}
  };

  struct V1 : virtual B { using B::B; };
  struct V2 : virtual B { using B::B; };

  struct D2 : V1, V2 {
    using V1::V1;
    using V2::V2;
  };

  D1 d1(0); // expected-error {{constructor of 'A' inherited from multiple base class subobjects}}
  D2 d2(0); // OK: initializes virtual B base class, which initializes the A base class
            // then initializes the V1 and V2 base classes as if by a defaulted default constructor

  struct M { M(); M(int); };
  struct N : M { using M::M; };
  struct O : M {};
  struct P : N, O { using N::N; using O::O; };
  P p(0); // OK: use M(0) to initialize N's base class,
          // use M() to initialize O's base class
}
