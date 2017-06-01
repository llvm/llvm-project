// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace A {
  class A {
    friend void func(A);
    friend A operator+(A,A);
  };
}

namespace B {
  class B {
    static void func(B);
  };
  B operator+(B,B);
}

namespace D {
  class D {};
}

namespace C {
  class C {}; // expected-note {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'B::B' to 'const C::C &' for 1st argument}}
#if __cplusplus >= 201103L // C++11 or later
  // expected-note@-2 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'B::B' to 'C::C &&' for 1st argument}}
#endif
  void func(C); // expected-note {{'C::func' declared here}} \
                // expected-note {{passing argument to parameter here}}
  C operator+(C,C);
  D::D operator+(D::D,D::D);
}

namespace D {
  using namespace C;
}

namespace Test {
  void test() {
    func(A::A());
    // FIXME: namespace-aware typo correction causes an extra, misleading
    // message in this case; some form of backtracking, diagnostic message
    // delaying, or argument checking before emitting diagnostics is needed to
    // avoid accepting and printing out a typo correction that proves to be
    // incorrect once argument-dependent lookup resolution has occurred.
    func(B::B()); // expected-error {{use of undeclared identifier 'func'; did you mean 'C::func'?}} \
                  // expected-error {{no viable conversion from 'B::B' to 'C::C'}}
    func(C::C());
    A::A() + A::A();
    B::B() + B::B();
    C::C() + C::C();
    D::D() + D::D(); // expected-error {{invalid operands to binary expression ('D::D' and 'D::D')}}
  }
}

// PR6716
namespace test1 {
  template <class T> class A {
    template <class U> friend void foo(A &, U); // expected-note {{not viable: 1st argument ('const A<int>') would lose const qualifier}}

  public:
    A();
  };

  void test() {
    const A<int> a;
    foo(a, 10); // expected-error {{no matching function for call to 'foo'}}
  }
}
