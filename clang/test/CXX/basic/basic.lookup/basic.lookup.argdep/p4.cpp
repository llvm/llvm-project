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
                  // expected-error {{no viable conversion from 'B::B' to 'C'}}
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


// Check the rules described in p4:
//  When considering an associated namespace, the lookup is the same as the lookup
//  performed when the associated namespace is used as a qualifier (6.4.3.2) except that:

//  - Any using-directives in the associated namespace are ignored.
namespace test_using_directives {
  namespace M { struct S; }
  namespace N {
    void f(M::S); // expected-note {{declared here}}
  }
  namespace M {
    using namespace N;
    struct S {};
  }
  void test() {
    M::S s;
    f(s); // expected-error {{use of undeclared}}
    M::f(s); // ok
  }
}

//  - Any namespace-scope friend functions or friend function templates declared in
//    associated classes are visible within their respective namespaces even if
//    they are not visible during an ordinary lookup
// (Note: For the friend declaration to be visible, the corresponding class must be
//  included in the set of associated classes. Merely including the namespace in
//  the set of associated namespaces is not enough.)
namespace test_friend1 {
  namespace N {
    struct S;
    struct T {
      friend void f(S); // #1
    };
    struct S { S(); S(T); };
  }

  void test() {
    N::S s;
    N::T t;
    f(s); // expected-error {{use of undeclared}}
    f(t); // ok, #1
  }
}

// credit: Arthur O’Dwyer
namespace test_friend2 {
  struct A {
    struct B {
        struct C {};
    };
    friend void foo(...); // #1
  };

  struct D {
    friend void foo(...); // #2
  };
  template<class> struct E {
    struct F {};
  };

  template<class> struct G {};
  template<class> struct H {};
  template<class> struct I {};
  struct J { friend void foo(...) {} }; // #3

  void test() {
    A::B::C c;
    foo(c); // #1 is not visible since A is not an associated class
            // expected-error@-1 {{use of undeclared}}
    E<D>::F f;
    foo(f); // #2 is not visible since D is not an associated class
            // expected-error@-1 {{use of undeclared}}
    G<H<I<J> > > j;
    foo(j);  // ok, #3.
  }
}

//  - All names except those of (possibly overloaded) functions and
//    function templates are ignored.
namespace test_other_names {
  namespace N {
    struct S {};
    struct Callable { void operator()(S); };
    static struct Callable Callable;
  }

  void test() {
    N::S s;
    Callable(s); // expected-error {{use of undeclared}}
  }
}
