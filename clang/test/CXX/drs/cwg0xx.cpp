// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98,cxx98-14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,since-cxx11,cxx98-14,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx11,cxx98-14,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace cwg1 { // cwg1: no
  namespace X { extern "C" void cwg1_f(int a = 1); }
  namespace Y { extern "C" void cwg1_f(int a = 1); }
  using X::cwg1_f; using Y::cwg1_f;
  void g() {
    cwg1_f(0);
    // FIXME: This should be rejected, due to the ambiguous default argument.
    cwg1_f();
  }
  namespace X {
    using Y::cwg1_f;
    void h() {
      cwg1_f(0);
      // FIXME: This should be rejected, due to the ambiguous default argument.
      cwg1_f();
    }
  }

  namespace X {
    void z(int);
  }
  void X::z(int = 1) {} // #cwg1-z
  namespace X {
    void z(int = 1);
    // expected-error@-1 {{redefinition of default argument}}
    //   expected-note@#cwg1-z {{previous definition is here}}
  }

  void i(int = 1);
  void j() {
    void i(int = 1);
    using cwg1::i;
    i(0);
    // FIXME: This should be rejected, due to the ambiguous default argument.
    i();
  }
  void k() {
    using cwg1::i;
    void i(int = 1);
    i(0);
    // FIXME: This should be rejected, due to the ambiguous default argument.
    i();
  }
} // namespace cwg1

namespace cwg3 { // cwg3: 2.7
  template<typename T> struct A {};
  template<typename T> void f(T) { A<T> a; } // #cwg3-f-T
  template void f(int);
  template<> struct A<int> {};
  // expected-error@-1 {{explicit specialization of 'cwg3::A<int>' after instantiation}}
  //   expected-note@#cwg3-f-T {{implicit instantiation first required here}}
} // namespace cwg3

namespace cwg4 { // cwg4: 2.8
  extern "C" {
    static void cwg4_f(int) {}
    static void cwg4_f(float) {}
    void cwg4_g(int) {} // #cwg4-g-int
    void cwg4_g(float) {}
    // expected-error@-1 {{conflicting types for 'cwg4_g'}}
    //   expected-note@#cwg4-g-int {{previous definition is here}}
  }
} // namespace cwg4

namespace cwg5 { // cwg5: 3.1
  struct A {} a;
  struct B {
    B(const A&);
    B(const B&);
  };
  const volatile B b = a;

  struct C { C(C&); };
  struct D : C {};
  struct E { operator D&(); } e;
  const C c = e;
} // namespace cwg5

namespace cwg7 { // cwg7: 3.4
  class A { public: ~A(); };
  class B : virtual private A {}; // #cwg7-B
  class C : public B {} c; // #cwg7-C
  // expected-error@#cwg7-C {{inherited virtual base class 'A' has private destructor}}
  //   expected-note@#cwg7-C {{in implicit default constructor for 'cwg7::C' first required here}}
  //   expected-note@#cwg7-B {{declared private here}}
  // expected-error@#cwg7-C {{inherited virtual base class 'A' has private destructor}}
  //   expected-note@#cwg7-C {{in implicit destructor for 'cwg7::C' first required here}}
  //   expected-note@#cwg7-B {{declared private here}}
  class VeryDerivedC : public B, virtual public A {} vdc;

  class X { ~X(); }; // #cwg7-X
  class Y : X { ~Y() {} };
  // expected-error@-1 {{base class 'X' has private destructor}}
  //   expected-note@#cwg7-X {{implicitly declared private here}}

  namespace PR16370 { // This regressed the first time CWG7 was fixed.
    struct S1 { virtual ~S1(); };
    struct S2 : S1 {};
    struct S3 : S2 {};
    struct S4 : virtual S2 {};
    struct S5 : S3, S4 {
      S5();
      ~S5();
    };
    S5::S5() {}
  }
} // namespace cwg7

namespace cwg8 { // cwg8: dup 45
  class A {
    struct U;
    static const int k = 5;
    void f();
    template<typename, int, void (A::*)()> struct T;

    T<U, k, &A::f> *g();
  };
  A::T<A::U, A::k, &A::f> *A::g() { return 0; }
} // namespace cwg8

namespace cwg9 { // cwg9: 2.8
  struct B {
  protected:
    int m; // #cwg9-m
    friend int R1();
  };
  struct N : protected B { // #cwg9-N
    friend int R2();
  } n;
  int R1() { return n.m; }
  // expected-error@-1 {{'m' is a protected member of 'cwg9::B'}}
  //   expected-note@#cwg9-N {{constrained by protected inheritance here}}
  //   expected-note@#cwg9-m {{member is declared here}}
  int R2() { return n.m; }
} // namespace cwg9

namespace cwg10 { // cwg10: dup 45
  class A {
    struct B {
      A::B *p;
    };
  };
} // namespace cwg10

namespace cwg11 { // cwg11: 2.7
  template<typename T> struct A : T {
    using typename T::U;
    U u;
  };
  template<typename T> struct B : T {
    using T::V;
    V v;
    // expected-error@-1 {{unknown type name 'V'}}
  };
  struct X { typedef int U; };
  A<X> ax;
} // namespace cwg11

namespace cwg12 { // cwg12: sup 239
  enum E { e };
  E &f(E, E = e);
  void g() {
    int &f(int, E = e);
    // Under CWG12, these call two different functions.
    // Under CWG239, they call the same function.
    int &b = f(e);
    int &c = f(1);
  }
} // namespace cwg12

namespace cwg13 { // cwg13: no
  extern "C" void f(int);
  void g(char);

  template<typename T> struct A {
    A(void (*fp)(T));
  };
  template<typename T> int h(void (T));

  A<int> a1(f); // FIXME: We should reject this.
  A<char> a2(g);
  int a3 = h(f); // FIXME: We should reject this.
  int a4 = h(g);
} // namespace cwg13

namespace cwg14 { // cwg14: 3.4
  namespace X { extern "C" int cwg14_f(); }
  namespace Y { extern "C" int cwg14_f(); }
  using namespace X;
  using namespace Y;
  int k = cwg14_f();

  class C {
    int k;
    friend int Y::cwg14_f();
  } c;
  namespace Z {
    extern "C" int cwg14_f() { return c.k; }
  }

  namespace X { typedef int T; typedef int U; } // #cwg14-X-U
  namespace Y { typedef int T; typedef long U; } // #cwg14-Y-U
  T t; // ok, same type both times
  U u;
  // expected-error@-1 {{reference to 'U' is ambiguous}}
  //   expected-note@#cwg14-X-U {{candidate found by name lookup is 'cwg14::X::U'}}
  //   expected-note@#cwg14-Y-U {{candidate found by name lookup is 'cwg14::Y::U'}}
} // namespace cwg14

namespace cwg15 { // cwg15: 2.7
  template<typename T> void f(int); // #cwg15-f-decl-first
  template<typename T> void f(int = 0);
  // expected-error@-1 {{default arguments cannot be added to a function template that has already been declared}}
  //   expected-note@#cwg15-f-decl-first {{previous template declaration is here}}
} // namespace cwg15

namespace cwg16 { // cwg16: 2.8
  class A { // #cwg16-A
    void f(); // #cwg16-A-f-decl
    friend class C;
  };
  class B : A {}; // #cwg16-B
  class C : B {
    void g() {
      f();
      // expected-error@-1 {{'f' is a private member of 'cwg16::A'}}
      //   expected-note@#cwg16-B {{constrained by implicitly private inheritance here}}
      //   expected-note@#cwg16-A-f-decl {{member is declared here}}
      A::f(); // #cwg16-A-f-call
      // expected-error@#cwg16-A-f-call {{'A' is a private member of 'cwg16::A'}}
      //   expected-note@#cwg16-B {{constrained by implicitly private inheritance here}}
      //   expected-note@#cwg16-A {{member is declared here}}
      // expected-error@#cwg16-A-f-call {{cannot cast 'cwg16::C' to its private base class 'cwg16::A'}}
      //   expected-note@#cwg16-B {{implicitly declared private here}}
    }
  };
} // namespace cwg16

namespace cwg17 { // cwg17: 2.7
  class A {
    int n;
    int f();
    struct C;
  };
  struct B : A {} b;
  int A::f() { return b.n; }
  struct A::C : A {
    int g() { return n; }
  };
} // namespace cwg17

// cwg18: sup 577

namespace cwg19 { // cwg19: 3.1
  struct A {
    int n; // #cwg19-n
  };
  struct B : protected A { // #cwg19-B
  };
  struct C : B {} c;
  struct D : B {
    int get1() { return c.n; }
    // expected-error@-1 {{'n' is a protected member of 'cwg19::A'}}
    //   expected-note@#cwg19-B {{constrained by protected inheritance here}}
    //   expected-note@#cwg19-n {{member is declared here}}
    int get2() { return ((A&)c).n; } // ok, A is an accessible base of B from here
  };
} // namespace cwg19

namespace cwg20 { // cwg20: 2.8
  class X {
  public:
    X();
  private:
    X(const X&); // #cwg20-X-ctor
  };
  X &f();
  X x = f();
  // expected-error@-1 {{calling a private constructor of class 'cwg20::X'}}
  //   expected-note@#cwg20-X-ctor {{declared private here}}
} // namespace cwg20

namespace cwg21 { // cwg21: 3.4
  template<typename T> struct A;
  struct X {
    template<typename T = int> friend struct A;
    // expected-error@-1 {{default template argument not permitted on a friend template}}
    template<typename T = int> friend struct B;
    // expected-error@-1 {{default template argument not permitted on a friend template}}
  };
} // namespace cwg21

namespace cwg22 { // cwg22: sup 481
  template<typename cwg22_T = cwg22_T> struct X;
  // expected-error@-1 {{unknown type name 'cwg22_T'}}
  typedef int T;
  template<typename T = T> struct Y;
} // namespace cwg22

namespace cwg23 { // cwg23: 2.7
  template<typename T> void f(T, T); // #cwg23-f-T-T
  template<typename T> void f(T, int); // #cwg23-f-T-int
  void g() { f(0, 0); }
  // expected-error@-1 {{call to 'f' is ambiguous}}
  //   expected-note@#cwg23-f-T-T {{candidate function [with T = int]}}
  //   expected-note@#cwg23-f-T-int {{candidate function [with T = int]}}
} // namespace cwg23

// cwg24: na

namespace cwg25 { // cwg25: 4
  struct A {
    void f() throw(int);
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  };
  void (A::*f)() throw (int);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (A::*g)() throw () = f;
  // cxx98-14-error@-1 {{target exception specification is not superset of source}}
  // since-cxx17-error@-2 {{different exception specifications}}
  void (A::*g2)() throw () = 0;
  void (A::*h)() throw (int, char) = f;
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (A::*i)() throw () = &A::f;
  // cxx98-14-error@-1 {{target exception specification is not superset of source}}
  // since-cxx17-error@-2 {{different exception specifications}}
  void (A::*i2)() throw () = 0;
  void (A::*j)() throw (int, char) = &A::f;
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void x() {
    g2 = f;
    // cxx98-14-error@-1 {{target exception specification is not superset of source}}
    // since-cxx17-error@-2 {{different exception specifications}}
    h = f;
    i2 = &A::f;
    // cxx98-14-error@-1 {{target exception specification is not superset of source}}
    // since-cxx17-error@-2 {{different exception specifications}}
    j = &A::f;
  }
} // namespace cwg25

namespace cwg26 { // cwg26: 2.7
  struct A { A(A, const A & = A()); };
  // expected-error@-1 {{copy constructor must pass its first argument by reference}}
  struct B {
    B();
    // FIXME: In C++98, we diagnose this twice.
    B(const B &, B = B());
    // cxx98-14-error@-1 {{recursive evaluation of default argument}}
    //   cxx98-14-note@-2 {{default argument used here}}
    // cxx98-error@-3 {{recursive evaluation of default argument}}
    //   cxx98-note@-4 {{default argument used here}}
  };
  struct C {
    static C &f();
    C(const C &, C = f());
    // expected-error@-1 {{recursive evaluation of default argument}}
    //   expected-note@-2 {{default argument used here}}
  };
} // namespace cwg26

namespace cwg27 { // cwg27: 2.7
  enum E { e } n;
  E &m = true ? n : n;
} // namespace cwg27

// cwg28: na lib

namespace cwg29 { // cwg29: 3.4
  void cwg29_f0(); // #cwg29-f0
  void g0() { void cwg29_f0(); }
  extern "C++" void g0_cxx() { void cwg29_f0(); }
  extern "C" void g0_c() { void cwg29_f0(); }
  // expected-error@-1 {{declaration of 'cwg29_f0' has a different language linkage}}
  //   expected-note@#cwg29-f0 {{previous declaration is here}}

  extern "C" void cwg29_f1(); // #cwg29-f1
  void g1() { void cwg29_f1(); }
  extern "C" void g1_c() { void cwg29_f1(); }
  extern "C++" void g1_cxx() { void cwg29_f1(); }
  // expected-error@-1 {{declaration of 'cwg29_f1' has a different language linkage}}
  //   expected-note@#cwg29-f1 {{previous declaration is here}}

  void g2() { void cwg29_f2(); } // #cwg29-f2
  extern "C" void cwg29_f2();
  // expected-error@-1 {{declaration of 'cwg29_f2' has a different language linkage}}
  //   expected-note@#cwg29-f2 {{previous declaration is here}}

  extern "C" void g3() { void cwg29_f3(); } // #cwg29-f3
  extern "C++" void cwg29_f3();
  // expected-error@-1 {{declaration of 'cwg29_f3' has a different language linkage}}
  //   expected-note@#cwg29-f3 {{previous declaration is here}}

  extern "C++" void g4() { void cwg29_f4(); } // #cwg29-f4
  extern "C" void cwg29_f4();
  // expected-error@-1 {{declaration of 'cwg29_f4' has a different language linkage}}
  //   expected-note@#cwg29-f4 {{previous declaration is here}}

  extern "C" void g5();
  extern "C++" void cwg29_f5();
  void g5() {
    void cwg29_f5(); // ok, g5 is extern "C" but we're not inside the linkage-specification here.
  }

  extern "C++" void g6();
  extern "C" void cwg29_f6();
  void g6() {
    void cwg29_f6(); // ok, g6 is extern "C" but we're not inside the linkage-specification here.
  }

  extern "C" void g7();
  extern "C++" void cwg29_f7(); // #cwg29-f7
  extern "C" void g7() {
    void cwg29_f7();
    // expected-error@-1 {{declaration of 'cwg29_f7' has a different language linkage}}
    //   expected-note@#cwg29-f7 {{previous declaration is here}}
  }

  extern "C++" void g8();
  extern "C" void cwg29_f8(); // #cwg29-f8
  extern "C++" void g8() {
    void cwg29_f8();
    // expected-error@-1 {{declaration of 'cwg29_f8' has a different language linkage}}
    //   expected-note@#cwg29-f8 {{previous declaration is here}}
  }
} // namespace cwg29

namespace cwg30 { // cwg30: sup 468 c++11
  struct A {
    template<int> static int f();
  } a, *p = &a;
  // FIXME: It's not clear whether CWG468 applies to C++98 too.
  int x = A::template f<0>();
  // cxx98-error@-1 {{'template' keyword outside of a template}}
  int y = a.template f<0>();
  // cxx98-error@-1 {{'template' keyword outside of a template}}
  int z = p->template f<0>();
  // cxx98-error@-1 {{'template' keyword outside of a template}}
} // namespace cwg30

namespace cwg31 { // cwg31: 2.8
  class X {
  private:
    void operator delete(void*); // #cwg31-delete
  };
  // We would call X::operator delete if X() threw (even though it can't,
  // and even though we allocated the X using ::operator delete).
  X *p = new X;
  // expected-error@-1 {{'operator delete' is a private member of 'cwg31::X'}}
  //   expected-note@#cwg31-delete {{declared private here}}
} // namespace cwg31

// cwg32: na

namespace cwg33 { // cwg33: 9
  namespace X { struct S; void f(void (*)(S)); } // #cwg33-f-S
  namespace Y { struct T; void f(void (*)(T)); } // #cwg33-f-T
  void g(X::S);
  template<typename Z> Z g(Y::T);
  void h() { f(&g); }
  // expected-error@-1 {{call to 'f' is ambiguous}}
  //   expected-note@#cwg33-f-S {{candidate function}}
  //   expected-note@#cwg33-f-T {{candidate function}}

  template<typename T> void t(X::S);
  template<typename T, typename U = void> void u(X::S);
  // expected-error@-1 0-1 {{default template arguments for a function template are a C++11 extension}}
  void templ() { f(t<int>); f(u<int>); }

  // Even though v<int> cannot select the first overload, ADL considers it
  // and adds namespace Z to the set of associated namespaces, and then picks
  // Z::f even though that function has nothing to do with any associated type.
  namespace Z { struct Q; void f(void(*)()); }
  template<int> Z::Q v();
  template<typename> void v();
  void unrelated_templ() { f(v<int>); }

  namespace dependent {
    struct X {};
    template<class T> struct Y {
      friend int operator+(X, void(*)(Y)) {}
    };

    template<typename T> void f(Y<T>);
    int use = X() + f<int>;
    // expected-error@-1 {{invalid operands to binary expression ('X' and 'void (Y<int>)')}}
  }

  namespace member {
    struct Q {};
    struct Y { friend int operator+(Q, Y (*)()); };
    struct X { template<typename> static Y f(); };
    int m = Q() + X().f<int>; // ok
    int n = Q() + (&(X().f<int>)); // ok
  }
} // namespace cwg33

// cwg34: na
// cwg35: dup 178

namespace cwg36 { // cwg36: 2.8
namespace example1 {
  namespace A {
    int i;
  }

  namespace A1 {
    using A::i;
    using A::i;
  }

  void f()
  {
    using A::i;
    using A::i;
  }
}

namespace example2 {
  struct A
  {
    int i;
    static int j;
  };

  struct B : A { };
  struct C : A { };

  struct D : virtual B, virtual C
  {
    using B::i; // #cwg36-ex2-B-i-first
    using B::i;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-ex2-B-i-first {{previous using declaration}}

    using C::i; // #cwg36-ex2-C-i-first
    using C::i;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-ex2-C-i-first {{previous using declaration}}

    using B::j; // #cwg36-ex2-B-j-first
    using B::j;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-ex2-B-j-first {{previous using declaration}}

    using C::j; // #cwg36-ex2-C-j-first
    using C::j;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-ex2-C-j-first {{previous using declaration}}
  };
}

namespace example3 {
  template<typename T>
  struct A
  {
    T i;
    static T j;
  };

  template<typename T>
  struct B : A<T> { };
  template<typename T>
  struct C : A<T> { };

  template<typename T>
  struct D : virtual B<T>, virtual C<T>
  {
    using B<T>::i; // #cwg36-ex3-B-i-first
    using B<T>::i;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-ex3-B-i-first {{previous using declaration}}

    using C<T>::i; // #cwg36-ex3-C-i-first
    using C<T>::i;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-ex3-C-i-first {{previous using declaration}}

    using B<T>::j; // #cwg36-ex3-B-j-first
    using B<T>::j;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-ex3-B-j-first {{previous using declaration}}

    using C<T>::j; // #cwg36-ex3-C-j-first
    using C<T>::j;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-ex3-C-j-first {{previous using declaration}}
  };
}
namespace example4 {
  template<typename T>
  struct E {
    T k;
  };

  template<typename T>
  struct G : E<T> {
    using E<T>::k; // #cwg36-E-k-first
    using E<T>::k;
    // expected-error@-1 {{redeclaration of using declaration}}
    //   expected-note@#cwg36-E-k-first {{previous using declaration}}
  };
}
} // namespace cwg36

// cwg37: sup 475

namespace cwg38 { // cwg38: 2.7
  template<typename T> struct X {};
  template<typename T> X<T> operator+(X<T> a, X<T> b) { return a; }
  template X<int> operator+<int>(X<int>, X<int>);
} // namespace cwg38

namespace cwg39 { // cwg39: no
  namespace example1 {
    struct A { int &f(int); };
    struct B : A {
      using A::f;
      float &f(float);
    } b;
    int &r = b.f(0);
  }

  namespace example2 {
    struct A {
      int &x(int); // #cwg39-A-x-decl
      static int &y(int); // #cwg39-A-y-decl
    };
    struct V {
      int &z(int);
    };
    struct B : A, virtual V {
      using A::x; // #cwg39-using-A-x
      float &x(float);
      using A::y; // #cwg39-using-A-y
      static float &y(float);
      using V::z;
      float &z(float);
    };
    struct C : A, B, virtual V {} c;
    /* expected-warning@-1
    {{direct base 'A' is inaccessible due to ambiguity:
    struct cwg39::example2::C -> A
    struct cwg39::example2::C -> B -> A}} */
    int &x = c.x(0);
    // expected-error@-1 {{member 'x' found in multiple base classes of different types}}
    //   expected-note@#cwg39-A-x-decl {{member found by ambiguous name lookup}}
    //   expected-note@#cwg39-using-A-x {{member found by ambiguous name lookup}}

    // FIXME: This is valid, because we find the same static data member either way.
    int &y = c.y(0);
    // expected-error@-1 {{member 'y' found in multiple base classes of different types}}
    //   expected-note@#cwg39-A-y-decl {{member found by ambiguous name lookup}}
    //   expected-note@#cwg39-using-A-y {{member found by ambiguous name lookup}}
    int &z = c.z(0);
  }

  namespace example3 {
    struct A { static int f(); };
    struct B : virtual A { using A::f; };
    struct C : virtual A { using A::f; };
    struct D : B, C {} d;
    int k = d.f();
  }

  namespace example4 {
    struct A { int n; }; // #cwg39-ex4-A-n
    struct B : A {};
    struct C : A {};
    struct D : B, C { int f() { return n; } };
    /* expected-error@-1
    {{non-static member 'n' found in multiple base-class subobjects of type 'A':
    struct cwg39::example4::D -> B -> A
    struct cwg39::example4::D -> C -> A}} */
    //   expected-note@#cwg39-ex4-A-n {{member found by ambiguous name lookup}}
  }

  namespace PR5916 {
    // FIXME: This is valid.
    struct A { int n; }; // #cwg39-A-n
    struct B : A {};
    struct C : A {};
    struct D : B, C {};
    int k = sizeof(D::n); // #cwg39-sizeof
    /* expected-error@#cwg39-sizeof
    {{non-static member 'n' found in multiple base-class subobjects of type 'A':
    struct cwg39::PR5916::D -> B -> A
    struct cwg39::PR5916::D -> C -> A}} */
    //   expected-note@#cwg39-A-n {{member found by ambiguous name lookup}}

    // expected-error@#cwg39-sizeof {{unknown type name}}
#if __cplusplus >= 201103L
    decltype(D::n) n;
    /* since-cxx11-error@-1
    {{non-static member 'n' found in multiple base-class subobjects of type 'A':
    struct cwg39::PR5916::D -> B -> A
    struct cwg39::PR5916::D -> C -> A}} */
    //   since-cxx11-note@#cwg39-A-n {{member found by ambiguous name lookup}}
#endif
  }
} // namespace cwg39

// cwg40: na

namespace cwg41 { // cwg41: 2.7
  struct S f(S);
} // namespace cwg41

namespace cwg42 { // cwg42: 2.7
  struct A { static const int k = 0; };
  struct B : A { static const int k = A::k; };
} // namespace cwg42

// cwg43: na

namespace cwg44 { // cwg44: sup 727
  struct A {
    template<int> void f();
    template<> void f<0>();
  };
} // namespace cwg44

namespace cwg45 { // cwg45: 2.7
  class A {
    class B {};
    class C : B {};
    C c;
  };
} // namespace cwg45

namespace cwg46 { // cwg46: 2.7
  template<typename> struct A { template<typename> struct B {}; };
  template template struct A<int>::B<int>;
  // expected-error@-1 {{expected unqualified-id}}
} // namespace cwg46

namespace cwg47 { // cwg47: sup 329
  template<typename T> struct A {
    friend void f() { T t; } // #cwg47-f
    // expected-error@-1 {{redefinition of 'f'}}
    //   expected-note@#cwg47-b {{in instantiation of template class 'cwg47::A<float>' requested here}}
    //   expected-note@#cwg47-f {{previous definition is here}}
  };
  A<int> a;
  A<float> b; // #cwg47-b

  void f();
  void g() { f(); }
} // namespace cwg47

namespace cwg48 { // cwg48: 2.7
  namespace {
    struct S {
      static const int m = 0;
      static const int n = 0;
      static const int o = 0;
    };
  }
  int a = S::m;
  // FIXME: We should produce a 'has internal linkage but is not defined'
  // diagnostic for 'S::n'.
  const int &b = S::n;
  const int S::o;
  const int &c = S::o;
} // namespace cwg48

namespace cwg49 { // cwg49: 2.8
  template<int*> struct A {}; // #cwg49-A
  int k;
#if __has_feature(cxx_constexpr)
  constexpr
#endif
  int *const p = &k; // #cwg49-p
  A<&k> a;
  A<p> b; // #cwg49-b
  // cxx98-error@#cwg49-b {{non-type template argument referring to object 'p' with internal linkage is a C++11 extension}}
  //   cxx98-note@#cwg49-p {{non-type template argument refers to object here}}
  // cxx98-14-error@#cwg49-b {{non-type template argument for template parameter of pointer type 'int *' must have its address taken}}
  //   cxx98-14-note@#cwg49-A {{template parameter is declared here}}
  int *q = &k; // #cwg49-q
  A<q> c; // #cwg49-c
  // cxx98-error@#cwg49-c {{non-type template argument for template parameter of pointer type 'int *' must have its address taken}}
  //   cxx98-note@#cwg49-A {{template parameter is declared here}}
  // cxx11-14-error@#cwg49-c {{non-type template argument of type 'int *' is not a constant expression}}
  //   cxx11-14-note@#cwg49-c {{read of non-constexpr variable 'q' is not allowed in a constant expression}}
  //   cxx11-14-note@#cwg49-q {{declared here}}
  //   cxx11-14-note@#cwg49-A {{template parameter is declared here}}
  // since-cxx17-error@#cwg49-c {{non-type template argument is not a constant expression}}
  //   since-cxx17-note@#cwg49-c {{read of non-constexpr variable 'q' is not allowed in a constant expression}}
  //   since-cxx17-note@#cwg49-q {{declared here}}
} // namespace cwg49

namespace cwg50 { // cwg50: 2.7
  struct X; // #cwg50-X
  extern X *p;
  X *q = (X*)p;
  X *r = static_cast<X*>(p);
  X *s = const_cast<X*>(p);
  X *t = reinterpret_cast<X*>(p);
  X *u = dynamic_cast<X*>(p);
  // expected-error@-1 {{'cwg50::X' is an incomplete type}}
  //   expected-note@#cwg50-X {{forward declaration of 'cwg50::X'}}
} // namespace cwg50

namespace cwg51 { // cwg51: 2.8
  struct A {};
  struct B : A {};
  struct S {
    operator A&();
    operator B&();
  } s;
  A &a = s;
} // namespace cwg51

namespace cwg52 { // cwg52: 2.8
  struct A { int n; }; // #cwg52-A
  struct B : private A {} b; // #cwg52-B
  int k = b.A::n; // #cwg52-k
  // FIXME: This first diagnostic is very strangely worded, and seems to be bogus.
  // expected-error@#cwg52-k {{'A' is a private member of 'cwg52::A'}}
  //   expected-note@#cwg52-B {{constrained by private inheritance here}}
  //   expected-note@#cwg52-A {{member is declared here}}
  // expected-error@#cwg52-k {{cannot cast 'struct B' to its private base class 'cwg52::A'}}
  //   expected-note@#cwg52-B {{declared private here}}
} // namespace cwg52

namespace cwg53 { // cwg53: 2.7
  int n = 0;
  enum E { e } x = static_cast<E>(n);
} // namespace cwg53

namespace cwg54 { // cwg54: 2.8
  struct A { int a; } a;
  struct V { int v; } v;
  struct B : private A, virtual V { int b; } b; // #cwg54-B

  A &sab = static_cast<A&>(b);
  // expected-error@-1 {{cannot cast 'struct B' to its private base class 'A'}}
  //   expected-note@#cwg54-B {{declared private here}}
  A *spab = static_cast<A*>(&b);
  // expected-error@-1 {{cannot cast 'struct B' to its private base class 'A'}}
  //   expected-note@#cwg54-B {{declared private here}}
  int A::*smab = static_cast<int A::*>(&B::b);
  // expected-error@-1 {{cannot cast 'cwg54::B' to its private base class 'A'}}
  //   expected-note@#cwg54-B {{declared private here}}
  B &sba = static_cast<B&>(a);
  // expected-error@-1 {{cannot cast private base class 'cwg54::A' to 'cwg54::B'}}
  //   expected-note@#cwg54-B {{declared private here}}
  B *spba = static_cast<B*>(&a);
  // expected-error@-1 {{cannot cast private base class 'cwg54::A' to 'cwg54::B'}}
  //   expected-note@#cwg54-B {{declared private here}}
  int B::*smba = static_cast<int B::*>(&A::a);
  // expected-error@-1 {{cannot cast private base class 'A' to 'B'}}
  //   expected-note@#cwg54-B {{declared private here}}

  V &svb = static_cast<V&>(b);
  V *spvb = static_cast<V*>(&b);
  int V::*smvb = static_cast<int V::*>(&B::b);
  // expected-error@-1 {{conversion from pointer to member of class 'cwg54::B' to pointer to member of class 'V' via virtual base 'cwg54::V' is not allowed}}
  B &sbv = static_cast<B&>(v);
  // expected-error@-1 {{cannot cast 'struct V' to 'B &' via virtual base 'cwg54::V'}}
  B *spbv = static_cast<B*>(&v);
  // expected-error@-1 {{cannot cast 'cwg54::V *' to 'B *' via virtual base 'cwg54::V'}}
  int B::*smbv = static_cast<int B::*>(&V::v);
  // expected-error@-1 {{conversion from pointer to member of class 'V' to pointer to member of class 'B' via virtual base 'cwg54::V' is not allowed}}

  A &cab = (A&)(b);
  A *cpab = (A*)(&b);
  int A::*cmab = (int A::*)(&B::b);
  B &cba = (B&)(a);
  B *cpba = (B*)(&a);
  int B::*cmba = (int B::*)(&A::a);

  V &cvb = (V&)(b);
  V *cpvb = (V*)(&b);
  int V::*cmvb = (int V::*)(&B::b);
  // expected-error@-1 {{conversion from pointer to member of class 'cwg54::B' to pointer to member of class 'V' via virtual base 'cwg54::V' is not allowed}}
  B &cbv = (B&)(v);
  // expected-error@-1 {{cannot cast 'struct V' to 'B &' via virtual base 'cwg54::V'}}
  B *cpbv = (B*)(&v);
  // expected-error@-1 {{cannot cast 'cwg54::V *' to 'B *' via virtual base 'cwg54::V'}}
  int B::*cmbv = (int B::*)(&V::v);
  // expected-error@-1 {{conversion from pointer to member of class 'V' to pointer to member of class 'B' via virtual base 'cwg54::V' is not allowed}}
} // namespace cwg54

namespace cwg55 { // cwg55: 2.7
  enum E { e = 5 };
  static_assert(e + 1 == 6, "");
} // namespace cwg55

namespace cwg56 { // cwg56: 2.7
  struct A {
    typedef int T; // #cwg56-typedef-int-T-first
    typedef int T;
    // expected-error@-1 {{redefinition of 'T'}}
    //   expected-note@#cwg56-typedef-int-T-first {{previous definition is here}}
  };
  struct B {
    struct X;
    typedef X X; // #cwg56-typedef-X-X-first
    typedef X X;
    // expected-error@-1 {{redefinition of 'X'}}
    //   expected-note@#cwg56-typedef-X-X-first {{previous definition is here}}
  };
} // namespace cwg56

namespace cwg58 { // cwg58: 3.1
  // FIXME: Ideally, we should have a CodeGen test for this.
#if __cplusplus >= 201103L
  enum E1 { E1_0 = 0, E1_1 = 1 };
  enum E2 { E2_0 = 0, E2_m1 = -1 };
  struct X { E1 e1 : 1; E2 e2 : 1; };
  static_assert(X{E1_1, E2_m1}.e1 == 1, "");
  static_assert(X{E1_1, E2_m1}.e2 == -1, "");
#endif
} // namespace cwg58

namespace cwg59 { // cwg59: 2.7
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-volatile"
  template<typename T> struct convert_to { operator T() const; };
  struct A {}; // #cwg59-A
  struct B : A {}; // #cwg59-B

  A a1 = convert_to<A>();
  A a2 = convert_to<A&>();
  A a3 = convert_to<const A>();
  A a4 = convert_to<const volatile A>();
  // cxx98-14-error@-1 {{no viable constructor copying variable of type 'const volatile cwg59::A'}}
  //   cxx98-14-note@#cwg59-A {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const volatile cwg59::A') would lose volatile qualifier}}
  //   cxx11-14-note@#cwg59-A {{candidate constructor (the implicit move constructor) not viable: 1st argument ('const volatile cwg59::A') would lose const and volatile qualifiers}}
  //   cxx98-14-note@#cwg59-A {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  A a5 = convert_to<const volatile A&>();
  // expected-error@-1 {{no viable constructor copying variable of type 'const volatile cwg59::A'}}
  //   expected-note@#cwg59-A {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const volatile cwg59::A') would lose volatile qualifier}}
  //   since-cxx11-note@#cwg59-A {{candidate constructor (the implicit move constructor) not viable: 1st argument ('const volatile cwg59::A') would lose const and volatile qualifiers}}
  //   expected-note@#cwg59-A {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}

  B b1 = convert_to<B>();
  B b2 = convert_to<B&>();
  B b3 = convert_to<const B>();
  B b4 = convert_to<const volatile B>();
  // cxx98-14-error@-1 {{no viable constructor copying variable of type 'const volatile cwg59::B'}}
  //   cxx98-14-note@#cwg59-B {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const volatile cwg59::B') would lose volatile qualifier}}
  //   cxx11-14-note@#cwg59-B {{candidate constructor (the implicit move constructor) not viable: 1st argument ('const volatile cwg59::B') would lose const and volatile qualifiers}}
  //   cxx98-14-note@#cwg59-B {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  B b5 = convert_to<const volatile B&>();
  // expected-error@-1 {{no viable constructor copying variable of type 'const volatile cwg59::B'}}
  //   expected-note@#cwg59-B {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const volatile cwg59::B') would lose volatile qualifier}}
  //   since-cxx11-note@#cwg59-B {{candidate constructor (the implicit move constructor) not viable: 1st argument ('const volatile cwg59::B') would lose const and volatile qualifiers}}
  //   expected-note@#cwg59-B {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}

  A c1 = convert_to<B>();
  A c2 = convert_to<B&>();
  A c3 = convert_to<const B>();
  A c4 = convert_to<const volatile B>();
  // expected-error@-1 {{no viable constructor copying variable of type 'const volatile cwg59::B'}}
  //   expected-note@#cwg59-A {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'const volatile cwg59::B' to 'const A &' for 1st argument}}
  //   since-cxx11-note@#cwg59-A {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'const volatile cwg59::B' to 'A &&' for 1st argument}}
  //   expected-note@#cwg59-A {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  A c5 = convert_to<const volatile B&>();
  // expected-error@-1 {{no viable constructor copying variable of type 'const volatile cwg59::B'}}
  //   expected-note@#cwg59-A {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'const volatile cwg59::B' to 'const A &' for 1st argument}}
  //   since-cxx11-note@#cwg59-A {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'const volatile cwg59::B' to 'A &&' for 1st argument}}
  //   expected-note@#cwg59-A {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}

  int n1 = convert_to<int>();
  int n2 = convert_to<int&>();
  int n3 = convert_to<const int>();
  int n4 = convert_to<const volatile int>();
  int n5 = convert_to<const volatile int&>();
#pragma clang diagnostic pop
} // namespace cwg59

namespace cwg60 { // cwg60: 2.7
  void f(int &);
  int &f(...);
  const int k = 0;
  int &n = f(k);
} // namespace cwg60

namespace cwg61 { // cwg61: 3.4
  struct X {
    static void f();
  } x;
  struct Y {
    static void f();
    static void f(int);
  } y;
  // This is (presumably) valid, because x.f does not refer to an overloaded
  // function name.
  void (*p)() = &x.f;
  void (*q)() = &y.f;
  // expected-error@-1 {{cannot create a non-constant pointer to member function}}
  void (*r)() = y.f;
  // expected-error@-1 {{cannot create a non-constant pointer to member function}}
} // namespace cwg61

namespace cwg62 { // cwg62: 2.9
  struct A {
    struct { int n; } b;
  };
  template<typename T> struct X {};
  template<typename T> T get() { return get<T>(); }
  template<typename T> int take(T) { return 0; }

  X<A> x1;
  A a = get<A>();

  typedef struct { } *NoNameForLinkagePtr; // #cwg62-unnamed
  NoNameForLinkagePtr noNameForLinkagePtr;

  struct Danger {
    NoNameForLinkagePtr p;
  };

  X<NoNameForLinkagePtr> x2;
  // cxx98-error@-1 {{template argument uses unnamed type}}
  //   cxx98-note@#cwg62-unnamed {{unnamed type used in template argument was declared here}}
  X<const NoNameForLinkagePtr> x3;
  // cxx98-error@-1 {{template argument uses unnamed type}}
  //   cxx98-note@#cwg62-unnamed {{unnamed type used in template argument was declared here}}
  NoNameForLinkagePtr p1 = get<NoNameForLinkagePtr>();
  // cxx98-error@-1 {{template argument uses unnamed type}}
  //   cxx98-note@#cwg62-unnamed {{unnamed type used in template argument was declared here}}
  //   cxx98-note@-3 {{while substituting explicitly-specified template arguments}}
  NoNameForLinkagePtr p2 = get<const NoNameForLinkagePtr>();
  // cxx98-error@-1 {{template argument uses unnamed type}}
  //   cxx98-note@#cwg62-unnamed {{unnamed type used in template argument was declared here}}
  //   cxx98-note@-3 {{while substituting explicitly-specified template arguments}}
  int n1 = take(noNameForLinkagePtr);
  // cxx98-error@-1 {{template argument uses unnamed type}}
  //   cxx98-note@#cwg62-unnamed {{unnamed type used in template argument was declared here}}
  //   cxx98-note@-3 {{while substituting deduced template arguments}}

  X<Danger> x4;

  void f() {
    struct NoLinkage {};
    X<NoLinkage> a;
    // cxx98-error@-1 {{template argument uses local type }}
    X<const NoLinkage> b;
    // cxx98-error@-1 {{template argument uses local type }}
    get<NoLinkage>();
    // cxx98-error@-1 {{template argument uses local type }}
    //   cxx98-note@-2 {{while substituting explicitly-specified template arguments}}
    get<const NoLinkage>();
    // cxx98-error@-1 {{template argument uses local type }}
    //   cxx98-note@-2 {{while substituting explicitly-specified template arguments}}
    X<void (*)(NoLinkage A::*)> c;
    // cxx98-error@-1 {{template argument uses local type }}
    X<int NoLinkage::*> d;
    // cxx98-error@-1 {{template argument uses local type }}
  }
} // namespace cwg62

namespace cwg63 { // cwg63: 2.7
  template<typename T> struct S { typename T::error e; };
  extern S<int> *p;
  void *q = p;
} // namespace cwg63

namespace cwg64 { // cwg64: 2.7
  template<class T> void f(T);
  template<class T> void f(T*);
  template<> void f(int*);
  template<> void f<int>(int*);
  template<> void f(int);
} // namespace cwg64

// cwg65: na

namespace cwg66 { // cwg66: no
  namespace X {
    int f(int n); // #cwg66-f-first
  }
  using X::f;
  namespace X {
    int f(int n = 0);
    int f(int, int);
  }
  // FIXME: The first two calls here should be accepted.
  int a = f();
  // expected-error@-1 {{no matching function for call to 'f'}}
  //   expected-note@#cwg66-f-first {{candidate function not viable: requires single argument 'n', but no arguments were provided}}
  int b = f(1);
  int c = f(1, 2);
  // expected-error@-1 {{no matching function for call to 'f'}}
  //   expected-note@#cwg66-f-first {{candidate function not viable: requires single argument 'n', but 2 arguments were provided}}
} // namespace cwg66

// cwg67: na

namespace cwg68 { // cwg68: 2.8
  template<typename T> struct X {};
  struct ::cwg68::X<int> x1;
  struct ::cwg68::template X<int> x2;
  // cxx98-error@-1 {{'template' keyword outside of a template}}
  struct Y {
    friend struct X<int>;
    friend struct ::cwg68::X<char>;
    friend struct ::cwg68::template X<double>;
    // cxx98-error@-1 {{'template' keyword outside of a template}}
  };
  template<typename>
  struct Z {
    friend struct ::cwg68::template X<double>;
    friend typename ::cwg68::X<double>;
    // cxx98-error@-1 {{unelaborated friend declaration is a C++11 extension; specify 'struct' to befriend 'typename ::cwg68::X<double>'}}
  };
} // namespace cwg68

namespace cwg69 { // cwg69: 9
  template<typename T> static void f() {} // #cwg69-f
  // FIXME: Should we warn here?
  inline void g() { f<int>(); }
  extern template void f<char>();
  // cxx98-error@-1 {{extern templates are a C++11 extension}}
  // expected-error@-2 {{explicit instantiation declaration of 'f' with internal linkage}}
  template<void(*)()> struct Q {};
  Q<&f<int> > q;
  // cxx98-error@-1 {{non-type template argument referring to function 'f<int>' with internal linkage is a C++11 extension}}
  //   cxx98-note@#cwg69-f {{non-type template argument refers to function here}}
} // namespace cwg69

namespace cwg70 { // cwg70: 2.7
  template<int> struct A {};
  template<int I, int J> int f(int (&)[I + J], A<I>, A<J>);
  int arr[7];
  int k = f(arr, A<3>(), A<4>());
} // namespace cwg70

// cwg71: na
// cwg72: dup 69

namespace cwg73 { // cwg73: sup 1652
#if __cplusplus >= 201103L
  int a, b;
  static_assert(&a + 1 != &b, "");
  // since-cxx11-error@-1 {{static assertion expression is not an integral constant expression}}
  //   since-cxx11-note@-2 {{comparison against pointer '&a + 1' that points past the end of a complete object has unspecified value}}
#endif
} // namespace cwg73

namespace cwg74 { // cwg74: 2.7
  enum E { k = 5 };
  int (*p)[k] = new int[k][k];
} // namespace cwg74

namespace cwg75 { // cwg75: 2.7
  struct S {
    static int n = 0;
    // expected-error@-1 {{non-const static data member must be initialized out of line}}
  };
} // namespace cwg75

namespace cwg76 { // cwg76: 2.7
  const volatile int n = 1;
  static_assert(n, "");
  // expected-error@-1 {{static assertion expression is not an integral constant expression}}
  //   expected-note@-2 {{read of volatile-qualified type 'const volatile int' is not allowed in a constant expression}}
} // namespace cwg76

namespace cwg77 { // cwg77: 2.7
  struct A {
    struct B {};
    friend struct B;
  };
} // namespace cwg77

namespace cwg78 { // cwg78: sup ????
  // Under CWG78, this is valid, because 'k' has static storage duration, so is
  // zero-initialized.
  const int k;
  // expected-error@-1 {{default initialization of an object of const type 'const int'}}
} // namespace cwg78

// cwg79: na

namespace cwg80 { // cwg80: 2.9
  struct A {
    int A;
  };
  struct B {
    static int B;
    // expected-error@-1 {{member 'B' has the same name as its class}}
  };
  struct C {
    int C;
    // expected-error@-1 {{member 'C' has the same name as its class}}
    C();
  };
  struct D {
    D();
    int D;
    // expected-error@-1 {{member 'D' has the same name as its class}}
  };
} // namespace cwg80

// cwg81: na
// cwg82: dup 48

namespace cwg83 { // cwg83: 2.7
  int &f(const char*);
  char &f(char *);
  int &k = f("foo");
} // namespace cwg83

namespace cwg84 { // cwg84: 2.7
  struct B;
  struct A { operator B() const; };
  struct C {};
  struct B {
    B(B&); // #cwg84-copy-ctor
    B(C); // #cwg84-ctor-from-C
    operator C() const;
  };
  A a;
  // Cannot use B(C) / operator C() pair to construct the B from the B temporary
  // here. In C++17, we initialize the B object directly using 'A::operator B()'.
  B b = a;
  // cxx98-14-error@-1 {{no viable constructor copying variable of type 'B'}}
  //   cxx98-14-note@#cwg84-copy-ctor {{candidate constructor not viable: expects an lvalue for 1st argument}}
  //   cxx98-14-note@#cwg84-ctor-from-C {{candidate constructor not viable: no known conversion from 'B' to 'C' for 1st argument}}
} // namespace cwg84

namespace cwg85 { // cwg85: 3.4
  struct A {
    struct B;
    struct B {}; // #cwg85-B-def
    struct B;
    // expected-error@-1 {{class member cannot be redeclared}}
    //   expected-note@#cwg85-B-def {{previous declaration is here}}

    union U;
    union U {}; // #cwg85-U-def
    union U;
    // expected-error@-1 {{class member cannot be redeclared}}
    //   expected-note@#cwg85-U-def {{previous declaration is here}}

#if __cplusplus >= 201103L
    enum E1 : int;
    enum E1 : int { e1 }; // #cwg85-E1-def
    enum E1 : int;
    // since-cxx11-error@-1 {{class member cannot be redeclared}}
    //   since-cxx11-note@#cwg85-E1-def {{previous declaration is here}}

    enum class E2;
    enum class E2 { e2 }; // #cwg85-E2-def
    enum class E2;
    // since-cxx11-error@-1 {{class member cannot be redeclared}}
    //   since-cxx11-note@#cwg85-E2-def {{previous declaration is here}}
#endif
  };

  template <typename T>
  struct C {
    struct B {}; // #cwg85-C-B-def
    struct B;
    // expected-error@-1 {{class member cannot be redeclared}}
    //   expected-note@#cwg85-C-B-def {{previous declaration is here}}
  };
} // namespace cwg85

// cwg86: dup 446

namespace cwg87 { // cwg87: no
  // FIXME: Superseded by cwg1975
  template<typename T> struct X {};
  // FIXME: This is invalid.
  X<void() throw()> x;
  // This is valid under cwg87 but not under cwg1975.
  X<void(void() throw())> y;
} // namespace cwg87

namespace cwg88 { // cwg88: 2.8
  template<typename T> struct S {
    static const int a = 1; // #cwg88-a
    static const int b;
  };
  template<> const int S<int>::a = 4;
  // expected-error@-1 {{static data member 'a' already has an initializer}}
  //   expected-note@#cwg88-a {{previous initialization is here}}
  template<> const int S<int>::b = 4;
} // namespace cwg88

// cwg89: na

namespace cwg90 { // cwg90: 2.7
  struct A {
    template<typename T> friend void cwg90_f(T);
  };
  struct B : A {
    template<typename T> friend void cwg90_g(T);
    struct C {};
    union D {};
  };
  struct E : B {};
  struct F : B::C {};

  void test() {
    cwg90_f(A());
    cwg90_f(B());
    cwg90_f(B::C());
    // expected-error@-1 {{use of undeclared identifier 'cwg90_f'}}
    cwg90_f(B::D());
    // expected-error@-1 {{use of undeclared identifier 'cwg90_f'}}
    cwg90_f(E());
    cwg90_f(F());
    // expected-error@-1 {{use of undeclared identifier 'cwg90_f'}}

    cwg90_g(A());
    // expected-error@-1 {{use of undeclared identifier 'cwg90_g'}}
    cwg90_g(B());
    cwg90_g(B::C());
    cwg90_g(B::D());
    cwg90_g(E());
    cwg90_g(F());
    // expected-error@-1 {{use of undeclared identifier 'cwg90_g'}}
  }
} // namespace cwg90

namespace cwg91 { // cwg91: 2.7
  union U { friend int f(U); };
  int k = f(U());
} // namespace cwg91

namespace cwg92 { // cwg92: 4 c++17
  void f() throw(int, float);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (*p)() throw(int) = &f; // #cwg92-p
  // since-cxx17-error@#cwg92-p {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@#cwg92-p {{use 'noexcept(false)' instead}}
  // cxx98-14-error@#cwg92-p {{target exception specification is not superset of source}}
  // since-cxx17-warning@#cwg92-p {{target exception specification is not superset of source}}
  void (*q)() throw(int);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (**pp)() throw() = &q;
  // cxx98-14-error@-1 {{exception specifications are not allowed beyond a single level of indirection}}
  // since-cxx17-error@-2 {{cannot initialize a variable of type 'void (**)() throw()' with an rvalue of type 'void (**)() throw(int)'}}

  void g(void() throw()); // #cwg92-g
  // cxx98-14-warning@-1 {{mangled name of 'g' will change in C++17 due to non-throwing exception specification in function signature}}
  void h() throw() {
    g(f);
    // cxx98-14-error@-1 {{target exception specification is not superset of source}}
    // since-cxx17-error@-2 {{no matching function for call to 'g'}}
    //   since-cxx17-note@#cwg92-g {{candidate function not viable: no known conversion from 'void () throw(int, float)' to 'void (*)() throw()' for 1st argument}}
    g(q);
    // cxx98-14-error@-1 {{target exception specification is not superset of source}}
    // since-cxx17-error@-2 {{no matching function for call to 'g'}}
    //   since-cxx17-note@#cwg92-g {{candidate function not viable: no known conversion from 'void (*)() throw(int)' to 'void (*)() throw()' for 1st argument}}
  }

  // Prior to C++17, this is OK because the exception specification is not
  // considered in this context. In C++17, we *do* perform an implicit
  // conversion (which performs initialization), and the exception specification
  // is part of the type of the parameter, so this is invalid.
  template<void() throw()> struct X {}; // since-cxx17-note {{template parameter is declared here}}
  X<&f> xp;
  // since-cxx17-error@-1 {{value of type 'void (*)() throw(int, float)' is not implicitly convertible to 'void (*)() throw()'}}

  template<void() throw(int)> struct Y {};
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  Y<&h> yp; // ok
} // namespace cwg92

// cwg93: na

namespace cwg94 { // cwg94: 2.7
  struct A { static const int n = 5; };
  int arr[A::n];
} // namespace cwg94

namespace cwg95 { // cwg95: 3.3
  struct A;
  struct B;
  namespace N {
    class C {
      friend struct A;
      friend struct B;
      static void f(); // #cwg95-C-f
    };
    struct A *p; // cwg95::A, not cwg95::N::A.
  }
  A *q = N::p; // ok, same type
  struct B { void f() { N::C::f(); } };
  // expected-error@-1 {{'f' is a private member of 'cwg95::N::C'}}
  //   expected-note@#cwg95-C-f {{implicitly declared private here}}
} // namespace cwg95

namespace cwg96 { // cwg96: sup P1787
  struct A {
    void f(int);
    template<typename T> int f(T);
    template<typename T> struct S {};
  } a;
  template<template<typename> class X> struct B {};

  template<typename T>
  void test() {
    int k1 = a.template f<int>(0);
    // FIXME: This is ill-formed, because 'f' is not a template-id and does not
    // name a class template.
    // FIXME: What about alias templates?
    int k2 = a.template f(1);
    // expected-error@-1 {{a template argument list is expected after a name prefixed by the template keyword}}
    A::template S<int> s;
    B<A::template S> b;
  }
} // namespace cwg96

namespace cwg97 { // cwg97: 2.7
  struct A {
    static const int a = false;
    static const int b = !a;
  };
} // namespace cwg97

namespace cwg98 { // cwg98: 2.7
  void test(int n) {
    switch (n) {
      try { // #cwg98-try
        case 0:
        // expected-error@-1 {{cannot jump from switch statement to this case label}}
        //   expected-note@#cwg98-try {{jump bypasses initialization of try block}}
        x:
          throw n;
      } catch (...) { // #cwg98-catch
        case 1:
        // expected-error@-1 {{cannot jump from switch statement to this case label}}
        //   expected-note@#cwg98-catch {{jump bypasses initialization of catch block}}
        y:
          throw n;
      }
      case 2:
        goto x;
        // expected-error@-1 {{cannot jump from this goto statement to its label}}
        //   expected-note@#cwg98-try {{jump bypasses initialization of try block}}
      case 3:
        goto y;
        // expected-error@-1 {{cannot jump from this goto statement to its label}}
        //   expected-note@#cwg98-catch {{jump bypasses initialization of catch block}}
    }
  }
} // namespace cwg98

namespace cwg99 { // cwg99: sup 214
  template<typename T> void f(T&);
  template<typename T> int &f(const T&);
  const int n = 0;
  int &r = f(n);
} // namespace cwg99
