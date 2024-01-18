// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98,cxx98-14 -fexceptions -fcxx-exceptions -pedantic-errors -Wno-bind-to-temporary-copy
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,since-cxx11,cxx98-14,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx11,cxx98-14,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple

namespace dr1 { // dr1: no
  namespace X { extern "C" void dr1_f(int a = 1); }
  namespace Y { extern "C" void dr1_f(int a = 1); }
  using X::dr1_f; using Y::dr1_f;
  void g() {
    dr1_f(0);
    // FIXME: This should be rejected, due to the ambiguous default argument.
    dr1_f();
  }
  namespace X {
    using Y::dr1_f;
    void h() {
      dr1_f(0);
      // FIXME: This should be rejected, due to the ambiguous default argument.
      dr1_f();
    }
  }

  namespace X {
    void z(int);
  }
  void X::z(int = 1) {} // #dr1-z
  namespace X {
    void z(int = 1);
    // expected-error@-1 {{redefinition of default argument}}
    // expected-note@#dr1-z {{previous definition is here}}
  }

  void i(int = 1);
  void j() {
    void i(int = 1);
    using dr1::i;
    i(0);
    // FIXME: This should be rejected, due to the ambiguous default argument.
    i();
  }
  void k() {
    using dr1::i;
    void i(int = 1);
    i(0);
    // FIXME: This should be rejected, due to the ambiguous default argument.
    i();
  }
}

namespace dr3 { // dr3: yes
  template<typename T> struct A {};
  template<typename T> void f(T) { A<T> a; } // #dr3-f-T
  template void f(int);
  template<> struct A<int> {};
  // expected-error@-1 {{explicit specialization of 'dr3::A<int>' after instantiation}}
  // expected-note@#dr3-f-T {{implicit instantiation first required here}}
}

namespace dr4 { // dr4: 2.8
  extern "C" {
    static void dr4_f(int) {}
    static void dr4_f(float) {}
    void dr4_g(int) {} // #dr4-g-int
    void dr4_g(float) {}
    // expected-error@-1 {{conflicting types for 'dr4_g'}}
    // expected-note@#dr4-g-int {{previous definition is here}}
  }
}

namespace dr5 { // dr5: 3.1
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
}

namespace dr7 { // dr7: 3.4
  class A { public: ~A(); };
  class B : virtual private A {}; // #dr7-B
  class C : public B {} c; // #dr7-C
  // expected-error@#dr7-C {{inherited virtual base class 'A' has private destructor}}
  //   expected-note@#dr7-C {{in implicit default constructor for 'dr7::C' first required here}}
  //   expected-note@#dr7-B {{declared private here}}
  // expected-error@#dr7-C {{inherited virtual base class 'A' has private destructor}}
  //   expected-note@#dr7-C {{in implicit destructor for 'dr7::C' first required here}}
  //   expected-note@#dr7-B {{declared private here}}
  class VeryDerivedC : public B, virtual public A {} vdc;

  class X { ~X(); }; // #dr7-X
  class Y : X { ~Y() {} };
  // expected-error@-1 {{base class 'X' has private destructor}}
  // expected-note@#dr7-X {{implicitly declared private here}}

  namespace PR16370 { // This regressed the first time DR7 was fixed.
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
}

namespace dr8 { // dr8: dup 45
  class A {
    struct U;
    static const int k = 5;
    void f();
    template<typename, int, void (A::*)()> struct T;

    T<U, k, &A::f> *g();
  };
  A::T<A::U, A::k, &A::f> *A::g() { return 0; }
}

namespace dr9 { // dr9: 2.8
  struct B {
  protected:
    int m; // #dr9-m
    friend int R1();
  };
  struct N : protected B { // #dr9-N
    friend int R2();
  } n;
  int R1() { return n.m; }
  // expected-error@-1 {{'m' is a protected member of 'dr9::B'}}
  // expected-note@#dr9-N {{constrained by protected inheritance here}}
  // expected-note@#dr9-m {{member is declared here}}
  int R2() { return n.m; }
}

namespace dr10 { // dr10: dup 45
  class A {
    struct B {
      A::B *p;
    };
  };
}

namespace dr11 { // dr11: yes
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
}

namespace dr12 { // dr12: sup 239
  enum E { e };
  E &f(E, E = e);
  void g() {
    int &f(int, E = e);
    // Under DR12, these call two different functions.
    // Under DR239, they call the same function.
    int &b = f(e);
    int &c = f(1);
  }
}

namespace dr13 { // dr13: no
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
}

namespace dr14 { // dr14: 3.4
  namespace X { extern "C" int dr14_f(); }
  namespace Y { extern "C" int dr14_f(); }
  using namespace X;
  using namespace Y;
  int k = dr14_f();

  class C {
    int k;
    friend int Y::dr14_f();
  } c;
  namespace Z {
    extern "C" int dr14_f() { return c.k; }
  }

  namespace X { typedef int T; typedef int U; } // #dr14-X-U
  namespace Y { typedef int T; typedef long U; } // #dr14-Y-U
  T t; // ok, same type both times
  U u;
  // expected-error@-1 {{reference to 'U' is ambiguous}}
  // expected-note@#dr14-X-U {{candidate found by name lookup is 'dr14::X::U'}}
  // expected-note@#dr14-Y-U {{candidate found by name lookup is 'dr14::Y::U'}}
}

namespace dr15 { // dr15: yes
  template<typename T> void f(int); // #dr15-f-decl-first
  template<typename T> void f(int = 0);
  // expected-error@-1 {{default arguments cannot be added to a function template that has already been declared}}
  // expected-note@#dr15-f-decl-first {{previous template declaration is here}}
}

namespace dr16 { // dr16: 2.8
  class A { // #dr16-A
    void f(); // #dr16-A-f-decl
    friend class C;
  };
  class B : A {}; // #dr16-B
  class C : B {
    void g() {
      f();
      // expected-error@-1 {{'f' is a private member of 'dr16::A'}}
      // expected-note@#dr16-B {{constrained by implicitly private inheritance here}}
      // expected-note@#dr16-A-f-decl {{member is declared here}}
      A::f(); // #dr16-A-f-call
      // expected-error@#dr16-A-f-call {{'A' is a private member of 'dr16::A'}}
      //   expected-note@#dr16-B {{constrained by implicitly private inheritance here}}
      //   expected-note@#dr16-A {{member is declared here}}
      // expected-error@#dr16-A-f-call {{cannot cast 'dr16::C' to its private base class 'dr16::A'}}
      //   expected-note@#dr16-B {{implicitly declared private here}}
    }
  };
}

namespace dr17 { // dr17: yes
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
}

// dr18: sup 577

namespace dr19 { // dr19: 3.1
  struct A {
    int n; // #dr19-n
  };
  struct B : protected A { // #dr19-B
  };
  struct C : B {} c;
  struct D : B {
    int get1() { return c.n; }
    // expected-error@-1 {{'n' is a protected member of 'dr19::A'}}
    // expected-note@#dr19-B {{constrained by protected inheritance here}}
    // expected-note@#dr19-n {{member is declared here}}
    int get2() { return ((A&)c).n; } // ok, A is an accessible base of B from here
  };
}

namespace dr20 { // dr20: 2.8
  class X {
  public:
    X();
  private:
    X(const X&); // #dr20-X-ctor
  };
  X &f();
  X x = f();
  // expected-error@-1 {{calling a private constructor of class 'dr20::X'}}
  // expected-note@#dr20-X-ctor {{declared private here}}
}

namespace dr21 { // dr21: 3.4
  template<typename T> struct A;
  struct X {
    template<typename T = int> friend struct A;
    // expected-error@-1 {{default template argument not permitted on a friend template}}
    template<typename T = int> friend struct B;
    // expected-error@-1 {{default template argument not permitted on a friend template}}
  };
}

namespace dr22 { // dr22: sup 481
  template<typename dr22_T = dr22_T> struct X;
  // expected-error@-1 {{unknown type name 'dr22_T'}}
  typedef int T;
  template<typename T = T> struct Y;
}

namespace dr23 { // dr23: yes
  template<typename T> void f(T, T); // #dr23-f-T-T
  template<typename T> void f(T, int); // #dr23-f-T-int
  void g() { f(0, 0); }
  // expected-error@-1 {{call to 'f' is ambiguous}}
  // expected-note@#dr23-f-T-T {{candidate function [with T = int]}}
  // expected-note@#dr23-f-T-int {{candidate function [with T = int]}}
}

// dr24: na

namespace dr25 { // dr25: yes
  struct A {
    void f() throw(int);
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  };
  void (A::*f)() throw (int);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (A::*g)() throw () = f;
  // cxx98-14-error@-1 {{target exception specification is not superset of source}}
  // since-cxx17-error@-2 {{different exception specifications}}
  void (A::*g2)() throw () = 0;
  void (A::*h)() throw (int, char) = f;
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (A::*i)() throw () = &A::f;
  // cxx98-14-error@-1 {{target exception specification is not superset of source}}
  // since-cxx17-error@-2 {{different exception specifications}}
  void (A::*i2)() throw () = 0;
  void (A::*j)() throw (int, char) = &A::f;
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
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
}

namespace dr26 { // dr26: yes
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
    // expected-note@-2 {{default argument used here}}
  };
}

namespace dr27 { // dr27: yes
  enum E { e } n;
  E &m = true ? n : n;
}

// dr28: na lib

namespace dr29 { // dr29: 3.4
  void dr29_f0(); // #dr29-f0
  void g0() { void dr29_f0(); }
  extern "C++" void g0_cxx() { void dr29_f0(); }
  extern "C" void g0_c() { void dr29_f0(); }
  // expected-error@-1 {{declaration of 'dr29_f0' has a different language linkage}}
  // expected-note@#dr29-f0 {{previous declaration is here}}

  extern "C" void dr29_f1(); // #dr29-f1
  void g1() { void dr29_f1(); }
  extern "C" void g1_c() { void dr29_f1(); }
  extern "C++" void g1_cxx() { void dr29_f1(); }
  // expected-error@-1 {{declaration of 'dr29_f1' has a different language linkage}}
  // expected-note@#dr29-f1 {{previous declaration is here}}

  void g2() { void dr29_f2(); } // #dr29-f2
  extern "C" void dr29_f2();
  // expected-error@-1 {{declaration of 'dr29_f2' has a different language linkage}}
  // expected-note@#dr29-f2 {{previous declaration is here}}

  extern "C" void g3() { void dr29_f3(); } // #dr29-f3
  extern "C++" void dr29_f3();
  // expected-error@-1 {{declaration of 'dr29_f3' has a different language linkage}}
  // expected-note@#dr29-f3 {{previous declaration is here}}

  extern "C++" void g4() { void dr29_f4(); } // #dr29-f4
  extern "C" void dr29_f4();
  // expected-error@-1 {{declaration of 'dr29_f4' has a different language linkage}}
  // expected-note@#dr29-f4 {{previous declaration is here}}

  extern "C" void g5();
  extern "C++" void dr29_f5();
  void g5() {
    void dr29_f5(); // ok, g5 is extern "C" but we're not inside the linkage-specification here.
  }

  extern "C++" void g6();
  extern "C" void dr29_f6();
  void g6() {
    void dr29_f6(); // ok, g6 is extern "C" but we're not inside the linkage-specification here.
  }

  extern "C" void g7();
  extern "C++" void dr29_f7(); // #dr29-f7
  extern "C" void g7() {
    void dr29_f7();
    // expected-error@-1 {{declaration of 'dr29_f7' has a different language linkage}}
    // expected-note@#dr29-f7 {{previous declaration is here}}
  }

  extern "C++" void g8();
  extern "C" void dr29_f8(); // #dr29-f8
  extern "C++" void g8() {
    void dr29_f8();
    // expected-error@-1 {{declaration of 'dr29_f8' has a different language linkage}}
    // expected-note@#dr29-f8 {{previous declaration is here}}
  }
}

namespace dr30 { // dr30: sup 468 c++11
  struct A {
    template<int> static int f();
  } a, *p = &a;
  // FIXME: It's not clear whether DR468 applies to C++98 too.
  int x = A::template f<0>();
  // cxx98-error@-1 {{'template' keyword outside of a template}}
  int y = a.template f<0>();
  // cxx98-error@-1 {{'template' keyword outside of a template}}
  int z = p->template f<0>();
  // cxx98-error@-1 {{'template' keyword outside of a template}}
}

namespace dr31 { // dr31: 2.8
  class X {
  private:
    void operator delete(void*); // #dr31-delete
  };
  // We would call X::operator delete if X() threw (even though it can't,
  // and even though we allocated the X using ::operator delete).
  X *p = new X;
  // expected-error@-1 {{'operator delete' is a private member of 'dr31::X'}}
  // expected-note@#dr31-delete {{declared private here}}
}

// dr32: na

namespace dr33 { // dr33: 9
  namespace X { struct S; void f(void (*)(S)); } // #dr33-f-S
  namespace Y { struct T; void f(void (*)(T)); } // #dr33-f-T
  void g(X::S);
  template<typename Z> Z g(Y::T);
  void h() { f(&g); }
  // expected-error@-1 {{call to 'f' is ambiguous}}
  // expected-note@#dr33-f-S {{candidate function}}
  // expected-note@#dr33-f-T {{candidate function}}

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
}

// dr34: na
// dr35: dup 178

namespace dr36 { // dr36: 2.8
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
    using B::i; // #dr36-ex2-B-i-first
    using B::i;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-ex2-B-i-first {{previous using declaration}}

    using C::i; // #dr36-ex2-C-i-first
    using C::i;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-ex2-C-i-first {{previous using declaration}}

    using B::j; // #dr36-ex2-B-j-first
    using B::j;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-ex2-B-j-first {{previous using declaration}}

    using C::j; // #dr36-ex2-C-j-first
    using C::j;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-ex2-C-j-first {{previous using declaration}}
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
    using B<T>::i; // #dr36-ex3-B-i-first
    using B<T>::i;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-ex3-B-i-first {{previous using declaration}}

    using C<T>::i; // #dr36-ex3-C-i-first
    using C<T>::i;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-ex3-C-i-first {{previous using declaration}}

    using B<T>::j; // #dr36-ex3-B-j-first
    using B<T>::j;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-ex3-B-j-first {{previous using declaration}}

    using C<T>::j; // #dr36-ex3-C-j-first
    using C<T>::j;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-ex3-C-j-first {{previous using declaration}}
  };
}
namespace example4 {
  template<typename T>
  struct E {
    T k;
  };

  template<typename T>
  struct G : E<T> {
    using E<T>::k; // #dr36-E-k-first
    using E<T>::k;
    // expected-error@-1 {{redeclaration of using declaration}}
    // expected-note@#dr36-E-k-first {{previous using declaration}}
  };
}
}

// dr37: sup 475

namespace dr38 { // dr38: yes
  template<typename T> struct X {};
  template<typename T> X<T> operator+(X<T> a, X<T> b) { return a; }
  template X<int> operator+<int>(X<int>, X<int>);
}

namespace dr39 { // dr39: no
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
      int &x(int); // #dr39-A-x-decl
      static int &y(int); // #dr39-A-y-decl
    };
    struct V {
      int &z(int);
    };
    struct B : A, virtual V {
      using A::x; // #dr39-using-A-x
      float &x(float);
      using A::y; // #dr39-using-A-y
      static float &y(float);
      using V::z;
      float &z(float);
    };
    struct C : A, B, virtual V {} c;
    /* expected-warning@-1
    {{direct base 'A' is inaccessible due to ambiguity:
    struct dr39::example2::C -> A
    struct dr39::example2::C -> B -> A}} */
    int &x = c.x(0);
    // expected-error@-1 {{member 'x' found in multiple base classes of different types}}
    // expected-note@#dr39-A-x-decl {{member found by ambiguous name lookup}}
    // expected-note@#dr39-using-A-x {{member found by ambiguous name lookup}}

    // FIXME: This is valid, because we find the same static data member either way.
    int &y = c.y(0);
    // expected-error@-1 {{member 'y' found in multiple base classes of different types}}
    // expected-note@#dr39-A-y-decl {{member found by ambiguous name lookup}}
    // expected-note@#dr39-using-A-y {{member found by ambiguous name lookup}}
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
    struct A { int n; }; // #dr39-ex4-A-n
    struct B : A {};
    struct C : A {};
    struct D : B, C { int f() { return n; } };
    /* expected-error@-1
    {{non-static member 'n' found in multiple base-class subobjects of type 'A':
    struct dr39::example4::D -> B -> A
    struct dr39::example4::D -> C -> A}} */
    // expected-note@#dr39-ex4-A-n {{member found by ambiguous name lookup}}
  }

  namespace PR5916 {
    // FIXME: This is valid.
    struct A { int n; }; // #dr39-A-n
    struct B : A {};
    struct C : A {};
    struct D : B, C {};
    int k = sizeof(D::n); // #dr39-sizeof
    /* expected-error@#dr39-sizeof
    {{non-static member 'n' found in multiple base-class subobjects of type 'A':
    struct dr39::PR5916::D -> B -> A
    struct dr39::PR5916::D -> C -> A}} */
    // expected-note@#dr39-A-n {{member found by ambiguous name lookup}}

    // expected-error@#dr39-sizeof {{unknown type name}}
#if __cplusplus >= 201103L
    decltype(D::n) n;
    /* expected-error@-1
    {{non-static member 'n' found in multiple base-class subobjects of type 'A':
    struct dr39::PR5916::D -> B -> A
    struct dr39::PR5916::D -> C -> A}} */
    // expected-note@#dr39-A-n {{member found by ambiguous name lookup}}
#endif
  }
}

// dr40: na

namespace dr41 { // dr41: yes
  struct S f(S);
}

namespace dr42 { // dr42: yes
  struct A { static const int k = 0; };
  struct B : A { static const int k = A::k; };
}

// dr43: na

namespace dr44 { // dr44: sup 727
  struct A {
    template<int> void f();
    template<> void f<0>();
  };
}

namespace dr45 { // dr45: yes
  class A {
    class B {};
    class C : B {};
    C c;
  };
}

namespace dr46 { // dr46: yes
  template<typename> struct A { template<typename> struct B {}; };
  template template struct A<int>::B<int>;
  // expected-error@-1 {{expected unqualified-id}}
}

namespace dr47 { // dr47: sup 329
  template<typename T> struct A {
    friend void f() { T t; } // #dr47-f
    // expected-error@-1 {{redefinition of 'f'}}
    // expected-note@#dr47-b {{in instantiation of template class 'dr47::A<float>' requested here}}
    // expected-note@#dr47-f {{previous definition is here}}
  };
  A<int> a;
  A<float> b; // #dr47-b

  void f();
  void g() { f(); }
}

namespace dr48 { // dr48: yes
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
}

namespace dr49 { // dr49: 2.8
  template<int*> struct A {}; // #dr49-A
  int k;
#if __has_feature(cxx_constexpr)
  constexpr
#endif
  int *const p = &k; // #dr49-p
  A<&k> a;
  A<p> b; // #dr49-b
  // cxx98-error@#dr49-b {{non-type template argument referring to object 'p' with internal linkage is a C++11 extension}}
  //   cxx98-note@#dr49-p {{non-type template argument refers to object here}}
  // cxx98-14-error@#dr49-b {{non-type template argument for template parameter of pointer type 'int *' must have its address taken}}
  //   cxx98-14-note@#dr49-A {{template parameter is declared here}}
  int *q = &k; // #dr49-q
  A<q> c; // #dr49-c
  // cxx98-error@#dr49-c {{non-type template argument for template parameter of pointer type 'int *' must have its address taken}}
  //   cxx98-note@#dr49-A {{template parameter is declared here}}
  // cxx11-14-error@#dr49-c {{non-type template argument of type 'int *' is not a constant expression}}
  //   cxx11-14-note@#dr49-c {{read of non-constexpr variable 'q' is not allowed in a constant expression}}
  //   cxx11-14-note@#dr49-q {{declared here}}
  //   cxx11-14-note@#dr49-A {{template parameter is declared here}}
  // since-cxx17-error@#dr49-c {{non-type template argument is not a constant expression}}
  //   since-cxx17-note@#dr49-c {{read of non-constexpr variable 'q' is not allowed in a constant expression}}
  //   since-cxx17-note@#dr49-q {{declared here}}
}

namespace dr50 { // dr50: yes
  struct X; // #dr50-X
  extern X *p;
  X *q = (X*)p;
  X *r = static_cast<X*>(p);
  X *s = const_cast<X*>(p);
  X *t = reinterpret_cast<X*>(p);
  X *u = dynamic_cast<X*>(p);
  // expected-error@-1 {{'dr50::X' is an incomplete type}}
  // expected-note@#dr50-X {{forward declaration of 'dr50::X'}}
}

namespace dr51 { // dr51: 2.8
  struct A {};
  struct B : A {};
  struct S {
    operator A&();
    operator B&();
  } s;
  A &a = s;
}

namespace dr52 { // dr52: 2.8
  struct A { int n; }; // #dr52-A
  struct B : private A {} b; // #dr52-B
  int k = b.A::n; // #dr52-k
  // FIXME: This first diagnostic is very strangely worded, and seems to be bogus.
  // expected-error@#dr52-k {{'A' is a private member of 'dr52::A'}}
  //   expected-note@#dr52-B {{constrained by private inheritance here}}
  //   expected-note@#dr52-A {{member is declared here}}
  // expected-error@#dr52-k {{cannot cast 'struct B' to its private base class 'dr52::A'}}
  //   expected-note@#dr52-B {{declared private here}}
}

namespace dr53 { // dr53: yes
  int n = 0;
  enum E { e } x = static_cast<E>(n);
}

namespace dr54 { // dr54: 2.8
  struct A { int a; } a;
  struct V { int v; } v;
  struct B : private A, virtual V { int b; } b; // #dr54-B

  A &sab = static_cast<A&>(b);
  // expected-error@-1 {{cannot cast 'struct B' to its private base class 'A'}}
  // expected-note@#dr54-B {{declared private here}}
  A *spab = static_cast<A*>(&b);
  // expected-error@-1 {{cannot cast 'struct B' to its private base class 'A'}}
  // expected-note@#dr54-B {{declared private here}}
  int A::*smab = static_cast<int A::*>(&B::b);
  // expected-error@-1 {{cannot cast 'dr54::B' to its private base class 'dr54::A'}}
  // expected-note@#dr54-B {{declared private here}}
  B &sba = static_cast<B&>(a);
  // expected-error@-1 {{cannot cast private base class 'dr54::A' to 'dr54::B'}}
  // expected-note@#dr54-B {{declared private here}}
  B *spba = static_cast<B*>(&a);
  // expected-error@-1 {{cannot cast private base class 'dr54::A' to 'dr54::B'}}
  // expected-note@#dr54-B {{declared private here}}
  int B::*smba = static_cast<int B::*>(&A::a);
  // expected-error@-1 {{cannot cast private base class 'dr54::A' to 'dr54::B'}}
  // expected-note@#dr54-B {{declared private here}}

  V &svb = static_cast<V&>(b);
  V *spvb = static_cast<V*>(&b);
  int V::*smvb = static_cast<int V::*>(&B::b);
  // expected-error@-1 {{conversion from pointer to member of class 'dr54::B' to pointer to member of class 'dr54::V' via virtual base 'dr54::V' is not allowed}}
  B &sbv = static_cast<B&>(v);
  // expected-error@-1 {{cannot cast 'struct V' to 'B &' via virtual base 'dr54::V'}}
  B *spbv = static_cast<B*>(&v);
  // expected-error@-1 {{cannot cast 'dr54::V *' to 'B *' via virtual base 'dr54::V'}}
  int B::*smbv = static_cast<int B::*>(&V::v);
  // expected-error@-1 {{conversion from pointer to member of class 'dr54::V' to pointer to member of class 'dr54::B' via virtual base 'dr54::V' is not allowed}}

  A &cab = (A&)(b);
  A *cpab = (A*)(&b);
  int A::*cmab = (int A::*)(&B::b);
  B &cba = (B&)(a);
  B *cpba = (B*)(&a);
  int B::*cmba = (int B::*)(&A::a);

  V &cvb = (V&)(b);
  V *cpvb = (V*)(&b);
  int V::*cmvb = (int V::*)(&B::b);
  // expected-error@-1 {{conversion from pointer to member of class 'dr54::B' to pointer to member of class 'dr54::V' via virtual base 'dr54::V' is not allowed}}
  B &cbv = (B&)(v);
  // expected-error@-1 {{cannot cast 'struct V' to 'B &' via virtual base 'dr54::V'}}
  B *cpbv = (B*)(&v);
  // expected-error@-1 {{cannot cast 'dr54::V *' to 'B *' via virtual base 'dr54::V'}}
  int B::*cmbv = (int B::*)(&V::v);
  // expected-error@-1 {{conversion from pointer to member of class 'dr54::V' to pointer to member of class 'dr54::B' via virtual base 'dr54::V' is not allowed}}
}

namespace dr55 { // dr55: yes
  enum E { e = 5 };
  int test[(e + 1 == 6) ? 1 : -1];
}

namespace dr56 { // dr56: yes
  struct A {
    typedef int T; // #dr56-typedef-int-T-first
    typedef int T;
    // expected-error@-1 {{redefinition of 'T'}}
    // expected-note@#dr56-typedef-int-T-first {{previous definition is here}}
  };
  struct B {
    struct X;
    typedef X X; // #dr56-typedef-X-X-first
    typedef X X;
    // expected-error@-1 {{redefinition of 'X'}}
    // expected-note@#dr56-typedef-X-X-first {{previous definition is here}}
  };
}

namespace dr58 { // dr58: 3.1
  // FIXME: Ideally, we should have a CodeGen test for this.
#if __cplusplus >= 201103L
  enum E1 { E1_0 = 0, E1_1 = 1 };
  enum E2 { E2_0 = 0, E2_m1 = -1 };
  struct X { E1 e1 : 1; E2 e2 : 1; };
  static_assert(X{E1_1, E2_m1}.e1 == 1, "");
  static_assert(X{E1_1, E2_m1}.e2 == -1, "");
#endif
}

namespace dr59 { // dr59: yes
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-volatile"
  template<typename T> struct convert_to { operator T() const; };
  struct A {}; // #dr59-A
  struct B : A {}; // #dr59-B

  A a1 = convert_to<A>();
  A a2 = convert_to<A&>();
  A a3 = convert_to<const A>();
  A a4 = convert_to<const volatile A>();
  // cxx98-14-error@-1 {{no viable constructor copying variable of type 'const volatile dr59::A'}}
  // cxx98-14-note@#dr59-A {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const volatile dr59::A') would lose volatile qualifier}}
  // cxx11-14-note@#dr59-A {{candidate constructor (the implicit move constructor) not viable: 1st argument ('const volatile dr59::A') would lose const and volatile qualifiers}}
  // cxx98-14-note@#dr59-A {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  A a5 = convert_to<const volatile A&>();
  // expected-error@-1 {{no viable constructor copying variable of type 'const volatile dr59::A'}}
  // expected-note@#dr59-A {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const volatile dr59::A') would lose volatile qualifier}}
  // since-cxx11-note@#dr59-A {{candidate constructor (the implicit move constructor) not viable: 1st argument ('const volatile dr59::A') would lose const and volatile qualifiers}}
  // expected-note@#dr59-A {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}

  B b1 = convert_to<B>();
  B b2 = convert_to<B&>();
  B b3 = convert_to<const B>();
  B b4 = convert_to<const volatile B>();
  // cxx98-14-error@-1 {{no viable constructor copying variable of type 'const volatile dr59::B'}}
  // cxx98-14-note@#dr59-B {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const volatile dr59::B') would lose volatile qualifier}}
  // cxx11-14-note@#dr59-B {{candidate constructor (the implicit move constructor) not viable: 1st argument ('const volatile dr59::B') would lose const and volatile qualifiers}}
  // cxx98-14-note@#dr59-B {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  B b5 = convert_to<const volatile B&>();
  // expected-error@-1 {{no viable constructor copying variable of type 'const volatile dr59::B'}}
  // expected-note@#dr59-B {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const volatile dr59::B') would lose volatile qualifier}}
  // since-cxx11-note@#dr59-B {{candidate constructor (the implicit move constructor) not viable: 1st argument ('const volatile dr59::B') would lose const and volatile qualifiers}}
  // expected-note@#dr59-B {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}

  A c1 = convert_to<B>();
  A c2 = convert_to<B&>();
  A c3 = convert_to<const B>();
  A c4 = convert_to<const volatile B>();
  // expected-error@-1 {{no viable constructor copying variable of type 'const volatile dr59::B'}}
  // expected-note@#dr59-A {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'const volatile dr59::B' to 'const A &' for 1st argument}}
  // since-cxx11-note@#dr59-A {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'const volatile dr59::B' to 'A &&' for 1st argument}}
  // expected-note@#dr59-A {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  A c5 = convert_to<const volatile B&>();
  // expected-error@-1 {{no viable constructor copying variable of type 'const volatile dr59::B'}}
  // expected-note@#dr59-A {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'const volatile dr59::B' to 'const A &' for 1st argument}}
  // since-cxx11-note@#dr59-A {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'const volatile dr59::B' to 'A &&' for 1st argument}}
  // expected-note@#dr59-A {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}

  int n1 = convert_to<int>();
  int n2 = convert_to<int&>();
  int n3 = convert_to<const int>();
  int n4 = convert_to<const volatile int>();
  int n5 = convert_to<const volatile int&>();
#pragma clang diagnostic pop
}

namespace dr60 { // dr60: yes
  void f(int &);
  int &f(...);
  const int k = 0;
  int &n = f(k);
}

namespace dr61 { // dr61: 3.4
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
}

namespace dr62 { // dr62: 2.9
  struct A {
    struct { int n; } b;
  };
  template<typename T> struct X {};
  template<typename T> T get() { return get<T>(); }
  template<typename T> int take(T) { return 0; }

  X<A> x1;
  A a = get<A>();

  typedef struct { } *NoNameForLinkagePtr; // #dr62-unnamed
  NoNameForLinkagePtr noNameForLinkagePtr;

  struct Danger {
    NoNameForLinkagePtr p;
  };

  X<NoNameForLinkagePtr> x2;
  // cxx98-error@-1 {{template argument uses unnamed type}}
  // cxx98-note@#dr62-unnamed {{unnamed type used in template argument was declared here}}
  X<const NoNameForLinkagePtr> x3;
  // cxx98-error@-1 {{template argument uses unnamed type}}
  // cxx98-note@#dr62-unnamed {{unnamed type used in template argument was declared here}}
  NoNameForLinkagePtr p1 = get<NoNameForLinkagePtr>();
  // cxx98-error@-1 {{template argument uses unnamed type}}
  // cxx98-note@#dr62-unnamed {{unnamed type used in template argument was declared here}}
  NoNameForLinkagePtr p2 = get<const NoNameForLinkagePtr>();
  // cxx98-error@-1 {{template argument uses unnamed type}}
  // cxx98-note@#dr62-unnamed {{unnamed type used in template argument was declared here}}
  int n1 = take(noNameForLinkagePtr);
  // cxx98-error@-1 {{template argument uses unnamed type}}
  // cxx98-note@#dr62-unnamed {{unnamed type used in template argument was declared here}}

  X<Danger> x4;

  void f() {
    struct NoLinkage {};
    X<NoLinkage> a;
    // cxx98-error@-1 {{template argument uses local type }}
    X<const NoLinkage> b;
    // cxx98-error@-1 {{template argument uses local type }}
    get<NoLinkage>();
    // cxx98-error@-1 {{template argument uses local type }}
    get<const NoLinkage>();
    // cxx98-error@-1 {{template argument uses local type }}
    X<void (*)(NoLinkage A::*)> c;
    // cxx98-error@-1 {{template argument uses local type }}
    X<int NoLinkage::*> d;
    // cxx98-error@-1 {{template argument uses local type }}
  }
}

namespace dr63 { // dr63: yes
  template<typename T> struct S { typename T::error e; };
  extern S<int> *p;
  void *q = p;
}

namespace dr64 { // dr64: yes
  template<class T> void f(T);
  template<class T> void f(T*);
  template<> void f(int*);
  template<> void f<int>(int*);
  template<> void f(int);
}

// dr65: na

namespace dr66 { // dr66: no
  namespace X {
    int f(int n); // #dr66-f-first
  }
  using X::f;
  namespace X {
    int f(int n = 0);
    int f(int, int);
  }
  // FIXME: The first two calls here should be accepted.
  int a = f();
  // expected-error@-1 {{no matching function for call to 'f'}}
  // expected-note@#dr66-f-first {{candidate function not viable: requires single argument 'n', but no arguments were provided}}
  int b = f(1);
  int c = f(1, 2);
  // expected-error@-1 {{no matching function for call to 'f'}}
  // expected-note@#dr66-f-first {{candidate function not viable: requires single argument 'n', but 2 arguments were provided}}
}

// dr67: na

namespace dr68 { // dr68: 2.8
  template<typename T> struct X {};
  struct ::dr68::X<int> x1;
  struct ::dr68::template X<int> x2;
  // cxx98-error@-1 {{'template' keyword outside of a template}}
  struct Y {
    friend struct X<int>;
    friend struct ::dr68::X<char>;
    friend struct ::dr68::template X<double>;
    // cxx98-error@-1 {{'template' keyword outside of a template}}
  };
  template<typename>
  struct Z {
    friend struct ::dr68::template X<double>;
    friend typename ::dr68::X<double>;
    // cxx98-error@-1 {{unelaborated friend declaration is a C++11 extension; specify 'struct' to befriend 'typename ::dr68::X<double>'}}
  };
}

namespace dr69 { // dr69: 9
  template<typename T> static void f() {} // #dr69-f
  // FIXME: Should we warn here?
  inline void g() { f<int>(); }
  extern template void f<char>();
  // cxx98-error@-1 {{extern templates are a C++11 extension}}
  // expected-error@-2 {{explicit instantiation declaration of 'f' with internal linkage}}
  template<void(*)()> struct Q {};
  Q<&f<int> > q;
  // cxx98-error@-1 {{non-type template argument referring to function 'f<int>' with internal linkage is a C++11 extension}}
  // cxx98-note@#dr69-f {{non-type template argument refers to function here}}
}

namespace dr70 { // dr70: yes
  template<int> struct A {};
  template<int I, int J> int f(int (&)[I + J], A<I>, A<J>);
  int arr[7];
  int k = f(arr, A<3>(), A<4>());
}

// dr71: na
// dr72: dup 69

#if __cplusplus >= 201103L
namespace dr73 { // dr73: sup 1652
  int a, b;
  static_assert(&a + 1 != &b, "");
  // expected-error@-1 {{static assertion expression is not an integral constant expression}}
  // expected-note@-2 {{comparison against pointer '&a + 1' that points past the end of a complete object has unspecified value}}
}
#endif

namespace dr74 { // dr74: yes
  enum E { k = 5 };
  int (*p)[k] = new int[k][k];
}

namespace dr75 { // dr75: yes
  struct S {
    static int n = 0;
    // expected-error@-1 {{non-const static data member must be initialized out of line}}
  };
}

namespace dr76 { // dr76: yes
  const volatile int n = 1;
  int arr[n]; // #dr76-vla
  // expected-error@#dr76-vla {{variable length arrays in C++ are a Clang extension}}
  //   expected-note@#dr76-vla {{read of volatile-qualified type 'const volatile int' is not allowed in a constant expression}}
  // expected-error@#dr76-vla {{variable length array declaration not allowed at file scope}}
}

namespace dr77 { // dr77: yes
  struct A {
    struct B {};
    friend struct B;
  };
}

namespace dr78 { // dr78: sup ????
  // Under DR78, this is valid, because 'k' has static storage duration, so is
  // zero-initialized.
  const int k;
  // expected-error@-1 {{default initialization of an object of const type 'const int'}}
}

// dr79: na

namespace dr80 { // dr80: 2.9
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
}

// dr81: na
// dr82: dup 48

namespace dr83 { // dr83: yes
  int &f(const char*);
  char &f(char *);
  int &k = f("foo");
}

namespace dr84 { // dr84: yes
  struct B;
  struct A { operator B() const; };
  struct C {};
  struct B {
    B(B&); // #dr84-copy-ctor
    B(C); // #dr84-ctor-from-C
    operator C() const;
  };
  A a;
  // Cannot use B(C) / operator C() pair to construct the B from the B temporary
  // here. In C++17, we initialize the B object directly using 'A::operator B()'.
  B b = a;
  // cxx98-14-error@-1 {{no viable constructor copying variable of type 'B'}}
  // cxx98-14-note@#dr84-copy-ctor {{candidate constructor not viable: expects an lvalue for 1st argument}}
  // cxx98-14-note@#dr84-ctor-from-C {{candidate constructor not viable: no known conversion from 'B' to 'C' for 1st argument}}
}

namespace dr85 { // dr85: 3.4
  struct A {
    struct B;
    struct B {}; // #dr85-B-def
    struct B;
    // expected-error@-1 {{class member cannot be redeclared}}
    // expected-note@#dr85-B-def {{previous declaration is here}}

    union U;
    union U {}; // #dr85-U-def
    union U;
    // expected-error@-1 {{class member cannot be redeclared}}
    // expected-note@#dr85-U-def {{previous declaration is here}}

#if __cplusplus >= 201103L
    enum E1 : int;
    enum E1 : int { e1 }; // #dr85-E1-def
    enum E1 : int;
    // expected-error@-1 {{class member cannot be redeclared}}
    // expected-note@#dr85-E1-def {{previous declaration is here}}

    enum class E2;
    enum class E2 { e2 }; // #dr85-E2-def
    enum class E2;
    // expected-error@-1 {{class member cannot be redeclared}}
    // expected-note@#dr85-E2-def {{previous declaration is here}}
#endif
  };

  template <typename T>
  struct C {
    struct B {}; // #dr85-C-B-def
    struct B;
    // expected-error@-1 {{class member cannot be redeclared}}
    // expected-note@#dr85-C-B-def {{previous declaration is here}}
  };
}

// dr86: dup 446

namespace dr87 { // dr87: no
  // FIXME: Superseded by dr1975
  template<typename T> struct X {};
  // FIXME: This is invalid.
  X<void() throw()> x;
  // This is valid under dr87 but not under dr1975.
  X<void(void() throw())> y;
}

namespace dr88 { // dr88: 2.8
  template<typename T> struct S {
    static const int a = 1; // #dr88-a
    static const int b;
  };
  template<> const int S<int>::a = 4;
  // expected-error@-1 {{static data member 'a' already has an initializer}}
  // expected-note@#dr88-a {{previous initialization is here}}
  template<> const int S<int>::b = 4;
}

// dr89: na

namespace dr90 { // dr90: yes
  struct A {
    template<typename T> friend void dr90_f(T);
  };
  struct B : A {
    template<typename T> friend void dr90_g(T);
    struct C {};
    union D {};
  };
  struct E : B {};
  struct F : B::C {};

  void test() {
    dr90_f(A());
    dr90_f(B());
    dr90_f(B::C());
    // expected-error@-1 {{use of undeclared identifier 'dr90_f'}}
    dr90_f(B::D());
    // expected-error@-1 {{use of undeclared identifier 'dr90_f'}}
    dr90_f(E());
    dr90_f(F());
    // expected-error@-1 {{use of undeclared identifier 'dr90_f'}}

    dr90_g(A());
    // expected-error@-1 {{use of undeclared identifier 'dr90_g'}}
    dr90_g(B());
    dr90_g(B::C());
    dr90_g(B::D());
    dr90_g(E());
    dr90_g(F());
    // expected-error@-1 {{use of undeclared identifier 'dr90_g'}}
  }
}

namespace dr91 { // dr91: yes
  union U { friend int f(U); };
  int k = f(U());
}

namespace dr92 { // dr92: 4 c++17
  void f() throw(int, float);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (*p)() throw(int) = &f; // #dr92-p
  // since-cxx17-error@#dr92-p {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@#dr92-p {{use 'noexcept(false)' instead}}
  // cxx98-14-error@#dr92-p {{target exception specification is not superset of source}}
  // since-cxx17-warning@#dr92-p {{target exception specification is not superset of source}}
  void (*q)() throw(int);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  void (**pp)() throw() = &q;
  // cxx98-14-error@-1 {{exception specifications are not allowed beyond a single level of indirection}}
  // since-cxx17-error@-2 {{cannot initialize a variable of type 'void (**)() throw()' with an rvalue of type 'void (**)() throw(int)'}}

  void g(void() throw()); // #dr92-g
  // cxx98-14-warning@-1 {{mangled name of 'g' will change in C++17 due to non-throwing exception specification in function signature}}
  void h() throw() {
    g(f);
    // cxx98-14-error@-1 {{target exception specification is not superset of source}}
    // since-cxx17-error@-2 {{no matching function for call to 'g'}}
    //   since-cxx17-note@#dr92-g {{candidate function not viable: no known conversion from 'void () throw(int, float)' to 'void (*)() throw()' for 1st argument}}
    g(q);
    // cxx98-14-error@-1 {{target exception specification is not superset of source}}
    // since-cxx17-error@-2 {{no matching function for call to 'g'}}
    //   since-cxx17-note@#dr92-g {{candidate function not viable: no known conversion from 'void (*)() throw(int)' to 'void (*)() throw()' for 1st argument}}
  }

  // Prior to C++17, this is OK because the exception specification is not
  // considered in this context. In C++17, we *do* perform an implicit
  // conversion (which performs initialization), and the exception specification
  // is part of the type of the parameter, so this is invalid.
  template<void() throw()> struct X {};
  X<&f> xp;
  // since-cxx17-error@-1 {{value of type 'void (*)() throw(int, float)' is not implicitly convertible to 'void (*)() throw()'}}

  template<void() throw(int)> struct Y {};
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  Y<&h> yp; // ok
}

// dr93: na

namespace dr94 { // dr94: yes
  struct A { static const int n = 5; };
  int arr[A::n];
}

namespace dr95 { // dr95: 3.3
  struct A;
  struct B;
  namespace N {
    class C {
      friend struct A;
      friend struct B;
      static void f(); // #dr95-C-f
    };
    struct A *p; // dr95::A, not dr95::N::A.
  }
  A *q = N::p; // ok, same type
  struct B { void f() { N::C::f(); } };
  // expected-error@-1 {{'f' is a private member of 'dr95::N::C'}}
  // expected-note@#dr95-C-f {{implicitly declared private here}}
}

namespace dr96 { // dr96: no
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
    A::template S<int> s;
    B<A::template S> b;
  }
}

namespace dr97 { // dr97: yes
  struct A {
    static const int a = false;
    static const int b = !a;
  };
}

namespace dr98 { // dr98: yes
  void test(int n) {
    switch (n) {
      try { // #dr98-try
        case 0:
        // expected-error@-1 {{cannot jump from switch statement to this case label}}
        // expected-note@#dr98-try {{jump bypasses initialization of try block}}
        x:
          throw n;
      } catch (...) { // #dr98-catch
        case 1:
        // expected-error@-1 {{cannot jump from switch statement to this case label}}
        // expected-note@#dr98-catch {{jump bypasses initialization of catch block}}
        y:
          throw n;
      }
      case 2:
        goto x;
        // expected-error@-1 {{cannot jump from this goto statement to its label}}
        // expected-note@#dr98-try {{jump bypasses initialization of try block}}
      case 3:
        goto y;
        // expected-error@-1 {{cannot jump from this goto statement to its label}}
        // expected-note@#dr98-catch {{jump bypasses initialization of catch block}}
    }
  }
}

namespace dr99 { // dr99: sup 214
  template<typename T> void f(T&);
  template<typename T> int &f(const T&);
  const int n = 0;
  int &r = f(n);
}
