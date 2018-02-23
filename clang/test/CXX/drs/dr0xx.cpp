// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -Wno-bind-to-temporary-copy
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -triple %itanium_abi_triple

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
  void X::z(int = 1) {} // expected-note {{previous}}
  namespace X {
    void z(int = 1); // expected-error {{redefinition of default argument}}
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
  template<typename T> void f(T) { A<T> a; } // expected-note {{implicit instantiation}}
  template void f(int);
  template<> struct A<int> {}; // expected-error {{explicit specialization of 'dr3::A<int>' after instantiation}}
}

namespace dr4 { // dr4: yes
  extern "C" {
    static void dr4_f(int) {}
    static void dr4_f(float) {}
    void dr4_g(int) {} // expected-note {{previous}}
    void dr4_g(float) {} // expected-error {{conflicting types}}
  }
}

namespace dr5 { // dr5: yes
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

namespace dr7 { // dr7: yes
  class A { public: ~A(); };
  class B : virtual private A {}; // expected-note 2 {{declared private here}}
  class C : public B {} c; // expected-error 2 {{inherited virtual base class 'dr7::A' has private destructor}} \
                           // expected-note {{implicit default constructor for 'dr7::C' first required here}} \
                           // expected-note {{implicit destructor for 'dr7::C' first required here}}
  class VeryDerivedC : public B, virtual public A {} vdc;

  class X { ~X(); }; // expected-note {{here}}
  class Y : X { ~Y() {} }; // expected-error {{private destructor}}

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

namespace dr9 { // dr9: yes
  struct B {
  protected:
    int m; // expected-note {{here}}
    friend int R1();
  };
  struct N : protected B { // expected-note 2{{protected}}
    friend int R2();
  } n;
  int R1() { return n.m; } // expected-error {{protected base class}} expected-error {{protected member}}
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
    V v; // expected-error {{unknown type name}}
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

namespace dr14 { // dr14: yes
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

  namespace X { typedef int T; typedef int U; } // expected-note {{candidate}}
  namespace Y { typedef int T; typedef long U; } // expected-note {{candidate}}
  T t; // ok, same type both times
  U u; // expected-error {{ambiguous}}
}

namespace dr15 { // dr15: yes
  template<typename T> void f(int); // expected-note {{previous}}
  template<typename T> void f(int = 0); // expected-error {{default arguments cannot be added}}
}

namespace dr16 { // dr16: yes
  class A { // expected-note {{here}}
    void f(); // expected-note {{here}}
    friend class C;
  };
  class B : A {}; // expected-note 4{{here}}
  class C : B {
    void g() {
      f(); // expected-error {{private member}} expected-error {{private base}}
      A::f(); // expected-error {{private member}} expected-error {{private base}}
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

namespace dr19 { // dr19: yes
  struct A {
    int n; // expected-note {{here}}
  };
  struct B : protected A { // expected-note {{here}}
  };
  struct C : B {} c;
  struct D : B {
    int get1() { return c.n; } // expected-error {{protected member}}
    int get2() { return ((A&)c).n; } // ok, A is an accessible base of B from here
  };
}

namespace dr20 { // dr20: yes
  class X {
  public:
    X();
  private:
    X(const X&); // expected-note {{here}}
  };
  X &f();
  X x = f(); // expected-error {{private}}
}

namespace dr21 { // dr21: yes
  template<typename T> struct A;
  struct X {
    template<typename T = int> friend struct A; // expected-error {{default template argument not permitted on a friend template}}
    template<typename T = int> friend struct B; // expected-error {{default template argument not permitted on a friend template}}
  };
}

namespace dr22 { // dr22: sup 481
  template<typename dr22_T = dr22_T> struct X; // expected-error {{unknown type name 'dr22_T'}}
  typedef int T;
  template<typename T = T> struct Y;
}

namespace dr23 { // dr23: yes
  template<typename T> void f(T, T); // expected-note {{candidate}}
  template<typename T> void f(T, int); // expected-note {{candidate}}
  void g() { f(0, 0); } // expected-error {{ambiguous}}
}

// dr24: na

namespace dr25 { // dr25: yes
  struct A {
    void f() throw(int); // expected-error 0-1{{ISO C++1z does not allow}} expected-note 0-1{{use 'noexcept}}
  };
  void (A::*f)() throw (int); // expected-error 0-1{{ISO C++1z does not allow}} expected-note 0-1{{use 'noexcept}}
  void (A::*g)() throw () = f;
#if __cplusplus <= 201402L
  // expected-error@-2 {{is not superset of source}}
#else
  // expected-error@-4 {{different exception specifications}}
#endif
  void (A::*g2)() throw () = 0;
  void (A::*h)() throw (int, char) = f; // expected-error 0-1{{ISO C++1z does not allow}} expected-note 0-1{{use 'noexcept}}
  void (A::*i)() throw () = &A::f;
#if __cplusplus <= 201402L
  // expected-error@-2 {{is not superset of source}}
#else
  // expected-error@-4 {{different exception specifications}}
#endif
  void (A::*i2)() throw () = 0;
  void (A::*j)() throw (int, char) = &A::f; // expected-error 0-1{{ISO C++1z does not allow}} expected-note 0-1{{use 'noexcept}}
  void x() {
    g2 = f;
#if __cplusplus <= 201402L
  // expected-error@-2 {{is not superset of source}}
#else
  // expected-error@-4 {{different exception specifications}}
#endif
    h = f;
    i2 = &A::f;
#if __cplusplus <= 201402L
  // expected-error@-2 {{is not superset of source}}
#else
  // expected-error@-4 {{different exception specifications}}
#endif
    j = &A::f;
  }
}

namespace dr26 { // dr26: yes
  struct A { A(A, const A & = A()); }; // expected-error {{must pass its first argument by reference}}
  struct B {
    B(); // expected-note 0-1{{candidate}}
    B(const B &, B = B());
#if __cplusplus <= 201402L
    // expected-error@-2 {{no matching constructor}} expected-note@-2 {{candidate}} expected-note@-2 {{here}}
#endif
  };
  struct C {
    static C &f();
    C(const C &, C = f()); // expected-error {{no matching constructor}} expected-note {{candidate}} expected-note {{here}}
  };
}

namespace dr27 { // dr27: yes
  enum E { e } n;
  E &m = true ? n : n;
}

// dr28: na lib

namespace dr29 { // dr29: 3.4
  void dr29_f0(); // expected-note {{here}}
  void g0() { void dr29_f0(); }
  extern "C++" void g0_cxx() { void dr29_f0(); }
  extern "C" void g0_c() { void dr29_f0(); } // expected-error {{different language linkage}}

  extern "C" void dr29_f1(); // expected-note {{here}}
  void g1() { void dr29_f1(); }
  extern "C" void g1_c() { void dr29_f1(); }
  extern "C++" void g1_cxx() { void dr29_f1(); } // expected-error {{different language linkage}}

  void g2() { void dr29_f2(); } // expected-note {{here}}
  extern "C" void dr29_f2(); // expected-error {{different language linkage}}

  extern "C" void g3() { void dr29_f3(); } // expected-note {{here}}
  extern "C++" void dr29_f3(); // expected-error {{different language linkage}}

  extern "C++" void g4() { void dr29_f4(); } // expected-note {{here}}
  extern "C" void dr29_f4(); // expected-error {{different language linkage}}

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
  extern "C++" void dr29_f7(); // expected-note {{here}}
  extern "C" void g7() {
    void dr29_f7(); // expected-error {{different language linkage}}
  }

  extern "C++" void g8();
  extern "C" void dr29_f8(); // expected-note {{here}}
  extern "C++" void g8() {
    void dr29_f8(); // expected-error {{different language linkage}}
  }
}

namespace dr30 { // dr30: sup 468 c++11
  struct A {
    template<int> static int f();
  } a, *p = &a;
  int x = A::template f<0>();
  int y = a.template f<0>();
  int z = p->template f<0>();
#if __cplusplus < 201103L
  // FIXME: It's not clear whether DR468 applies to C++98 too.
  // expected-error@-5 {{'template' keyword outside of a template}}
  // expected-error@-5 {{'template' keyword outside of a template}}
  // expected-error@-5 {{'template' keyword outside of a template}}
#endif
}

namespace dr31 { // dr31: yes
  class X {
  private:
    void operator delete(void*); // expected-note {{here}}
  };
  // We would call X::operator delete if X() threw (even though it can't,
  // and even though we allocated the X using ::operator delete).
  X *p = new X; // expected-error {{private}}
}

// dr32: na

namespace dr33 { // dr33: yes
  namespace X { struct S; void f(void (*)(S)); } // expected-note {{candidate}}
  namespace Y { struct T; void f(void (*)(T)); } // expected-note {{candidate}}
  void g(X::S);
  template<typename Z> Z g(Y::T);
  void h() { f(&g); } // expected-error {{ambiguous}}
}

// dr34: na
// dr35: dup 178
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
      int &x(int); // expected-note {{found}}
      static int &y(int); // expected-note {{found}}
    };
    struct V {
      int &z(int);
    };
    struct B : A, virtual V {
      using A::x; // expected-note {{found}}
      float &x(float);
      using A::y; // expected-note {{found}}
      static float &y(float);
      using V::z;
      float &z(float);
    };
    struct C : A, B, virtual V {} c; // expected-warning {{direct base 'dr39::example2::A' is inaccessible due to ambiguity:\n    struct dr39::example2::C -> struct dr39::example2::A\n    struct dr39::example2::C -> struct dr39::example2::B -> struct dr39::example2::A}}
    int &x = c.x(0); // expected-error {{found in multiple base classes}}
    // FIXME: This is valid, because we find the same static data member either way.
    int &y = c.y(0); // expected-error {{found in multiple base classes}}
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
    struct A { int n; }; // expected-note {{found}}
    struct B : A {};
    struct C : A {};
    struct D : B, C { int f() { return n; } }; // expected-error {{found in multiple base-class}}
  }

  namespace PR5916 {
    // FIXME: This is valid.
    struct A { int n; }; // expected-note +{{found}}
    struct B : A {};
    struct C : A {};
    struct D : B, C {};
    int k = sizeof(D::n); // expected-error {{found in multiple base}} expected-error {{unknown type name}}
#if __cplusplus >= 201103L
    decltype(D::n) n; // expected-error {{found in multiple base}}
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

namespace dr44 { // dr44: yes
  struct A {
    template<int> void f();
    template<> void f<0>(); // expected-error {{explicit specialization of 'f' in class scope}}
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
  template template struct A<int>::B<int>; // expected-error {{expected unqualified-id}}
}

namespace dr47 { // dr47: sup 329
  template<typename T> struct A {
    friend void f() { T t; } // expected-error {{redefinition}} expected-note {{previous}}
  };
  A<int> a;
  A<float> b; // expected-note {{instantiation of}}

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

namespace dr49 { // dr49: yes
  template<int*> struct A {}; // expected-note 0-2{{here}}
  int k;
#if __has_feature(cxx_constexpr)
  constexpr
#endif
  int *const p = &k; // expected-note 0-2{{here}}
  A<&k> a;
  A<p> b;
#if __cplusplus <= 201402L
  // expected-error@-2 {{must have its address taken}}
#endif
#if __cplusplus < 201103L
  // expected-error@-5 {{internal linkage}}
#endif
  int *q = &k;
  A<q> c;
#if __cplusplus < 201103L
  // expected-error@-2 {{must have its address taken}}
#else
  // expected-error@-4 {{constant expression}}
  // expected-note@-5 {{read of non-constexpr}}
  // expected-note@-7 {{declared here}}
#endif
}

namespace dr50 { // dr50: yes
  struct X; // expected-note {{forward}}
  extern X *p;
  X *q = (X*)p;
  X *r = static_cast<X*>(p);
  X *s = const_cast<X*>(p);
  X *t = reinterpret_cast<X*>(p);
  X *u = dynamic_cast<X*>(p); // expected-error {{incomplete}}
}

namespace dr51 { // dr51: yes
  struct A {};
  struct B : A {};
  struct S {
    operator A&();
    operator B&();
  } s;
  A &a = s;
}

namespace dr52 { // dr52: yes
  struct A { int n; }; // expected-note {{here}}
  struct B : private A {} b; // expected-note 2{{private}}
  // FIXME: This first diagnostic is very strangely worded, and seems to be bogus.
  int k = b.A::n; // expected-error {{'A' is a private member of 'dr52::A'}}
  // expected-error@-1 {{cannot cast 'struct B' to its private base}}
}

namespace dr53 { // dr53: yes
  int n = 0;
  enum E { e } x = static_cast<E>(n);
}

namespace dr54 { // dr54: yes
  struct A { int a; } a;
  struct V { int v; } v;
  struct B : private A, virtual V { int b; } b; // expected-note 6{{private here}}

  A &sab = static_cast<A&>(b); // expected-error {{private base}}
  A *spab = static_cast<A*>(&b); // expected-error {{private base}}
  int A::*smab = static_cast<int A::*>(&B::b); // expected-error {{private base}}
  B &sba = static_cast<B&>(a); // expected-error {{private base}}
  B *spba = static_cast<B*>(&a); // expected-error {{private base}}
  int B::*smba = static_cast<int B::*>(&A::a); // expected-error {{private base}}

  V &svb = static_cast<V&>(b);
  V *spvb = static_cast<V*>(&b);
  int V::*smvb = static_cast<int V::*>(&B::b); // expected-error {{virtual base}}
  B &sbv = static_cast<B&>(v); // expected-error {{virtual base}}
  B *spbv = static_cast<B*>(&v); // expected-error {{virtual base}}
  int B::*smbv = static_cast<int B::*>(&V::v); // expected-error {{virtual base}}

  A &cab = (A&)(b);
  A *cpab = (A*)(&b);
  int A::*cmab = (int A::*)(&B::b);
  B &cba = (B&)(a);
  B *cpba = (B*)(&a);
  int B::*cmba = (int B::*)(&A::a);

  V &cvb = (V&)(b);
  V *cpvb = (V*)(&b);
  int V::*cmvb = (int V::*)(&B::b); // expected-error {{virtual base}}
  B &cbv = (B&)(v); // expected-error {{virtual base}}
  B *cpbv = (B*)(&v); // expected-error {{virtual base}}
  int B::*cmbv = (int B::*)(&V::v); // expected-error {{virtual base}}
}

namespace dr55 { // dr55: yes
  enum E { e = 5 };
  int test[(e + 1 == 6) ? 1 : -1];
}

namespace dr56 { // dr56: yes
  struct A {
    typedef int T; // expected-note {{previous}}
    typedef int T; // expected-error {{redefinition}}
  };
  struct B {
    struct X;
    typedef X X; // expected-note {{previous}}
    typedef X X; // expected-error {{redefinition}}
  };
}

namespace dr58 { // dr58: yes
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
  template<typename T> struct convert_to { operator T() const; };
  struct A {}; // expected-note 5+{{candidate}}
  struct B : A {}; // expected-note 0+{{candidate}}

  A a1 = convert_to<A>();
  A a2 = convert_to<A&>();
  A a3 = convert_to<const A>();
  A a4 = convert_to<const volatile A>();
#if __cplusplus <= 201402L
  // expected-error@-2 {{no viable}}
#endif
  A a5 = convert_to<const volatile A&>(); // expected-error {{no viable}}

  B b1 = convert_to<B>();
  B b2 = convert_to<B&>();
  B b3 = convert_to<const B>();
  B b4 = convert_to<const volatile B>();
#if __cplusplus <= 201402L
  // expected-error@-2 {{no viable}}
#endif
  B b5 = convert_to<const volatile B&>(); // expected-error {{no viable}}

  A c1 = convert_to<B>();
  A c2 = convert_to<B&>();
  A c3 = convert_to<const B>();
  A c4 = convert_to<const volatile B>(); // expected-error {{no viable}}
  A c5 = convert_to<const volatile B&>(); // expected-error {{no viable}}

  int n1 = convert_to<int>();
  int n2 = convert_to<int&>();
  int n3 = convert_to<const int>();
  int n4 = convert_to<const volatile int>();
  int n5 = convert_to<const volatile int&>();
}

namespace dr60 { // dr60: yes
  void f(int &);
  int &f(...);
  const int k = 0;
  int &n = f(k);
}

namespace dr61 { // dr61: yes
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
  void (*q)() = &y.f; // expected-error {{cannot create a non-constant pointer to member function}}
  void (*r)() = y.f; // expected-error {{cannot create a non-constant pointer to member function}}
}

namespace dr62 { // dr62: yes
  struct A {
    struct { int n; } b;
  };
  template<typename T> struct X {};
  template<typename T> T get() { return get<T>(); }
  template<typename T> int take(T) { return 0; }

  X<A> x1;
  A a = get<A>();

  typedef struct { } *NoNameForLinkagePtr;
#if __cplusplus < 201103L
  // expected-note@-2 5{{here}}
#endif
  NoNameForLinkagePtr noNameForLinkagePtr;

  struct Danger {
    NoNameForLinkagePtr p;
  };

  X<NoNameForLinkagePtr> x2;
  X<const NoNameForLinkagePtr> x3;
  NoNameForLinkagePtr p1 = get<NoNameForLinkagePtr>();
  NoNameForLinkagePtr p2 = get<const NoNameForLinkagePtr>();
  int n1 = take(noNameForLinkagePtr);
#if __cplusplus < 201103L
  // expected-error@-6 {{uses unnamed type}}
  // expected-error@-6 {{uses unnamed type}}
  // expected-error@-6 {{uses unnamed type}}
  // expected-error@-6 {{uses unnamed type}}
  // expected-error@-6 {{uses unnamed type}}
#endif

  X<Danger> x4;

  void f() {
    struct NoLinkage {};
    X<NoLinkage> a;
    X<const NoLinkage> b;
    get<NoLinkage>();
    get<const NoLinkage>();
    X<void (*)(NoLinkage A::*)> c;
    X<int NoLinkage::*> d;
#if __cplusplus < 201103L
  // expected-error@-7 {{uses local type}}
  // expected-error@-7 {{uses local type}}
  // expected-error@-7 {{uses local type}}
  // expected-error@-7 {{uses local type}}
  // expected-error@-7 {{uses local type}}
  // expected-error@-7 {{uses local type}}
#endif
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
    int f(int n); // expected-note 2{{candidate}}
  }
  using X::f;
  namespace X {
    int f(int n = 0);
    int f(int, int);
  }
  // FIXME: The first two calls here should be accepted.
  int a = f(); // expected-error {{no matching function}}
  int b = f(1);
  int c = f(1, 2); // expected-error {{no matching function}}
}

// dr67: na

namespace dr68 { // dr68: yes
  template<typename T> struct X {};
  struct ::dr68::X<int> x1;
  struct ::dr68::template X<int> x2;
#if __cplusplus < 201103L
  // expected-error@-2 {{'template' keyword outside of a template}}
#endif
  struct Y {
    friend struct X<int>;
    friend struct ::dr68::X<char>;
    friend struct ::dr68::template X<double>;
#if __cplusplus < 201103L
  // expected-error@-2 {{'template' keyword outside of a template}}
#endif
  };
  template<typename>
  struct Z {
    friend struct ::dr68::template X<double>;
    friend typename ::dr68::X<double>;
#if __cplusplus < 201103L
  // expected-error@-2 {{C++11 extension}}
#endif
  };
}

namespace dr69 { // dr69: yes
  template<typename T> static void f() {}
  // FIXME: Should we warn here?
  inline void g() { f<int>(); }
  // FIXME: This should be rejected, per [temp.explicit]p11.
  extern template void f<char>();
#if __cplusplus < 201103L
  // expected-error@-2 {{C++11 extension}}
#endif
  template<void(*)()> struct Q {};
  Q<&f<int> > q;
#if __cplusplus < 201103L
  // expected-error@-2 {{internal linkage}} expected-note@-11 {{here}}
#endif
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
namespace dr73 { // dr73: no
  // The resolution to dr73 is unworkable. Consider:
  int a, b;
  static_assert(&a + 1 != &b, ""); // expected-error {{not an integral constant expression}}
}
#endif

namespace dr74 { // dr74: yes
  enum E { k = 5 };
  int (*p)[k] = new int[k][k];
}

namespace dr75 { // dr75: yes
  struct S {
    static int n = 0; // expected-error {{non-const}}
  };
}

namespace dr76 { // dr76: yes
  const volatile int n = 1;
  int arr[n]; // expected-error +{{variable length array}}
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
  const int k; // expected-error {{default initialization of an object of const}}
}

// dr79: na

namespace dr80 { // dr80: yes
  struct A {
    int A;
  };
  struct B {
    static int B; // expected-error {{same name as its class}}
  };
  struct C {
    int C; // expected-note {{hidden by}}
    // FIXME: These diagnostics aren't very good.
    C(); // expected-error {{must use 'struct' tag to refer to}} expected-error {{expected member name}}
  };
  struct D {
    D();
    int D; // expected-error {{same name as its class}}
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
    B(B&); // expected-note 0-1{{candidate}}
    B(C); // expected-note 0-1{{no known conversion from 'dr84::B' to 'dr84::C'}}
    operator C() const;
  };
  A a;
  // Cannot use B(C) / operator C() pair to construct the B from the B temporary
  // here. In C++1z, we initialize the B object directly using 'A::operator B()'.
  B b = a;
#if __cplusplus <= 201402L
  // expected-error@-2 {{no viable}}
#endif
}

namespace dr85 { // dr85: yes
  struct A {
    struct B;
    struct B {}; // expected-note{{previous declaration is here}}
    struct B; // expected-error{{class member cannot be redeclared}}

    union U;
    union U {}; // expected-note{{previous declaration is here}}
    union U; // expected-error{{class member cannot be redeclared}}

#if __cplusplus >= 201103L
    enum E1 : int;
    enum E1 : int { e1 }; // expected-note{{previous declaration is here}}
    enum E1 : int; // expected-error{{class member cannot be redeclared}}

    enum class E2;
    enum class E2 { e2 }; // expected-note{{previous declaration is here}}
    enum class E2; // expected-error{{class member cannot be redeclared}}
#endif
  };

  template <typename T>
  struct C {
    struct B {}; // expected-note{{previous declaration is here}}
    struct B; // expected-error{{class member cannot be redeclared}}
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

namespace dr88 { // dr88: yes
  template<typename T> struct S {
    static const int a = 1; // expected-note {{previous}}
    static const int b;
  };
  template<> const int S<int>::a = 4; // expected-error {{already has an initializer}}
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
    dr90_f(B::C()); // expected-error {{undeclared identifier}}
    dr90_f(B::D()); // expected-error {{undeclared identifier}}
    dr90_f(E());
    dr90_f(F()); // expected-error {{undeclared identifier}}

    dr90_g(A()); // expected-error {{undeclared identifier}}
    dr90_g(B());
    dr90_g(B::C());
    dr90_g(B::D());
    dr90_g(E());
    dr90_g(F()); // expected-error {{undeclared identifier}}
  }
}

namespace dr91 { // dr91: yes
  union U { friend int f(U); };
  int k = f(U());
}

namespace dr92 { // dr92: 4 c++17
  void f() throw(int, float); // expected-error 0-1{{ISO C++1z does not allow}} expected-note 0-1{{use 'noexcept}}
  void (*p)() throw(int) = &f; // expected-error 0-1{{ISO C++1z does not allow}} expected-note 0-1{{use 'noexcept}}
#if __cplusplus <= 201402L
  // expected-error@-2 {{target exception specification is not superset of source}}
#else
  // expected-warning@-4 {{target exception specification is not superset of source}}
#endif
  void (*q)() throw(int); // expected-error 0-1{{ISO C++1z does not allow}} expected-note 0-1{{use 'noexcept}}
  void (**pp)() throw() = &q;
#if __cplusplus <= 201402L
  // expected-error@-2 {{exception specifications are not allowed}}
#else
  // expected-error@-4 {{cannot initialize}}
#endif

  void g(void() throw()); // expected-note 0-2 {{no known conversion}} expected-warning 0-1{{mangled name of 'g' will change in C++17}}
  void h() throw() {
    g(f); // expected-error-re {{{{is not superset|no matching function}}}}
    g(q); // expected-error-re {{{{is not superset|no matching function}}}}
  }

  // Prior to C++17, this is OK because the exception specification is not
  // considered in this context. In C++17, we *do* perform an implicit
  // conversion (which performs initialization), and the exception specification
  // is part of the type of the parameter, so this is invalid.
  template<void() throw()> struct X {};
  X<&f> xp;
#if __cplusplus > 201402L
  // expected-error@-2 {{not implicitly convertible}}
#endif

  template<void() throw(int)> struct Y {}; // expected-error 0-1{{ISO C++1z does not allow}} expected-note 0-1{{use 'noexcept}}
  Y<&h> yp; // ok
}

// dr93: na

namespace dr94 { // dr94: yes
  struct A { static const int n = 5; };
  int arr[A::n];
}

namespace dr95 { // dr95: yes
  struct A;
  struct B;
  namespace N {
    class C {
      friend struct A;
      friend struct B;
      static void f(); // expected-note {{here}}
    };
    struct A *p; // dr95::A, not dr95::N::A.
  }
  A *q = N::p; // ok, same type
  struct B { void f() { N::C::f(); } }; // expected-error {{private}}
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
      try { // expected-note 2{{bypasses}}
        case 0: // expected-error {{cannot jump}}
        x:
          throw n;
      } catch (...) { // expected-note 2{{bypasses}}
        case 1: // expected-error {{cannot jump}}
        y:
          throw n;
      }
      case 2:
        goto x; // expected-error {{cannot jump}}
      case 3:
        goto y; // expected-error {{cannot jump}}
    }
  }
}

namespace dr99 { // dr99: sup 214
  template<typename T> void f(T&);
  template<typename T> int &f(const T&);
  const int n = 0;
  int &r = f(n);
}
