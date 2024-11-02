// RUN: %clang_cc1 -std=c++2b -verify=expected,cxx20_2b,cxx2b    -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -verify=expected,cxx98_20,cxx20_2b -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -verify=expected,cxx98_17,cxx98_20 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -verify=expected,cxx98_17,cxx98_20 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -verify=expected,cxx98_17,cxx98_20 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++98 -verify=expected,cxx98_17,cxx98_20 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr300 { // dr300: yes
  template<typename R, typename A> void f(R (&)(A)) {}
  int g(int);
  void h() { f(g); }
}

namespace dr301 { // dr301: yes
  // see also dr38
  struct S;
  template<typename T> void operator+(T, T);
  void operator-(S, S);

  void f() {
    bool a = (void(*)(S, S))operator+<S> < // expected-warning {{ordered comparison of function pointers}}
             (void(*)(S, S))operator+<S>;
    bool b = (void(*)(S, S))operator- < // cxx20_2b-note {{to match this '<'}} cxx98_17-warning {{ordered comparison of function pointers}}
             (void(*)(S, S))operator-; // cxx20_2b-error {{expected '>'}}
    bool c = (void(*)(S, S))operator+ < // expected-note {{to match this '<'}}
             (void(*)(S, S))operator-; // expected-error {{expected '>'}}
  }

  template<typename T> void f() {
    typename T::template operator+<int> a; // expected-error {{typename specifier refers to a non-type template}} expected-error +{{}}
    // FIXME: This shouldn't say (null).
    class T::template operator+<int> b; // expected-error {{identifier followed by '<' indicates a class template specialization but (null) refers to a function template}}
    enum T::template operator+<int> c; // expected-error {{expected identifier}}
    enum T::template operator+<int>::E d; // expected-error {{qualified name refers into a specialization of function template 'T::template operator +'}} expected-error {{forward reference}}
    enum T::template X<int>::E e;
    T::template operator+<int>::foobar(); // expected-error {{qualified name refers into a specialization of function template 'T::template operator +'}}
    T::template operator+<int>(0); // ok
  }

  template<typename T> class operator&<T*> {}; // expected-error +{{}}
  template<typename T> class T::operator& {}; // expected-error +{{}}
  template<typename T> class S::operator&<T*> {}; // expected-error +{{}}
}

namespace dr302 { // dr302: yes
  struct A { A(); ~A(); };
#if __cplusplus < 201103L
  struct B { // expected-error {{implicit default constructor for 'dr302::B' must explicitly initialize the const member 'n'}}
    const int n; // expected-note {{declared here}}
    A a;
  } b = B(); // expected-note {{first required here}}
  // Trivial default constructor C::C() is not called here.
  struct C {
    const int n;
  } c = C();
#else
  struct B {
    const int n; // expected-note {{deleted because field 'n' of const-qualified type 'const int' would not be initialized}}
    A a;
  } b = B(); // expected-error {{call to implicitly-deleted default constructor}}
  // C::C() is called here, because even though it's trivial, it's deleted.
  struct C {
    const int n; // expected-note {{deleted because field 'n' of const-qualified type 'const int' would not be initialized}}
  } c = C(); // expected-error {{call to implicitly-deleted default constructor}}
  struct D {
    const int n = 0;
  } d = D();
#endif
}

// dr303: na

namespace dr304 { // dr304: yes
  typedef int &a;
  int n = a(); // expected-error {{requires an initializer}}

  struct S { int &b; };
  int m = S().b;
#if __cplusplus < 201103L
  // expected-error@-3 {{requires an initializer}}
  // expected-note@-3 {{in value-initialization}}
#else
  // expected-error@-5 {{deleted}}
  // expected-note@-7 {{reference}}
#endif
}

namespace dr305 { // dr305: no
  struct A {
    typedef A C;
  };
  void f(A *a) {
    struct A {};
    a->~A();
    a->~C();
  }
  typedef A B;
  void g(B *b) {
    b->~B();
    b->~C();
  }
  void h(B *b) {
    struct B {}; // expected-note {{type 'B' found by destructor name lookup}}
    b->~B(); // expected-error {{does not match}}
  }

  template<typename T> struct X {};
  void i(X<int>* x) {
    struct X {};
    x->~X<int>();
    x->~X();
    x->~X<char>(); // expected-error {{no member named}}
  }

#if __cplusplus >= 201103L
  struct Y {
    template<typename T> using T1 = Y;
  };
  template<typename T> using T2 = Y;
  void j(Y *y) {
    y->~T1<int>();
    y->~T2<int>();
  }
  struct Z {
    template<typename T> using T2 = T;
  };
  void k(Z *z) {
    z->~T1<int>(); // expected-error {{no member named 'T1' in 'dr305::Z'}}
    z->~T2<int>(); // expected-error {{no member named '~int'}}
    z->~T2<Z>();
  }

  // FIXME: This is valid.
  namespace Q {
    template<typename A> struct R {};
  }
  template<typename A> using R = Q::R<int>;
  void qr(Q::R<int> x) { x.~R<char>(); } // expected-error {{no member named}}
#endif
}

namespace dr306 { // dr306: dup 39
  struct A { struct B {}; };
  struct C { typedef A::B B; };
  struct D : A, A::B, C {};
  D::B b;

  struct X {}; // expected-note {{member type 'dr306::X' found}}
  template<typename T> struct Y { typedef T X; }; // expected-note {{member type 'const dr306::X' found}}
  template<typename T> struct Z : X, Y<T> {};
  Z<X>::X zx;
  Z<const X>::X zcx; // expected-error {{member 'X' found in multiple base classes of different types}}
}

// dr307: na

namespace dr308 { // dr308: yes
  // This is mostly an ABI library issue.
  struct A {};
  struct B : A {};
  struct C : A {};
  struct D : B, C {};
  void f() {
    try {
      throw D();
    } catch (const A&) { // expected-note {{for type 'const A &'}}
      // unreachable
    } catch (const B&) { // expected-warning {{exception of type 'const B &' will be caught by earlier handler}}
      // get here instead
    }
  }
}

// dr309: dup 485

namespace dr311 { // dr311: yes
  namespace X { namespace Y {} }
  namespace X::Y {}
#if __cplusplus <= 201402L
  // expected-error@-2 {{define each namespace separately}}
#endif
  namespace X {
    namespace X::Y {}
#if __cplusplus <= 201402L
  // expected-error@-2 {{define each namespace separately}}
#endif
  }
  // FIXME: The diagnostics here are not very good.
  namespace ::dr311::X {} // expected-error 2+{{}} // expected-warning {{extra qual}}
}

// dr312: dup 616

namespace dr313 { // dr313: dup 299 c++11
  struct A { operator int() const; };
  int *p = new int[A()];
#if __cplusplus < 201103L
  // FIXME: should this be available in c++98 mode? expected-error@-2 {{extension}}
#endif
}

namespace dr314 { // FIXME 314: dup 1710
  template<typename T> struct A {
    template<typename U> struct B {};
  };
  template<typename T> struct C : public A<T>::template B<T> {
    C() : A<T>::template B<T>() {}
  };
}

// dr315: na
// dr316: sup 1004

namespace dr317 { // dr317: 3.5
  void f() {} // expected-note {{previous}}
  inline void f(); // expected-error {{inline declaration of 'f' follows non-inline definition}}

  int g();
  int n = g();
  inline int g() { return 0; }

  int h();
  int m = h();
  int h() { return 0; } // expected-note {{previous}}
  inline int h(); // expected-error {{inline declaration of 'h' follows non-inline definition}}
}

namespace dr318 { // dr318: sup 1310
  struct A {};
  struct A::A a;
}

namespace dr319 { // dr319: no
  // FIXME: dup dr389
  // FIXME: We don't have a diagnostic for a name with linkage
  //        having a type without linkage.
  typedef struct {
    int i;
  } *ps;
  extern "C" void f(ps);
  void g(ps); // FIXME: ill-formed, type 'ps' has no linkage

  static enum { e } a1;
  enum { e2 } a2; // FIXME: ill-formed, enum type has no linkage

  enum { n1 = 1u };
  typedef int (*pa)[n1];
  pa parr; // ok, type has linkage despite using 'n1'

  template<typename> struct X {};

  void f() {
    struct A { int n; };
    extern A a; // FIXME: ill-formed
    X<A> xa;

    typedef A B;
    extern B b; // FIXME: ill-formed
    X<B> xb;

    const int n = 1;
    typedef int (*C)[n];
    extern C c; // ok
    X<C> xc;
  }
#if __cplusplus < 201103L
  // expected-error@-12 {{uses local type 'A'}}
  // expected-error@-9 {{uses local type 'A'}}
#endif
}

namespace dr320 { // dr320: yes
#if __cplusplus >= 201103L
  struct X {
    constexpr X() {}
    constexpr X(const X &x) : copies(x.copies + 1) {}
    unsigned copies = 0;
  };
  constexpr X f(X x) { return x; }
  constexpr unsigned g(X x) { return x.copies; }
  static_assert(f(X()).copies == g(X()) + 1, "expected one extra copy for return value");
#endif
}

namespace dr321 { // dr321: dup 557
  namespace N {
    template<int> struct A {
      template<int> struct B;
    };
    template<> template<> struct A<0>::B<0>;
    void f(A<0>::B<0>);
  }
  template<> template<> struct N::A<0>::B<0> {};

  template<typename T> void g(T t) { f(t); }
  template void g(N::A<0>::B<0>);

  namespace N {
    template<typename> struct I { friend bool operator==(const I&, const I&); };
  }
  N::I<int> i, j;
  bool x = i == j;
}

namespace dr322 { // dr322: yes
  struct A {
    template<typename T> operator T&();
  } a;
  int &r = static_cast<int&>(a);
  int &s = a;
}

// dr323: no

namespace dr324 { // dr324: yes
  struct S { int n : 1; } s; // expected-note 3{{bit-field is declared here}}
  int &a = s.n; // expected-error {{non-const reference cannot bind to bit-field}}
  int *b = &s.n; // expected-error {{address of bit-field}}
  int &c = (s.n = 0); // expected-error {{non-const reference cannot bind to bit-field}}
  int *d = &(s.n = 0); // expected-error {{address of bit-field}}
  int &e = true ? s.n : s.n; // expected-error {{non-const reference cannot bind to bit-field}}
  int *f = &(true ? s.n : s.n); // expected-error {{address of bit-field}}
  int &g = (void(), s.n); // expected-error {{non-const reference cannot bind to bit-field}}
  int *h = &(void(), s.n); // expected-error {{address of bit-field}}
  int *i = &++s.n; // expected-error {{address of bit-field}}
}

namespace dr326 { // dr326: yes
  struct S {};
  int test[__is_trivially_constructible(S, const S&) ? 1 : -1];
}

namespace dr327 { // dr327: dup 538
  struct A;
  class A {};

  class B;
  struct B {};
}

namespace dr328 { // dr328: yes
  struct A; // expected-note 3{{forward declaration}}
  struct B { A a; }; // expected-error {{incomplete}}
  template<typename> struct C { A a; }; // expected-error {{incomplete}}
  A *p = new A[0]; // expected-error {{incomplete}}
}

namespace dr329 { // dr329: 3.5
  struct B {};
  template<typename T> struct A : B {
    friend void f(A a) { g(a); }
    friend void h(A a) { g(a); } // expected-error {{undeclared}}
    friend void i(B b) {} // expected-error {{redefinition}} expected-note {{previous}}
  };
  A<int> a;
  A<char> b; // expected-note {{instantiation}}

  void test() {
    h(a); // expected-note {{instantiation}}
  }
}

namespace dr330 { // dr330: 7
  // Conversions between P and Q will be allowed by P0388.
  typedef int *(*P)[3];
  typedef const int *const (*Q)[3];
  typedef const int *Qinner[3];
  typedef Qinner const *Q2; // same as Q, but 'const' written outside the array type
  typedef const int *const (*R)[4];
  typedef const int *const (*S)[];
  typedef const int *(*T)[];
  void f(P p, Q q, Q2 q2, R r, S s, T t) {
    q = p; // ok
    q2 = p; // ok
    r = p; // expected-error {{incompatible}}
    s = p;
#if __cplusplus < 202002
    // expected-error@-2 {{incompatible}} (fixed by p0388)
#endif
    t = p; // expected-error {{incompatible}}
    s = q;
#if __cplusplus < 202002
    // expected-error@-2 {{incompatible}} (fixed by p0388)
#endif
    s = q2;
#if __cplusplus < 202002
    // expected-error@-2 {{incompatible}} (fixed by p0388)
#endif
    s = t; // ok, adding const
    t = s; // expected-error {{discards qualifiers}}
    (void) const_cast<P>(q);
    (void) const_cast<P>(q2);
    (void) const_cast<Q>(p);
    (void) const_cast<Q2>(p);
    (void) const_cast<S>(p); // expected-error {{not allowed}} (for now)
    (void) const_cast<P>(s); // expected-error {{not allowed}} (for now)
    (void) const_cast<S>(q); // expected-error {{not allowed}}
    (void) const_cast<S>(q2); // expected-error {{not allowed}}
    (void) const_cast<Q>(s); // expected-error {{not allowed}}
    (void) const_cast<Q2>(s); // expected-error {{not allowed}}
    (void) const_cast<T>(s);
    (void) const_cast<S>(t);
    (void) const_cast<T>(q); // expected-error {{not allowed}}
    (void) const_cast<Q>(t); // expected-error {{not allowed}}

    (void) reinterpret_cast<P>(q); // expected-error {{casts away qualifiers}}
    (void) reinterpret_cast<P>(q2); // expected-error {{casts away qualifiers}}
    (void) reinterpret_cast<Q>(p);
    (void) reinterpret_cast<Q2>(p);
    (void) reinterpret_cast<S>(p);
    (void) reinterpret_cast<P>(s); // expected-error {{casts away qualifiers}}
    (void) reinterpret_cast<S>(q);
    (void) reinterpret_cast<S>(q2);
    (void) reinterpret_cast<Q>(s);
    (void) reinterpret_cast<Q2>(s);
    (void) reinterpret_cast<T>(s); // expected-error {{casts away qualifiers}}
    (void) reinterpret_cast<S>(t);
    (void) reinterpret_cast<T>(q); // expected-error {{casts away qualifiers}}
    (void) reinterpret_cast<Q>(t);
  }

  namespace swift_17882 {
    typedef const char P[72];
    typedef int *Q;
    void f(P &pr, P *pp) {
      (void) reinterpret_cast<const Q&>(pr);
      (void) reinterpret_cast<const Q*>(pp);
    }

    struct X {};
    typedef const volatile int A[1][2][3];
    typedef int *const X::*volatile *B1;
    typedef int *const X::*         *B2;
    typedef int *X::*      volatile *B3;
    typedef volatile int *(*const B4)[4];
    void f(A *a) {
      (void) reinterpret_cast<B1*>(a);
      (void) reinterpret_cast<B2*>(a); // expected-error {{casts away qualifiers}}
      (void) reinterpret_cast<B3*>(a); // expected-error {{casts away qualifiers}}
      (void) reinterpret_cast<B4*>(a);
    }
  }
}

namespace dr331 { // dr331: yes
  struct A {
    A(volatile A&); // expected-note 2{{candidate}}
  } const a, b(a); // expected-error 2{{no matching constructor}}
}

namespace dr332 { // dr332: dup 577
  void f(volatile void); // expected-error {{'void' as parameter must not have type qualifiers}}
  // cxx20_2b-warning@-1 {{volatile-qualified parameter type 'volatile void' is deprecated}}
  void g(const void); // expected-error {{'void' as parameter must not have type qualifiers}}
  void h(int n, volatile void); // expected-error {{'void' must be the first and only parameter}}
  // cxx20_2b-warning@-1 {{volatile-qualified parameter type 'volatile void' is deprecated}}
}

namespace dr333 { // dr333: yes
  int n = 0;
  int f(int(n));
  int g((int(n)));
  int h = f(g);
}

namespace dr334 { // dr334: yes
  template<typename T> void f() {
    T x;
    f((x, 123));
  }
  struct S {
    friend S operator,(S, int);
    friend void f(S);
  };
  template void f<S>();
}

// dr335: no

namespace dr336 { // dr336: yes
  namespace Pre {
    template<class T1> class A {
      template<class T2> class B {
        template<class T3> void mf1(T3);
        void mf2();
      };
    };
    template<> template<class X> class A<int>::B {};
    template<> template<> template<class T> void A<int>::B<double>::mf1(T t) {} // expected-error {{does not match}}
    template<class Y> template<> void A<Y>::B<double>::mf2() {} // expected-error {{does not refer into a class}}
  }
  namespace Post {
    template<class T1> class A {
      template<class T2> class B {
        template<class T3> void mf1(T3);
        void mf2();
      };
    };
    template<> template<class X> class A<int>::B {
      template<class T> void mf1(T);
    };
    template<> template<> template<class T> void A<int>::B<double>::mf1(T t) {}
    // FIXME: This diagnostic isn't very good.
    template<class Y> template<> void A<Y>::B<double>::mf2() {} // expected-error {{does not refer into a class}}
  }
}

namespace dr337 { // dr337: yes
  template<typename T> void f(T (*)[1]);
  template<typename T> int &f(...);

  struct A { virtual ~A() = 0; };
  int &r = f<A>(0);

  // FIXME: The language rules here are completely broken. We cannot determine
  // whether an incomplete type is abstract. See DR1640, which will probably
  // supersede this one and remove this rule.
  struct B;
  int &s = f<B>(0); // expected-error {{of type 'void'}}
  struct B { virtual ~B() = 0; };
}

namespace dr339 { // dr339: yes
  template <int I> struct A { static const int value = I; };

  char xxx(int);
  char (&xxx(float))[2];

  template<class T> A<sizeof(xxx((T)0))> f(T) {} // expected-note {{candidate}}

  void test() {
    A<1> a = f(0);
    A<2> b = f(0.0f);
    A<3> c = f("foo"); // expected-error {{no matching function}}
  }


  char f(int);
  int f(...);

  template <class T> struct conv_int {
    static const bool value = sizeof(f(T())) == 1;
  };

  template <class T> bool conv_int2(A<sizeof(f(T()))> p);

  template<typename T> A<sizeof(f(T()))> make_A();

  int a[conv_int<char>::value ? 1 : -1];
  bool b = conv_int2<char>(A<1>());
  A<1> c = make_A<char>();
}

namespace dr340 { // dr340: yes
  struct A { A(int); };
  struct B { B(A, A, int); };
  int x, y;
  B b(A(x), A(y), 3);
}

namespace dr341 { // dr341: sup 1708
  namespace A {
    int n;
    extern "C" int &dr341_a = n; // expected-note {{previous}} expected-note {{declared with C language linkage here}}
  }
  namespace B {
    extern "C" int &dr341_a = dr341_a; // expected-error {{redefinition}}
  }
  extern "C" void dr341_b(); // expected-note {{declared with C language linkage here}}
}
int dr341_a; // expected-error {{declaration of 'dr341_a' in global scope conflicts with declaration with C language linkage}}
int dr341_b; // expected-error {{declaration of 'dr341_b' in global scope conflicts with declaration with C language linkage}}
int dr341_c; // expected-note {{declared in global scope here}}
int dr341_d; // expected-note {{declared in global scope here}}
namespace dr341 {
  extern "C" int dr341_c; // expected-error {{declaration of 'dr341_c' with C language linkage conflicts with declaration in global scope}}
  extern "C" void dr341_d(); // expected-error {{declaration of 'dr341_d' with C language linkage conflicts with declaration in global scope}}

  namespace A { extern "C" int dr341_e; } // expected-note {{previous}}
  namespace B { extern "C" void dr341_e(); } // expected-error {{redefinition of 'dr341_e' as different kind of symbol}}
}

// dr342: na

namespace dr343 { // FIXME 343: no
  // FIXME: dup 1710
  template<typename T> struct A {
    template<typename U> struct B {};
  };
  // FIXME: In these contexts, the 'template' keyword is optional.
  template<typename T> struct C : public A<T>::B<T> { // expected-error {{use 'template'}}
    C() : A<T>::B<T>() {} // expected-error {{use 'template'}}
  };
}

namespace dr344 { // dr344: dup 1435
  struct A { inline virtual ~A(); };
  struct B { friend A::~A(); };
}

namespace dr345 { // dr345: yes
  struct A {
    struct X {};
    int X; // expected-note {{here}}
  };
  struct B {
    struct X {};
  };
  template <class T> void f(T t) { typename T::X x; } // expected-error {{refers to non-type member 'X'}}
  void f(A a, B b) {
    f(b);
    f(a); // expected-note {{instantiation}}
  }
}

// dr346: na

namespace dr347 { // dr347: yes
  struct base {
    struct nested;
    static int n;
    static void f();
    void g();
  };

  struct derived : base {};

  struct derived::nested {}; // expected-error {{no struct named 'nested'}}
  int derived::n; // expected-error {{no member named 'n'}}
  void derived::f() {} // expected-error {{does not match any}}
  void derived::g() {} // expected-error {{does not match any}}
}

// dr348: na

namespace dr349 { // dr349: no
  struct A {
    template <class T> operator T ***() {
      int ***p = 0;
      return p; // cxx98_20-error {{cannot initialize return object of type 'const int ***' with an lvalue of type 'int ***'}}
      // cxx2b-error@-1 {{cannot initialize return object of type 'const int ***' with an rvalue of type 'int ***'}}
    }
  };

  // FIXME: This is valid.
  A a;
  const int *const *const *p1 = a; // expected-note {{in instantiation of}}

  struct B {
    template <class T> operator T ***() {
      const int ***p = 0;
      return p;
    }
  };

  // FIXME: This is invalid.
  B b;
  const int *const *const *p2 = b;
}

// dr351: na

namespace dr352 { // dr352: yes
  namespace example1 {
    namespace A {
      enum E {};
      template<typename R, typename A> void foo(E, R (*)(A)); // expected-note 2{{couldn't infer template argument 'R'}}
    }

    template<typename T> void arg(T);
    template<typename T> int arg(T) = delete; // expected-note {{here}} expected-error 0-1{{extension}}

    void f(A::E e) {
      foo(e, &arg); // expected-error {{no matching function}}

      using A::foo;
      foo<int, int>(e, &arg); // expected-error {{deleted}}
    }

    int arg(int);

    void g(A::E e) {
      foo(e, &arg); // expected-error {{no matching function}}

      using A::foo;
      foo<int, int>(e, &arg); // ok, uses non-template
    }
  }

  namespace contexts {
    template<int I> void f1(int (&)[I]);
    template<int I> void f2(int (&)[I+1]); // expected-note {{couldn't infer}}
    template<int I> void f3(int (&)[I+1], int (&)[I]);
    void f() {
      int a[4];
      int b[3];
      f1(a);
      f2(a); // expected-error {{no matching function}}
      f3(a, b);
    }

    template<int I> struct S {};
    template<int I> void g1(S<I>);
    template<int I> void g2(S<I+1>); // expected-note {{couldn't infer}}
    template<int I> void g3(S<I+1>, S<I>);
    void g() {
      S<4> a;
      S<3> b;
      g1(a);
      g2(a); // expected-error {{no matching function}}
      g3(a, b);
    }

    template<typename T> void h1(T = 0); // expected-note {{couldn't infer}}
    template<typename T> void h2(T, T = 0);
    void h() {
      h1(); // expected-error {{no matching function}}
      h1(0);
      h1<int>();
      h2(0);
    }

    template<typename T> int tmpl(T);
    template<typename R, typename A> void i1(R (*)(A)); // expected-note 3{{couldn't infer}}
    template<typename R, typename A> void i2(R, A, R (*)(A)); // expected-note {{not viable}}
    void i() {
      extern int single(int);
      i1(single);
      i2(0, 0, single);

      extern int ambig(float), ambig(int);
      i1(ambig); // expected-error {{no matching function}}
      i2(0, 0, ambig);

      extern void no_match(float), no_match(int);
      i1(no_match); // expected-error {{no matching function}}
      i2(0, 0, no_match); // expected-error {{no matching function}}

      i1(tmpl); // expected-error {{no matching function}}
      i2(0, 0, tmpl);
    }
  }

  template<typename T> struct is_int;
  template<> struct is_int<int> {};

  namespace example2 {
    template<typename T> int f(T (*p)(T)) { is_int<T>(); }
    int g(int);
    int g(char);
    int i = f(g);
  }

  namespace example3 {
    template<typename T> int f(T, T (*p)(T)) { is_int<T>(); }
    int g(int);
    char g(char);
    int i = f(1, g);
  }

  namespace example4 {
    template <class T> int f(T, T (*p)(T)) { is_int<T>(); }
    char g(char);
    template <class T> T g(T);
    int i = f(1, g);
  }

  namespace example5 {
    template<int I> class A {};
    template<int I> void g(A<I+1>); // expected-note {{couldn't infer}}
    template<int I> void f(A<I>, A<I+1>);
    void h(A<1> a1, A<2> a2) {
      g(a1); // expected-error {{no matching function}}
      g<0>(a1);
      f(a1, a2);
    }
  }
}

// dr353 needs an IRGen test.

namespace dr354 { // dr354: yes c++11
  // FIXME: Should we allow this in C++98 too?
  struct S {};

  template<int*> struct ptr {}; // expected-note 0-4{{here}}
  ptr<0> p0;
  ptr<(int*)0> p1;
  ptr<(float*)0> p2;
  ptr<(int S::*)0> p3;
#if __cplusplus < 201103L
  // expected-error@-5 {{does not refer to any decl}}
  // expected-error@-5 {{does not refer to any decl}}
  // expected-error@-5 {{does not refer to any decl}}
  // expected-error@-5 {{does not refer to any decl}}
#elif __cplusplus <= 201402L
  // expected-error@-10 {{must be cast}}
  // ok
  // expected-error@-10 {{does not match}}
  // expected-error@-10 {{does not match}}
#else
  // expected-error@-15 {{conversion from 'int' to 'int *' is not allowed}}
  // ok
  // expected-error@-15 {{'float *' is not implicitly convertible to 'int *'}}
  // expected-error@-15 {{'int dr354::S::*' is not implicitly convertible to 'int *'}}
#endif

  template<int*> int both();
  template<int> int both();
  int b0 = both<0>();
  int b1 = both<(int*)0>();
#if __cplusplus < 201103L
  // expected-error@-2 {{no matching function}}
  // expected-note@-6 {{candidate}}
  // expected-note@-6 {{candidate}}
#endif

  template<int S::*> struct ptr_mem {}; // expected-note 0-4{{here}}
  ptr_mem<0> m0;
  ptr_mem<(int S::*)0> m1;
  ptr_mem<(float S::*)0> m2;
  ptr_mem<(int *)0> m3;
#if __cplusplus < 201103L
  // expected-error@-5 {{cannot be converted}}
  // expected-error@-5 {{is not a pointer to member constant}}
  // expected-error@-5 {{cannot be converted}}
  // expected-error@-5 {{cannot be converted}}
#elif __cplusplus <= 201402L
  // expected-error@-10 {{must be cast}}
  // ok
  // expected-error@-10 {{does not match}}
  // expected-error@-10 {{does not match}}
#else
  // expected-error@-15 {{conversion from 'int' to 'int dr354::S::*' is not allowed}}
  // ok
  // expected-error@-15 {{'float dr354::S::*' is not implicitly convertible to 'int dr354::S::*'}}
  // expected-error@-15 {{'int *' is not implicitly convertible to 'int dr354::S::*'}}
#endif
}

struct dr355_S; // dr355: yes
struct ::dr355_S {}; // expected-warning {{extra qualification}}
namespace dr355 { struct ::dr355_S s; }

// dr356: na

namespace dr357 { // dr357: yes
  template<typename T> struct A {
    void f() const; // expected-note {{const qualified}}
  };
  template<typename T> void A<T>::f() {} // expected-error {{does not match}}

  struct B {
    template<typename T> void f();
  };
  template<typename T> void B::f() const {} // expected-error {{does not match}}
}

namespace dr358 { // dr358: yes
  extern "C" void dr358_f();
  namespace N {
    int var;
    extern "C" void dr358_f() { var = 10; }
  }
}

namespace dr359 { // dr359: yes
  // Note, the example in the DR is wrong; it doesn't contain an anonymous
  // union.
  struct E {
    union {
      struct {
        int x;
      } s;
    } v;

    union {
      struct { // expected-error {{extension}}
        int x;
      } s;

      struct S { // expected-error {{types cannot be declared in an anonymous union}}
        int x;
      } t;

      union { // expected-error {{extension}}
        int u;
      };
    };
  };
}

namespace dr360 { // dr360: yes
struct A {
  int foo();
  int bar();

protected:
  int baz();
};

struct B : A {
private:
  using A::foo; // #dr360-foo-using-decl
protected:
  using A::bar; // #dr360-bar-using-decl
public:
  using A::baz;
};

int main() {
  int foo = B().foo(); // expected-error {{is a private member}}
  // expected-note@#dr360-foo-using-decl {{declared private here}}
  int bar = B().bar(); // expected-error {{is a protected member}}
  // expected-note@#dr360-bar-using-decl {{declared protected here}}
  int baz = B().baz();
}
} // namespace dr360

// dr362: na
// dr363: na

namespace dr364 { // dr364: yes
  struct S {
    static void f(int);
    void f(char);
  };

  void g() {
    S::f('a'); // expected-error {{call to non-static}}
    S::f(0);
  }
}

#if "foo" // expected-error {{invalid token}} dr366: yes
#endif

namespace dr367 { // dr367: yes
  // FIXME: These diagnostics are terrible. Don't diagnose an ill-formed global
  // array as being a VLA!
  int a[true ? throw 0 : 4]; // expected-error 2{{variable length array}}
  int b[true ? 4 : throw 0];
  int c[true ? *new int : 4]; // expected-error 2{{variable length array}} expected-note {{read of uninitialized}}
  int d[true ? 4 : *new int];
#if __cplusplus < 201103L
  // expected-error@-4 2{{variable length array}}
  // expected-error@-3 2{{variable length array}}
#endif
}

namespace dr368 { // dr368: yes
  template<typename T, T> struct S {}; // expected-note {{here}}
  template<typename T> int f(S<T, T()> *); // expected-error {{function type}}
  template<typename T> int g(S<T, (T())> *); // cxx98_17-note {{type 'X'}}
  // cxx20_2b-note@-1 {{candidate function [with T = dr368::X]}}
  template<typename T> int g(S<T, true ? T() : T()> *); // cxx98_17-note {{type 'X'}}
  // cxx20_2b-note@-1 {{candidate function [with T = dr368::X]}}
  struct X {};
  int n = g<X>(0); // cxx98_17-error {{no matching}}
  // cxx20_2b-error@-1 {{call to 'g' is ambiguous}}
}

// dr370: na

namespace dr372 { // dr372: no
  namespace example1 {
    template<typename T> struct X {
    protected:
      typedef T Type; // expected-note 2{{protected}}
    };
    template<typename T> struct Y {};

    // FIXME: These two are valid; deriving from T1<T> gives Z1 access to
    // the protected member T1<T>::Type.
    template<typename T,
             template<typename> class T1,
             template<typename> class T2> struct Z1 :
      T1<T>,
      T2<typename T1<T>::Type> {}; // expected-error {{protected}}

    template<typename T,
             template<typename> class T1,
             template<typename> class T2> struct Z2 :
      T2<typename T1<T>::Type>, // expected-error {{protected}}
      T1<T> {};

    Z1<int, X, Y> z1; // expected-note {{instantiation of}}
    Z2<int, X, Y> z2; // expected-note {{instantiation of}}
  }

  namespace example2 {
    struct X {
    private:
      typedef int Type; // expected-note {{private}}
    };
    template<typename T> struct A {
      typename T::Type t; // expected-error {{private}}
    };
    A<X> ax; // expected-note {{instantiation of}}
  }

  namespace example3 {
    struct A {
    protected:
      typedef int N; // expected-note 2{{protected}}
    };

    template<typename T> struct B {};
    template<typename U> struct C : U, B<typename U::N> {}; // expected-error {{protected}}
    template<typename U> struct D : B<typename U::N>, U {}; // expected-error {{protected}}

    C<A> x; // expected-note {{instantiation of}}
    D<A> y; // expected-note {{instantiation of}}
  }

  namespace example4 {
    class A {
      class B {};
      friend class X;
    };

    struct X : A::B {
      A::B mx;
      class Y {
        A::B my;
      };
    };
  }

  // FIXME: This is valid: deriving from A gives D access to A::B
  namespace std_example {
    class A {
    protected:
      struct B {}; // expected-note {{here}}
    };
    struct D : A::B, A {}; // expected-error {{protected}}
  }

  // FIXME: This is valid: deriving from A::B gives access to A::B!
  namespace badwolf {
    class A {
    protected:
      struct B; // expected-note {{here}}
    };
    struct A::B : A {};
    struct C : A::B {}; // expected-error {{protected}}
  }
}

namespace dr373 { // dr373: 5
  namespace X { int dr373; }
  struct dr373 { // expected-note {{here}}
    void f() {
      using namespace dr373::X;
      int k = dr373; // expected-error {{does not refer to a value}}

      namespace Y = dr373::X;
      k = Y::dr373;
    }
  };

  struct A { struct B {}; }; // expected-note 2{{here}}
  namespace X = A::B;   // expected-error {{expected namespace name}}
  using namespace A::B; // expected-error {{expected namespace name}}
}

namespace dr374 { // dr374: yes
  namespace N {
    template<typename T> void f();
    template<typename T> struct A { void f(); };
  }
  template<> void N::f<char>() {}
  template<> void N::A<char>::f() {}
  template<> struct N::A<int> {};
}

// dr375: dup 345
// dr376: na

namespace dr377 { // dr377: yes
  enum E { // expected-error {{enumeration values exceed range of largest integer}}
    a = -__LONG_LONG_MAX__ - 1, // expected-error 0-1{{extension}}
    b = 2 * (unsigned long long)__LONG_LONG_MAX__ // expected-error 0-2{{extension}}
  };
}

// dr378: dup 276
// dr379: na

namespace dr381 { // dr381: yes
  struct A {
    int a;
  };
  struct B : virtual A {};
  struct C : B {};
  struct D : B {};
  struct E : public C, public D {};
  struct F : public A {};
  void f() {
    E e;
    e.B::a = 0; // expected-error {{ambiguous conversion}}
    F f;
    f.A::a = 1;
  }
}

namespace dr382 { // dr382: yes c++11
  // FIXME: Should we allow this in C++98 mode?
  struct A { typedef int T; };
  typename A::T t;
  typename dr382::A a;
#if __cplusplus < 201103L
  // expected-error@-3 {{occurs outside of a template}}
  // expected-error@-3 {{occurs outside of a template}}
#endif
  typename A b; // expected-error {{expected a qualified name}}
}

namespace dr383 { // dr383: yes
  struct A { A &operator=(const A&); };
  struct B { ~B(); };
  union C { C &operator=(const C&); };
  union D { ~D(); };
  int check[(__is_pod(A) || __is_pod(B) || __is_pod(C) || __is_pod(D)) ? -1 : 1];
}

namespace dr384 { // dr384: yes
  namespace N1 {
    template<typename T> struct Base {};
    template<typename T> struct X {
      struct Y : public Base<T> {
        Y operator+(int) const;
      };
      Y f(unsigned i) { return Y() + i; }
    };
  }

  namespace N2 {
    struct Z {};
    template<typename T> int *operator+(T, unsigned);
  }

  int main() {
    N1::X<N2::Z> v;
    v.f(0);
  }
}

namespace dr385 { // dr385: yes
  struct A { protected: void f(); };
  struct B : A { using A::f; };
  struct C : A { void g(B b) { b.f(); } };
  void h(B b) { b.f(); }

  struct D { int n; }; // expected-note {{member}}
  struct E : protected D {}; // expected-note {{protected}}
  struct F : E { friend int i(E); };
  int i(E e) { return e.n; } // expected-error {{protected member}}
}

namespace dr387 { // dr387: yes
  namespace old {
    template<typename T> class number {
      number(int); // expected-note 2{{here}}
      friend number gcd(number &x, number &y) {}
    };

    void g() {
      number<double> a(3), b(4); // expected-error 2{{private}}
      a = gcd(a, b);
      b = gcd(3, 4); // expected-error {{undeclared}}
    }
  }

  namespace newer {
    template <typename T> class number {
    public:
      number(int);
      friend number gcd(number x, number y) { return 0; }
    };

    void g() {
      number<double> a(3), b(4);
      a = gcd(a, b);
      b = gcd(3, 4); // expected-error {{undeclared}}
    }
  }
}

// FIXME: dr388 needs codegen test

namespace dr389 { // dr389: no
  struct S {
    typedef struct {} A;
    typedef enum {} B;
    typedef struct {} const C; // expected-note 0-2{{here}}
    typedef enum {} const D; // expected-note 0-1{{here}}
  };
  template<typename> struct T {};

  struct WithLinkage1 {};
  enum WithLinkage2 {};
  typedef struct {} *WithLinkage3a, WithLinkage3b;
  typedef enum {} WithLinkage4a, *WithLinkage4b;
  typedef S::A WithLinkage5;
  typedef const S::B WithLinkage6;
  typedef int WithLinkage7;
  typedef void (*WithLinkage8)(WithLinkage2 WithLinkage1::*, WithLinkage5 *);
  typedef T<WithLinkage5> WithLinkage9;

  typedef struct {} *WithoutLinkage1; // expected-note 0-1{{here}}
  typedef enum {} const WithoutLinkage2; // expected-note 0-1{{here}}
  // These two types don't have linkage even though they are externally visible
  // and the ODR requires them to be merged across TUs.
  typedef S::C WithoutLinkage3;
  typedef S::D WithoutLinkage4;
  typedef void (*WithoutLinkage5)(int (WithoutLinkage3::*)(char));

#if __cplusplus >= 201103L
  // This has linkage even though its template argument does not.
  // FIXME: This is probably a defect.
  typedef T<WithoutLinkage1> WithLinkage10;
#else
  typedef int WithLinkage10; // dummy

  typedef T<WithLinkage1> GoodArg1;
  typedef T<WithLinkage2> GoodArg2;
  typedef T<WithLinkage3a> GoodArg3a;
  typedef T<WithLinkage3b> GoodArg3b;
  typedef T<WithLinkage4a> GoodArg4a;
  typedef T<WithLinkage4b> GoodArg4b;
  typedef T<WithLinkage5> GoodArg5;
  typedef T<WithLinkage6> GoodArg6;
  typedef T<WithLinkage7> GoodArg7;
  typedef T<WithLinkage8> GoodArg8;
  typedef T<WithLinkage9> GoodArg9;

  typedef T<WithoutLinkage1> BadArg1; // expected-error{{template argument uses}}
  typedef T<WithoutLinkage2> BadArg2; // expected-error{{template argument uses}}
  typedef T<WithoutLinkage3> BadArg3; // expected-error{{template argument uses}}
  typedef T<WithoutLinkage4> BadArg4; // expected-error{{template argument uses}}
  typedef T<WithoutLinkage5> BadArg5; // expected-error{{template argument uses}}
#endif

  extern WithLinkage1 withLinkage1;
  extern WithLinkage2 withLinkage2;
  extern WithLinkage3a withLinkage3a;
  extern WithLinkage3b withLinkage3b;
  extern WithLinkage4a withLinkage4a;
  extern WithLinkage4b withLinkage4b;
  extern WithLinkage5 withLinkage5;
  extern WithLinkage6 withLinkage6;
  extern WithLinkage7 withLinkage7;
  extern WithLinkage8 withLinkage8;
  extern WithLinkage9 withLinkage9;
  extern WithLinkage10 withLinkage10;

  // FIXME: These are all ill-formed.
  extern WithoutLinkage1 withoutLinkage1;
  extern WithoutLinkage2 withoutLinkage2;
  extern WithoutLinkage3 withoutLinkage3;
  extern WithoutLinkage4 withoutLinkage4;
  extern WithoutLinkage5 withoutLinkage5;

  // OK, extern "C".
  extern "C" {
    extern WithoutLinkage1 dr389_withoutLinkage1;
    extern WithoutLinkage2 dr389_withoutLinkage2;
    extern WithoutLinkage3 dr389_withoutLinkage3;
    extern WithoutLinkage4 dr389_withoutLinkage4;
    extern WithoutLinkage5 dr389_withoutLinkage5;
  }

  // OK, defined.
  WithoutLinkage1 withoutLinkageDef1;
  WithoutLinkage2 withoutLinkageDef2 = WithoutLinkage2();
  WithoutLinkage3 withoutLinkageDef3 = {};
  WithoutLinkage4 withoutLinkageDef4 = WithoutLinkage4();
  WithoutLinkage5 withoutLinkageDef5;

  void use(const void *);
  void use_all() {
    use(&withLinkage1); use(&withLinkage2); use(&withLinkage3a); use(&withLinkage3b);
    use(&withLinkage4a); use(&withLinkage4b); use(&withLinkage5); use(&withLinkage6);
    use(&withLinkage7); use(&withLinkage8); use(&withLinkage9); use(&withLinkage10);

    use(&withoutLinkage1); use(&withoutLinkage2); use(&withoutLinkage3);
    use(&withoutLinkage4); use(&withoutLinkage5);

    use(&dr389_withoutLinkage1); use(&dr389_withoutLinkage2);
    use(&dr389_withoutLinkage3); use(&dr389_withoutLinkage4);
    use(&dr389_withoutLinkage5);

    use(&withoutLinkageDef1); use(&withoutLinkageDef2); use(&withoutLinkageDef3);
    use(&withoutLinkageDef4); use(&withoutLinkageDef5);
  }

  void local() {
    // FIXME: This is ill-formed.
    extern WithoutLinkage1 withoutLinkageLocal;
  }
}

namespace dr390 { // dr390: yes
  template<typename T>
  struct A {
    A() { f(); } // expected-warning {{call to pure virt}}
    virtual void f() = 0; // expected-note {{here}}
    virtual ~A() = 0;
  };
  template<typename T> A<T>::~A() { T::error; } // expected-error {{cannot be used prior to}}
  template<typename T> void A<T>::f() { T::error; } // ok, not odr-used
  struct B : A<int> { // expected-note 2{{in instantiation of}}
    void f() {}
  } b;
}

namespace dr391 { // dr391: yes c++11
  // FIXME: Should this apply to C++98 too?
  class A { A(const A&); }; // expected-note 0-1{{here}}
  A fa();
  const A &a = fa();
#if __cplusplus < 201103L
  // expected-error@-2 {{C++98 requires an accessible copy constructor}}
#endif

  struct B { B(const B&) = delete; }; // expected-error 0-1{{extension}} expected-note 0-1{{here}}
  B fb();
  const B &b = fb();
#if __cplusplus < 201103L
  // expected-error@-2 {{deleted}}
#endif

  template<typename T>
  struct C {
    C(const C&) { T::error; }
  };
  C<int> fc();
  const C<int> &c = fc();
}

// dr392 FIXME write codegen test
// dr394: na

namespace dr395 { // dr395: yes
  struct S {
    template <typename T, int N>(&operator T())[N]; // expected-error {{cannot specify any part of a return type}}
    template <typename T, int N> operator(T (&)[N])(); // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error +{{}}
    template <typename T> operator T *() const { return 0; }
    template <typename T, typename U> operator T U::*() const { return 0; }
    template <typename T, typename U> operator T (U::*)()() const { return 0; } // expected-error +{{}}
  };

  struct null1_t {
    template <class T, class U> struct ptr_mem_fun_t {
      typedef T (U::*type)();
    };

    template <class T, class U>
    operator typename ptr_mem_fun_t<T, U>::type() const { // expected-note {{couldn't infer}}
      return 0;
    }
  } null1;
  int (S::*p)() = null1; // expected-error {{no viable conversion}}

  template <typename T> using id = T; // expected-error 0-1{{extension}}

  struct T {
    template <typename T, int N> operator id<T[N]> &();
    template <typename T, typename U> operator id<T (U::*)()>() const;
  };

  struct null2_t {
    template<class T, class U> using ptr_mem_fun_t = T (U::*)(); // expected-error 0-1{{extension}}
    template<class T, class U> operator ptr_mem_fun_t<T, U>() const { return 0; };
  } null2;
  int (S::*q)() = null2;
}

namespace dr396 { // dr396: yes
  void f() {
    auto int a(); // expected-error {{storage class on function}}
    int (i); // expected-note {{previous}}
    auto int (i); // expected-error {{redefinition}}
#if __cplusplus >= 201103L
  // expected-error@-4 {{'auto' storage class}} expected-error@-2 {{'auto' storage class}}
#endif
  }
}

// dr397: sup 1823

namespace dr398 { // dr398: yes
  namespace example1 {
    struct S {
      static int const I = 42;
    };
    template <int N> struct X {};
    template <typename T> void f(X<T::I> *) {}
    template <typename T> void f(X<T::J> *) {}
    void foo() { f<S>(0); }
  }

  namespace example2 {
    template <int I> struct X {};
    template <template <class T> class> struct Z {};
    template <class T> void f(typename T::Y *) {} // expected-note 2{{substitution failure}}
    template <class T> void g(X<T::N> *) {} // expected-note {{substitution failure}}
    template <class T> void h(Z<T::template TT> *) {} // expected-note {{substitution failure}}
    struct A {};
    struct B {
      int Y;
    };
    struct C {
      typedef int N;
    };
    struct D {
      typedef int TT;
    };

    void test() {
      f<A>(0); // expected-error {{no matching function}}
      f<B>(0); // expected-error {{no matching function}}
      g<C>(0); // expected-error {{no matching function}}
      h<D>(0); // expected-error {{no matching function}}
    }
  }
}
