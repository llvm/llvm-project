// RUN: %clang_cc1 -std=c++98 -verify=expected,cxx98-14,cxx98-17,cxx98-20,cxx98 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -verify=expected,cxx98-14,cxx98-17,cxx98-20,cxx11-14,since-cxx11 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -verify=expected,cxx98-14,cxx98-17,cxx98-20,cxx11-14,since-cxx11 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -verify=expected,cxx98-17,cxx98-20,since-cxx11,since-cxx17 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -verify=expected,cxx98-20,cxx20-23,since-cxx11,since-cxx17 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -verify=expected,cxx20-23,cxx23,since-cxx11,since-cxx17,since-cxx23 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -verify=expected,cxx20-23,cxx23,since-cxx11,since-cxx17,since-cxx23 -triple %itanium_abi_triple %s -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

#if __cplusplus == 199711L
#define __enable_constant_folding(x) (__builtin_constant_p(x) ? (x) : (x))
#else
#define __enable_constant_folding
#endif

namespace cwg300 { // cwg300: 2.7
  template<typename R, typename A> void f(R (&)(A)) {}
  int g(int);
  void h() { f(g); }
} // namespace cwg300

namespace cwg301 { // cwg301: 3.5
  // see also cwg38
  struct S;
  template<typename T> void operator+(T, T);
  void operator-(S, S);

  void f() {
    bool a = (void(*)(S, S))operator+<S> < (void(*)(S, S))operator+<S>;
    // expected-warning@-1 {{ordered comparison of function pointers ('void (*)(S, S)' and 'void (*)(S, S)')}}
    bool b = (void(*)(S, S))operator- < (void(*)(S, S))operator-;
    // cxx98-17-warning@-1 {{ordered comparison of function pointers ('void (*)(S, S)' and 'void (*)(S, S)')}}
    // cxx20-23-error@-2 {{expected '>'}}
    //   cxx20-23-note@-3 {{to match this '<'}}
    bool c = (void(*)(S, S))operator+ < (void(*)(S, S))operator-;
    // expected-error@-1 {{expected '>'}}
    //   expected-note@-2 {{to match this '<'}}
  }

  template<typename T> void f() {
    // FIXME: We are emitting a lot of bogus diagnostics here.
    typename T::template operator+<int> a;
    // expected-error@-1 {{typename specifier refers to a non-type template}}
    // expected-error@-2 {{'template' keyword not permitted here}}
    // expected-error@-3 {{a type specifier is required for all declarations}}
    // expected-error@-4 {{'operator+' cannot be the name of a variable or data member}}
    // expected-error@-5 {{expected ';' at end of declaration}}
    // FIXME: This shouldn't say (null).
    class T::template operator+<int> b;
    // expected-error@-1 {{identifier followed by '<' indicates a class template specialization but (null) refers to a function template}}
    enum T::template operator+<int> c;
    // expected-error@-1 {{expected identifier}}
    enum T::template operator+<int>::E d;
    // expected-error@-1 {{qualified name refers into a specialization of function template 'T::template operator +'}}
    // expected-error@-2 {{ISO C++ forbids forward references to 'enum' types}}
    enum T::template X<int>::E e;
    T::template operator+<int>::foobar();
    // expected-error@-1 {{qualified name refers into a specialization of function template 'T::template operator +'}}
    T::template operator+<int>(0); // ok
  }

  // FIXME: We are emitting a bunch of bogus diagnostics for the next 3 lines.
  //        All of them do a bad job at explaining that 'class' is not allowed here.
  template<typename T> class operator&<T*> {};
  // expected-error@-1 {{declaration of anonymous class must be a definition}}
  // expected-error@-2 {{declaration does not declare anything}}
  template<typename T> class T::operator& {};
  // expected-error@-1 {{expected identifier}}
  // expected-error@-2 {{declaration of anonymous class must be a definition}}
  // expected-error@-3 {{declaration does not declare anything}}
  template<typename T> class S::operator&<T*> {};
  // expected-error@-1 {{expected identifier}}
  // expected-error@-2 {{declaration of anonymous class must be a definition}}
  // expected-error@-3 {{declaration does not declare anything}}
} // namespace cwg301

namespace cwg302 { // cwg302: 3.0
  struct A { A(); ~A(); };
#if __cplusplus < 201103L
  struct B {
  // expected-error@-1 {{implicit default constructor for 'cwg302::B' must explicitly initialize the const member 'n'}}
  //   expected-note@#cwg302-b {{in implicit default constructor for 'cwg302::B' first required here}}
  //   expected-note@#cwg302-B-n {{declared here}}
    const int n; // #cwg302-B-n
    A a;
  } b = B(); // #cwg302-b
  // Trivial default constructor C::C() is not called here.
  struct C {
    const int n;
  } c = C();
#else
  struct B {
    const int n; // #cwg302-B-n
    A a;
  } b = B();
  // expected-error@-1 {{call to implicitly-deleted default constructor of 'B'}}
  //   expected-note@#cwg302-B-n {{default constructor of 'B' is implicitly deleted because field 'n' of const-qualified type 'const int' would not be initialized}}
  // C::C() is called here, because even though it's trivial, it's deleted.
  struct C {
    const int n; // #cwg302-C-n
  } c = C();
  // expected-error@-1 {{call to implicitly-deleted default constructor of 'C'}}
  //   expected-note@#cwg302-C-n {{default constructor of 'C' is implicitly deleted because field 'n' of const-qualified type 'const int' would not be initialized}}
  struct D {
    const int n = 0;
  } d = D();
#endif
} // namespace cwg302

// cwg303: na

namespace cwg304 { // cwg304: 2.9
  typedef int &a;
  int n = a();
  // expected-error@-1 {{reference to type 'int' requires an initializer}}

  struct S { int &b; }; // #cwg304-S
  // cxx98-error@-1 {{reference to type 'int' requires an initializer}}
  //   cxx98-note@#cwg304-m {{in value-initialization of type 'S' here}}
  int m = S().b; // #cwg304-m
  // since-cxx11-error@-1 {{call to implicitly-deleted default constructor of 'S'}}
  //   since-cxx11-note@#cwg304-S {{default constructor of 'S' is implicitly deleted because field 'b' of reference type 'int &' would not be initialized}}
} // namespace cwg304

namespace cwg305 { // cwg305: no
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
    struct B {}; // #cwg305-h-B
    b->~B();
    // expected-error@-1 {{destructor type 'B' in object destruction expression does not match the type 'B' (aka 'cwg305::A') of the object being destroyed}}
    //   expected-note@#cwg305-h-B {{type 'B' found by destructor name lookup}}
  }

  template<typename T> struct X {};
  void i(X<int>* x) {
    struct X {};
    x->~X<int>();
    x->~X();
    x->~X<char>();
    // expected-error@-1 {{no member named '~X' in 'cwg305::X<int>'}}
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
    z->~T1<int>();
    // since-cxx11-error@-1 {{no member named 'T1' in 'cwg305::Z'}}
    z->~T2<int>();
    // since-cxx11-error@-1 {{no member named '~int' in 'cwg305::Z'}}
    z->~T2<Z>();
  }

  // FIXME: This is valid.
  namespace Q {
    template<typename A> struct R {};
  }
  template<typename A> using R = Q::R<int>;
  void qr(Q::R<int> x) { x.~R<char>(); }
  // since-cxx11-error@-1 {{no member named '~R' in 'cwg305::Q::R<int>'}}
#endif
} // namespace cwg305

namespace cwg306 { // cwg306: dup 39
  struct A { struct B {}; };
  struct C { typedef A::B B; };
  struct D : A, A::B, C {};
  D::B b;

  struct X {}; // #cwg306-X
  template<typename T> struct Y { typedef T X; }; // #cwg306-typedef-X
  template<typename T> struct Z : X, Y<T> {};
  Z<X>::X zx;
  Z<const X>::X zcx;
  // expected-error@-1 {{member 'X' found in multiple base classes of different types}}
  //   expected-note@#cwg306-X {{member type 'cwg306::X' found}}
  //   expected-note@#cwg306-typedef-X {{member type 'const cwg306::X' found}}
} // namespace cwg306

// cwg307: na

namespace cwg308 { // cwg308: 3.7
  // This is mostly an ABI library issue.
  struct A {};
  struct B : A {};
  struct C : A {};
  struct D : B, C {};
  void f() {
    // NB: the warning here is correct despite being the opposite of the
    // comments in the catch handlers. The "unreachable" comment is correct
    // because there is an ambiguous base path to A from the D that is thrown.
    // The warnings generated are also correct because the handlers handle
    // const B& and const A& and we don't check to see if other derived classes
    // exist that would cause an ambiguous base path. We issue the diagnostic
    // despite the potential for a false positive because users are not
    // expected to have ambiguous base paths all that often, so the false
    // positive rate should be acceptably low.
    try {
      throw D();
    } catch (const A&) { // #cwg308-catch-A
      // unreachable
    } catch (const B&) {
      // expected-warning@-1 {{exception of type 'const B &' will be caught by earlier handler}}
      //   expected-note@#cwg308-catch-A {{for type 'const A &'}}
      // get here instead
    }
  }
} // namespace cwg308

// cwg309: dup 485

namespace cwg311 { // cwg311: 3.0
  namespace X { namespace Y {} }
  namespace X::Y {}
  // cxx98-14-error@-1 {{nested namespace definition is a C++17 extension; define each namespace separately}}
  namespace X {
    namespace X::Y {}
    // cxx98-14-error@-1 {{nested namespace definition is a C++17 extension; define each namespace separately}}
  }
  // FIXME: The diagnostics here are not very good.
  namespace ::cwg311::X {}
  // expected-error@-1 {{expected identifier or '{'}}
  // expected-warning@-2 {{extra qualification on member 'X'}}
  // expected-error@-3 {{a type specifier is required for all declarations}}
  // expected-error@-4 {{expected ';' after top level declarator}}
} // namespace cwg311

// cwg312: dup 616

namespace cwg313 { // cwg313: dup 299 c++11
  struct A { operator int() const; };
  // FIXME: should this be available in c++98 mode?
  int *p = new int[A()];
  // cxx98-error@-1 {{implicit conversion from array size expression of type 'A' to integral type 'int' is a C++11 extension}}
} // namespace cwg313

namespace cwg314 { // cwg314: no
                  // NB: dup 1710
template <typename T> struct A {
  template <typename U> struct B {};
};
template <typename T> struct C : public A<T>::template B<T> {
  C() : A<T>::template B<T>() {}
};
template <typename T> struct C2 : public A<T>::B<T> {
  // expected-error@-1 {{use 'template' keyword to treat 'B' as a dependent template name}}
  C2() : A<T>::B<T>() {}
  // expected-error@-1 {{use 'template' keyword to treat 'B' as a dependent template name}}
};
} // namespace cwg314

// cwg315: na
// cwg316: sup 1004

namespace cwg317 { // cwg317: 3.5
  void f() {} // #cwg317-f
  inline void f();
  // expected-error@-1 {{inline declaration of 'f' follows non-inline definition}}
  //   expected-note@#cwg317-f {{previous definition is here}}

  int g();
  int n = g();
  inline int g() { return 0; }

  int h();
  int m = h();
  int h() { return 0; } // #cwg317-h
  inline int h();
  // expected-error@-1 {{inline declaration of 'h' follows non-inline definition}}
  //   expected-note@#cwg317-h {{previous definition is here}}
} // namespace cwg317

namespace cwg318 { // cwg318: sup 1310
  struct A {};
  struct A::A a;
} // namespace cwg318

namespace cwg319 { // cwg319: no
  // FIXME: dup cwg389
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
    // cxx98-error@-1 {{template argument uses local type 'A'}}

    typedef A B;
    extern B b; // FIXME: ill-formed
    X<B> xb;
    // cxx98-error@-1 {{template argument uses local type 'A'}}

    const int n = 1;
    typedef int (*C)[n];
    extern C c; // ok
    X<C> xc;
  }
} // namespace cwg319

namespace cwg320 { // cwg320: 3.1
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
} // namespace cwg320

namespace cwg321 { // cwg321: dup 557
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
} // namespace cwg321

namespace cwg322 { // cwg322: 2.8
  struct A {
    template<typename T> operator T&();
  } a;
  int &r = static_cast<int&>(a);
  int &s = a;
} // namespace cwg322

// cwg323: sup 820

namespace cwg324 { // cwg324: 3.6
  struct S { int n : 1; } s; // #cwg324-n
  int &a = s.n;
  // expected-error@-1 {{non-const reference cannot bind to bit-field 'n'}}
  //   expected-note@#cwg324-n {{bit-field is declared here}}
  int *b = &s.n;
  // expected-error@-1 {{address of bit-field requested}}
  int &c = (s.n = 0);
  // expected-error@-1 {{non-const reference cannot bind to bit-field 'n'}}
  //   expected-note@#cwg324-n {{bit-field is declared here}}
  int *d = &(s.n = 0);
  // expected-error@-1 {{address of bit-field requested}}
  // FIXME: why don't we emit a note here, as for the rest of this type of diagnostic in this test?
  int &e = true ? s.n : s.n;
  // expected-error@-1 {{non-const reference cannot bind to bit-field}}
  int *f = &(true ? s.n : s.n);
  // expected-error@-1 {{address of bit-field requested}}
  int &g = (void(), s.n);
  // expected-error@-1 {{non-const reference cannot bind to bit-field 'n'}}
  //   expected-note@#cwg324-n {{bit-field is declared here}}
  int *h = &(void(), s.n);
  // expected-error@-1 {{address of bit-field requested}}
  int *i = &++s.n;
  // expected-error@-1 {{address of bit-field requested}}
} // namespace cwg324

namespace cwg326 { // cwg326: 3.1
  struct S {};
  static_assert(__is_trivially_constructible(S, const S&), "");
} // namespace cwg326

namespace cwg327 { // cwg327: dup 538
  struct A;
  class A {};

  class B;
  struct B {};
} // namespace cwg327

namespace cwg328 { // cwg328: 2.7
  struct A; // #cwg328-A
  struct B { A a; };
  // expected-error@-1 {{field has incomplete type 'A'}}
  //   expected-note@#cwg328-A {{forward declaration of 'cwg328::A'}}
  template<typename> struct C { A a; };
  // expected-error@-1 {{field has incomplete type 'A'}}
  //   expected-note@#cwg328-A {{forward declaration of 'cwg328::A'}}
  A *p = new A[0];
  // expected-error@-1 {{allocation of incomplete type 'A'}}
  //   expected-note@#cwg328-A {{forward declaration of 'cwg328::A'}}
} // namespace cwg328

namespace cwg329 { // cwg329: 3.5
  struct B {};
  template<typename T> struct A : B {
    friend void f(A a) { g(a); }
    friend void h(A a) { g(a); }
    // expected-error@-1 {{use of undeclared identifier 'g'}}
    //   expected-note@#cwg329-h-call {{in instantiation of member function 'cwg329::h' requested here}}
    friend void i(B b) {} // #cwg329-i
    // expected-error@-1 {{redefinition of 'i'}}
    //   expected-note@#cwg329-b {{in instantiation of template class 'cwg329::A<char>' requested here}}
    //   expected-note@#cwg329-i {{previous definition is here}}
  };
  A<int> a;
  A<char> b; // #cwg329-b

  void test() {
    h(a); // #cwg329-h-call
  }
} // namespace cwg329

namespace cwg330 { // cwg330: 7
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
    r = p;
    // expected-error@-1 {{incompatible pointer types assigning to 'R' (aka 'const int *const (*)[4]') from 'P' (aka 'int *(*)[3]')}}
    s = p;
    // cxx98-17-error@-1 {{incompatible pointer types assigning to 'S' (aka 'const int *const (*)[]') from 'P' (aka 'int *(*)[3]')}} (fixed by p0388)
    t = p;
    // expected-error@-1 {{incompatible pointer types assigning to 'T' (aka 'const int *(*)[]') from 'P' (aka 'int *(*)[3]')}}
    s = q;
    // cxx98-17-error@-1 {{incompatible pointer types assigning to 'S' (aka 'const int *const (*)[]') from 'Q' (aka 'const int *const (*)[3]')}} (fixed by p0388)
    s = q2;
    // cxx98-17-error@-1 {{incompatible pointer types assigning to 'S' (aka 'const int *const (*)[]') from 'Q2' (aka 'const int *const (*)[3]')}} (fixed by p0388)
    s = t; // ok, adding const
    t = s;
    // expected-error@-1 {{assigning to 'T' (aka 'const int *(*)[]') from 'S' (aka 'const int *const (*)[]') discards qualifiers}}
    (void) const_cast<P>(q);
    (void) const_cast<P>(q2);
    (void) const_cast<Q>(p);
    (void) const_cast<Q2>(p);
    (void) const_cast<S>(p);
    // expected-error@-1 {{const_cast from 'P' (aka 'int *(*)[3]') to 'S' (aka 'const int *const (*)[]') is not allowed}} (for now)
    (void) const_cast<P>(s);
    // expected-error@-1 {{const_cast from 'S' (aka 'const int *const (*)[]') to 'P' (aka 'int *(*)[3]') is not allowed}} (for now)
    (void) const_cast<S>(q);
    // expected-error@-1 {{const_cast from 'Q' (aka 'const int *const (*)[3]') to 'S' (aka 'const int *const (*)[]') is not allowed}}
    (void) const_cast<S>(q2);
    // expected-error@-1 {{const_cast from 'Q2' (aka 'const int *const (*)[3]') to 'S' (aka 'const int *const (*)[]') is not allowed}}
    (void) const_cast<Q>(s);
    // expected-error@-1 {{const_cast from 'S' (aka 'const int *const (*)[]') to 'Q' (aka 'const int *const (*)[3]') is not allowed}}
    (void) const_cast<Q2>(s);
    // expected-error@-1 {{const_cast from 'S' (aka 'const int *const (*)[]') to 'Q2' (aka 'const int *const (*)[3]') is not allowed}}
    (void) const_cast<T>(s);
    (void) const_cast<S>(t);
    (void) const_cast<T>(q);
    // expected-error@-1 {{const_cast from 'Q' (aka 'const int *const (*)[3]') to 'T' (aka 'const int *(*)[]') is not allowed}}
    (void) const_cast<Q>(t);
    // expected-error@-1 {{const_cast from 'T' (aka 'const int *(*)[]') to 'Q' (aka 'const int *const (*)[3]') is not allowed}}

    (void) reinterpret_cast<P>(q);
    // expected-error@-1 {{reinterpret_cast from 'Q' (aka 'const int *const (*)[3]') to 'P' (aka 'int *(*)[3]') casts away qualifiers}}
    (void) reinterpret_cast<P>(q2);
    // expected-error@-1 {{reinterpret_cast from 'Q2' (aka 'const int *const (*)[3]') to 'P' (aka 'int *(*)[3]') casts away qualifiers}}
    (void) reinterpret_cast<Q>(p);
    (void) reinterpret_cast<Q2>(p);
    (void) reinterpret_cast<S>(p);
    (void) reinterpret_cast<P>(s);
    // expected-error@-1 {{reinterpret_cast from 'S' (aka 'const int *const (*)[]') to 'P' (aka 'int *(*)[3]') casts away qualifiers}}
    (void) reinterpret_cast<S>(q);
    (void) reinterpret_cast<S>(q2);
    (void) reinterpret_cast<Q>(s);
    (void) reinterpret_cast<Q2>(s);
    (void) reinterpret_cast<T>(s);
    // expected-error@-1 {{reinterpret_cast from 'S' (aka 'const int *const (*)[]') to 'T' (aka 'const int *(*)[]') casts away qualifiers}}
    (void) reinterpret_cast<S>(t);
    (void) reinterpret_cast<T>(q);
    // expected-error@-1 {{reinterpret_cast from 'Q' (aka 'const int *const (*)[3]') to 'T' (aka 'const int *(*)[]') casts away qualifiers}}
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
      (void) reinterpret_cast<B2*>(a);
      // expected-error@-1 {{ISO C++ does not allow reinterpret_cast from 'A *' (aka 'const volatile int (*)[1][2][3]') to 'B2 *' (aka 'int *const X::***') because it casts away qualifiers, even though the source and destination types are unrelated}}
      (void) reinterpret_cast<B3*>(a);
      // expected-error@-1 {{ISO C++ does not allow reinterpret_cast from 'A *' (aka 'const volatile int (*)[1][2][3]') to 'B3 *' (aka 'int *X::*volatile **') because it casts away qualifiers, even though the source and destination types are unrelated}}
      (void) reinterpret_cast<B4*>(a);
    }
  }
} // namespace cwg330

namespace cwg331 { // cwg331: 11
  struct A {
    A(volatile A&); // #cwg331-A-ctor
  };
  const A a;
  // expected-error@-1 {{no matching constructor for initialization of 'const A'}}
  //   expected-note@#cwg331-A-ctor {{candidate constructor not viable: requires 1 argument, but 0 were provided}}
  const A b(a);
  // expected-error@-1 {{no matching constructor for initialization of 'const A'}}
  //   expected-note@#cwg331-A-ctor {{candidate constructor not viable: 1st argument ('const A') would lose const qualifier}}
} // namespace cwg331

namespace cwg332 { // cwg332: dup 577
  void f(volatile void);
  // expected-error@-1 {{'void' as parameter must not have type qualifiers}}
  // cxx20-23-warning@-2 {{volatile-qualified parameter type 'volatile void' is deprecated}}
  void g(const void);
  // expected-error@-1 {{'void' as parameter must not have type qualifiers}}
  void h(int n, volatile void);
  // expected-error@-1 {{'void' must be the first and only parameter if specified}}
  // cxx20-23-warning@-2 {{volatile-qualified parameter type 'volatile void' is deprecated}}
} // namespace cwg332

namespace cwg333 { // cwg333: 2.7
  int n = 0;
  int f(int(n));
  int g((int(n)));
  int h = f(g);
} // namespace cwg333

namespace cwg334 { // cwg334: 2.7
  template<typename T> void f() {
    T x;
    f((x, 123));
  }
  struct S {
    friend S operator,(S, int);
    friend void f(S);
  };
  template void f<S>();
} // namespace cwg334

// cwg335: sup 820

namespace cwg336 { // cwg336: 2.7
  namespace Pre {
    template<class T1> class A {
      template<class T2> class B {
        template<class T3> void mf1(T3);
        void mf2();
      };
    };
    template<> template<class X> class A<int>::B {}; // #cwg336-B
    template<> template<> template<class T> void A<int>::B<double>::mf1(T t) {}
    // expected-error@-1 {{out-of-line definition of 'mf1' does not match any declaration in 'cwg336::Pre::A<int>::B<double>'}}
    //   expected-note@#cwg336-B {{defined here}}
    template<class Y> template<> void A<Y>::B<double>::mf2() {}
    // expected-error@-1 {{nested name specifier 'A<Y>::B<double>::' for declaration does not refer into a class, class template or class template partial specialization}}
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
    template<class Y> template<> void A<Y>::B<double>::mf2() {}
    // expected-error@-1 {{nested name specifier 'A<Y>::B<double>::' for declaration does not refer into a class, class template or class template partial specialization}}
  }
} // namespace cwg336

namespace cwg337 { // cwg337: 2.7
  template<typename T> void f(T (*)[1]);
  template<typename T> int &f(...);

  struct A { virtual ~A() = 0; };
  int &r = f<A>(0);

  // FIXME: The language rules here are completely broken. We cannot determine
  // whether an incomplete type is abstract. See CWG1640, which will probably
  // supersede this one and remove this rule.
  struct B;
  int &s = f<B>(0);
  // expected-error@-1 {{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'void'}}
  struct B { virtual ~B() = 0; };
} // namespace cwg337

// cwg338: dup 1884

namespace cwg339 { // cwg339: 2.8
  template <int I> struct A { static const int value = I; };

  char xxx(int);
  char (&xxx(float))[2];

  template<class T> A<sizeof(xxx((T)0))> f(T) {} // #cwg339-f

  void test() {
    A<1> a = f(0);
    A<2> b = f(0.0f);
    A<3> c = f("foo");
    // expected-error@-1 {{no matching function}}
    //   expected-note@#cwg339-f {{candidate}}
  }


  char f(int);
  int f(...);

  template <class T> struct conv_int {
    static const bool value = sizeof(f(T())) == 1;
  };

  template <class T> bool conv_int2(A<sizeof(f(T()))> p);

  template<typename T> A<sizeof(f(T()))> make_A();

  static_assert(conv_int<char>::value, "");
  bool b = conv_int2<char>(A<1>());
  A<1> c = make_A<char>();
} // namespace cwg339

namespace cwg340 { // cwg340: 2.7
  struct A { A(int); };
  struct B { B(A, A, int); };
  int x, y;
  B b(A(x), A(y), 3);
} // namespace cwg340

namespace cwg341 { // cwg341: sup 1708
  namespace A {
    int n;
    extern "C" int &cwg341_a = n; // #cwg341_a
  }
  namespace B {
    extern "C" int &cwg341_a = cwg341_a;
    // expected-error@-1 {{redefinition of 'cwg341_a'}}
    //   expected-note@#cwg341_a {{previous definition is here}}
  }
  extern "C" void cwg341_b(); // #cwg341_b
}
int cwg341_a;
// expected-error@-1 {{declaration of 'cwg341_a' in global scope conflicts with declaration with C language linkage}}
//   expected-note@#cwg341_a {{declared with C language linkage here}}
int cwg341_b;
// expected-error@-1 {{declaration of 'cwg341_b' in global scope conflicts with declaration with C language linkage}}
//   expected-note@#cwg341_b {{declared with C language linkage here}}
int cwg341_c; // #cwg341_c
int cwg341_d; // #cwg341_d
namespace cwg341 {
  extern "C" int cwg341_c;
  // expected-error@-1 {{declaration of 'cwg341_c' with C language linkage conflicts with declaration in global scope}}
  //   expected-note@#cwg341_c {{declared in global scope here}}
  extern "C" void cwg341_d();
  // expected-error@-1 {{declaration of 'cwg341_d' with C language linkage conflicts with declaration in global scope}}
  //   expected-note@#cwg341_d {{declared in global scope here}}

  namespace A { extern "C" int cwg341_e; } // #cwg341_e
  namespace B { extern "C" void cwg341_e(); }
  // expected-error@-1 {{redefinition of 'cwg341_e' as different kind of symbol}}
  //   expected-note@#cwg341_e {{previous definition is here}}
} // namespace cwg341

// cwg342: na

namespace cwg343 { // cwg343: no
  // FIXME: dup 1710
  template<typename T> struct A {
    template<typename U> struct B {};
  };
  // FIXME: In these contexts, the 'template' keyword is optional.
  template<typename T> struct C : public A<T>::B<T> {
  // expected-error@-1 {{use 'template' keyword to treat 'B' as a dependent template name}}
    C() : A<T>::B<T>() {}
    // expected-error@-1 {{use 'template' keyword to treat 'B' as a dependent template name}}
  };
} // namespace cwg343

namespace cwg344 { // cwg344: dup 1435
  struct A { inline virtual ~A(); };
  struct B { friend A::~A(); };
} // namespace cwg344

namespace cwg345 { // cwg345: 2.7
  struct A {
    struct X {};
    int X; // #cwg345-int-X
  };
  struct B {
    struct X {};
  };
  template <class T> void f(T t) { typename T::X x; }
  // expected-error@-1 {{typename specifier refers to non-type member 'X' in 'cwg345::A'}}
  //   expected-note@#cwg345-f-a {{in instantiation of function template specialization 'cwg345::f<cwg345::A>' requested here}}
  //   expected-note@#cwg345-int-X {{referenced member 'X' is declared here}}
  void f(A a, B b) {
    f(b);
    f(a); // #cwg345-f-a
  }
} // namespace cwg345

// cwg346: na

namespace cwg347 { // cwg347: 2.7
  struct base {
    struct nested;
    static int n;
    static void f();
    void g();
  };

  struct derived : base {}; // #cwg347-derived

  struct derived::nested {};
  // expected-error@-1 {{no struct named 'nested' in 'cwg347::derived'}}
  int derived::n;
  // expected-error@-1 {{no member named 'n' in 'cwg347::derived'}}
  void derived::f() {}
  // expected-error@-1 {{out-of-line definition of 'f' does not match any declaration in 'cwg347::derived'}}
  //   expected-note@#cwg347-derived {{defined here}}
  void derived::g() {}
  // expected-error@-1 {{out-of-line definition of 'g' does not match any declaration in 'cwg347::derived'}}
  //   expected-note@#cwg347-derived {{defined here}}
} // namespace cwg347

// cwg348: na

namespace cwg349 { // cwg349: no
  struct A {
    template <class T> operator T ***() {
      int ***p = 0;
      return p;
      // cxx98-20-error@-1 {{cannot initialize return object of type 'const int ***' with an lvalue of type 'int ***'}}
      // since-cxx23-error@-2 {{cannot initialize return object of type 'const int ***' with an rvalue of type 'int ***'}}
      //   expected-note@#cwg349-p1 {{in instantiation of function template specialization 'cwg349::A::operator const int ***<const int>' requested here}}
    }
  };

  // FIXME: This is valid.
  A a;
  const int *const *const *p1 = a; // #cwg349-p1

  struct B {
    template <class T> operator T ***() {
      const int ***p = 0;
      return p;
    }
  };

  // FIXME: This is invalid.
  B b;
  const int *const *const *p2 = b;
} // namespace cwg349

// cwg351: na

namespace cwg352 { // cwg352: 2.8
  namespace example1 {
    namespace A {
      enum E {};
      template<typename R, typename A> void foo(E, R (*)(A)); // #cwg352-foo
    }

    template<typename T> void arg(T);
    template<typename T> int arg(T) = delete; // #cwg352-deleted
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}

    void f(A::E e) {
      foo(e, &arg);
      // expected-error@-1 {{no matching function for call to 'foo'}}
      //   expected-note@#cwg352-foo {{candidate template ignored: couldn't infer template argument 'R'}}

      using A::foo;
      foo<int, int>(e, &arg);
      // expected-error@-1 {{attempt to use a deleted function}}
      //   expected-note@#cwg352-deleted {{'arg<int>' has been explicitly marked deleted here}}
    }

    int arg(int);

    void g(A::E e) {
      foo(e, &arg);
      // expected-error@-1 {{no matching function for call to 'foo'}}
      //   expected-note@#cwg352-foo {{candidate template ignored: couldn't infer template argument 'R'}}

      using A::foo;
      foo<int, int>(e, &arg); // ok, uses non-template
    }
  }

  namespace contexts {
    template<int I> void f1(int (&)[I]);
    template<int I> void f2(int (&)[I+1]); // #cwg352-f2
    template<int I> void f3(int (&)[I+1], int (&)[I]);
    void f() {
      int a[4];
      int b[3];
      f1(a);
      f2(a);
      // expected-error@-1 {{no matching function for call to 'f2'}}
      //   expected-note@#cwg352-f2 {{candidate template ignored: couldn't infer template argument 'I'}}
      f3(a, b);
    }

    template<int I> struct S {};
    template<int I> void g1(S<I>);
    template<int I> void g2(S<I+1>); // #cwg352-g2
    template<int I> void g3(S<I+1>, S<I>);
    void g() {
      S<4> a;
      S<3> b;
      g1(a);
      g2(a);
      // expected-error@-1 {{no matching function for call to 'g2'}}
      //   expected-note@#cwg352-g2 {{candidate template ignored: couldn't infer template argument 'I'}}
      g3(a, b);
    }

    template<typename T> void h1(T = 0); // #cwg352-h1
    template<typename T> void h2(T, T = 0);
    void h() {
      h1();
      // expected-error@-1 {{no matching function for call to 'h1'}}
      //   expected-note@#cwg352-h1 {{candidate template ignored: couldn't infer template argument 'T'}}
      h1(0);
      h1<int>();
      h2(0);
    }

    template<typename T> int tmpl(T);
    template<typename R, typename A> void i1(R (*)(A)); // #cwg352-i1
    template<typename R, typename A> void i2(R, A, R (*)(A)); // #cwg352-i2
    void i() {
      extern int single(int);
      i1(single);
      i2(0, 0, single);

      extern int ambig(float), ambig(int);
      i1(ambig);
      // expected-error@-1 {{no matching function for call to 'i1'}}
      //   expected-note@#cwg352-i1 {{candidate template ignored: couldn't infer template argument 'R'}}
      i2(0, 0, ambig);

      extern void no_match(float), no_match(int);
      i1(no_match);
      // expected-error@-1 {{no matching function for call to 'i1'}}
      //   expected-note@#cwg352-i1 {{candidate template ignored: couldn't infer template argument 'R'}}
      i2(0, 0, no_match);
      // expected-error@-1 {{no matching function for call to 'i2'}}
      //   expected-note@#cwg352-i2 {{candidate function [with R = int, A = int] not viable: no overload of 'no_match' matching 'int (*)(int)' for 3rd argument}}

      i1(tmpl);
      // expected-error@-1 {{no matching function for call to 'i1'}}
      //   expected-note@#cwg352-i1 {{candidate template ignored: couldn't infer template argument 'R'}}
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
    template<int I> void g(A<I+1>); // #cwg352-g
    template<int I> void f(A<I>, A<I+1>);
    void h(A<1> a1, A<2> a2) {
      g(a1);
      // expected-error@-1 {{no matching function for call to 'g'}}
      //   expected-note@#cwg352-g {{candidate template ignored: couldn't infer template argument 'I'}}
      g<0>(a1);
      f(a1, a2);
    }
  }
} // namespace cwg352

// cwg353 needs an IRGen test.

namespace cwg354 { // cwg354: 3.1 c++11
  // FIXME: Should we allow this in C++98 too?
  struct S {};

  template<int*> struct ptr {}; // #cwg354-ptr
  ptr<0> p0; // #cwg354-p0
  // cxx98-error@#cwg354-p0 {{non-type template argument does not refer to any declaration}}
  //   cxx98-note@#cwg354-ptr {{template parameter is declared here}}
  // cxx11-14-error@#cwg354-p0 {{null non-type template argument must be cast to template parameter type 'int *'}}
  //   cxx11-14-note@#cwg354-ptr {{template parameter is declared here}}
  // since-cxx17-error@#cwg354-p0 {{conversion from 'int' to 'int *' is not allowed in a converted constant expression}}
  //   since-cxx17-note@#cwg354-ptr {{template parameter is declared here}}
  ptr<(int*)0> p1;
  // cxx98-error@-1 {{non-type template argument does not refer to any declaration}}
  //   cxx98-note@#cwg354-ptr {{template parameter is declared here}}
  ptr<(float*)0> p2; // #cwg354-p2
  // cxx98-error@#cwg354-p2 {{non-type template argument does not refer to any declaration}}
  //   cxx98-note@#cwg354-ptr {{template parameter is declared here}}
  // cxx11-14-error@#cwg354-p2 {{null non-type template argument of type 'float *' does not match template parameter of type 'int *'}}
  //   cxx11-14-note@#cwg354-ptr {{template parameter is declared here}}
  // since-cxx17-error@#cwg354-p2 {{value of type 'float *' is not implicitly convertible to 'int *'}}
  //   since-cxx17-note@#cwg354-ptr {{template parameter is declared here}}
  ptr<(int S::*)0> p3; // #cwg354-p3
  // cxx98-error@#cwg354-p3 {{non-type template argument does not refer to any declaration}}
  //   cxx98-note@#cwg354-ptr {{template parameter is declared here}}
  // cxx11-14-error@#cwg354-p3 {{null non-type template argument of type 'int S::*' does not match template parameter of type 'int *'}}
  //   cxx11-14-note@#cwg354-ptr {{template parameter is declared here}}
  // since-cxx17-error@#cwg354-p3 {{value of type 'int S::*' is not implicitly convertible to 'int *'}}
  //   since-cxx17-note@#cwg354-ptr {{template parameter is declared here}}

  template<int*> int both(); // #cwg354-both-int-ptr
  template<int> int both(); // #cwg354-both-int
  int b0 = both<0>();
  int b1 = both<(int*)0>();
  // cxx98-error@-1 {{no matching function for call to 'both'}}
  //   cxx98-note@#cwg354-both-int-ptr {{candidate template ignored: invalid explicitly-specified argument for 1st template parameter}}
  //   cxx98-note@#cwg354-both-int {{candidate template ignored: invalid explicitly-specified argument for 1st template parameter}}

  template<int S::*> struct ptr_mem {}; // #cwg354-ptr_mem
  ptr_mem<0> m0; // #cwg354-m0
  // cxx98-error@#cwg354-m0 {{non-type template argument of type 'int' cannot be converted to a value of type 'int S::*'}}
  //   cxx98-note@#cwg354-ptr_mem {{template parameter is declared here}}
  // cxx11-14-error@#cwg354-m0 {{null non-type template argument must be cast to template parameter type 'int S::*'}}
  //   cxx11-14-note@#cwg354-ptr_mem {{template parameter is declared here}}
  // since-cxx17-error@#cwg354-m0 {{conversion from 'int' to 'int S::*' is not allowed in a converted constant expression}}
  //   since-cxx17-note@#cwg354-ptr_mem {{template parameter is declared here}}
  ptr_mem<(int S::*)0> m1;
  // cxx98-error@-1 {{non-type template argument is not a pointer to member constant}}
  ptr_mem<(float S::*)0> m2; // #cwg354-m2
  // cxx98-error@#cwg354-m2 {{non-type template argument of type 'float S::*' cannot be converted to a value of type 'int S::*'}}
  //   cxx98-note@#cwg354-ptr_mem {{template parameter is declared here}}
  // cxx11-14-error@#cwg354-m2 {{null non-type template argument of type 'float S::*' does not match template parameter of type 'int S::*'}}
  //   cxx11-14-note@#cwg354-ptr_mem {{template parameter is declared here}}
  // since-cxx17-error@#cwg354-m2 {{value of type 'float S::*' is not implicitly convertible to 'int S::*'}}
  //   since-cxx17-note@#cwg354-ptr_mem {{template parameter is declared here}}
  ptr_mem<(int *)0> m3; // #cwg354-m3
  // cxx98-error@#cwg354-m3 {{non-type template argument of type 'int *' cannot be converted to a value of type 'int S::*'}}
  //   cxx98-note@#cwg354-ptr_mem {{template parameter is declared here}}
  // cxx11-14-error@#cwg354-m3 {{null non-type template argument of type 'int *' does not match template parameter of type 'int S::*'}}
  //   cxx11-14-note@#cwg354-ptr_mem {{template parameter is declared here}}
  // since-cxx17-error@#cwg354-m3 {{value of type 'int *' is not implicitly convertible to 'int S::*'}}
  //   since-cxx17-note@#cwg354-ptr_mem {{template parameter is declared here}}
} // namespace cwg354

struct cwg355_S; // cwg355: 2.7
struct ::cwg355_S {};
// expected-warning@-1 {{extra qualification on member 'cwg355_S'}}
namespace cwg355 {
struct ::cwg355_S s;
} // namespace cwg355

// cwg356: na

namespace cwg357 { // cwg357: 2.7
  template<typename T> struct A { // #cwg357-A
    void f() const; // #cwg357-f
  };
  template<typename T> void A<T>::f() {}
  // expected-error@-1 {{out-of-line definition of 'f' does not match any declaration in 'A<T>'}}
  //   expected-note@#cwg357-A {{defined here}}
  //   expected-note@#cwg357-f {{member declaration does not match because it is const qualified}}

  struct B { // #cwg357-B
    template<typename T> void f();
  };
  template<typename T> void B::f() const {}
  // expected-error@-1 {{out-of-line definition of 'f' does not match any declaration in 'cwg357::B'}}
  //   expected-note@#cwg357-B {{defined here}}
} // namespace cwg357

namespace cwg358 { // cwg358: 2.7
  extern "C" void cwg358_f();
  namespace N {
    int var;
    extern "C" void cwg358_f() { var = 10; }
  }
} // namespace cwg358

namespace cwg359 { // cwg359: 3.3
  // Note, the example in the DR is wrong; it doesn't contain an anonymous
  // union.
  struct E {
    union {
      struct {
        int x;
      } s;
    } v;

    union {
      struct {
        // expected-error@-1 {{anonymous types declared in an anonymous union are an extension}}
        int x;
      } s;

      struct S {
        // expected-error@-1 {{types cannot be declared in an anonymous union}}
        int x;
      } t;

      union {
        // expected-error@-1 {{anonymous types declared in an anonymous union are an extension}}
        int u;
      };
    };
  };
} // namespace cwg359

namespace cwg360 { // cwg360: 2.8
struct A {
  int foo();
  int bar();

protected:
  int baz();
};

struct B : A {
private:
  using A::foo; // #cwg360-using-foo
protected:
  using A::bar; // #cwg360-using-bar
public:
  using A::baz;
};

int main() {
  int foo = B().foo();
  // expected-error@-1 {{'foo' is a private member of 'cwg360::B'}}
  //   expected-note@#cwg360-using-foo {{declared private here}}
  int bar = B().bar();
  // expected-error@-1 {{'bar' is a protected member of 'cwg360::B'}}
  //   expected-note@#cwg360-using-bar {{declared protected here}}
  int baz = B().baz();
}
} // namespace cwg360

// cwg362: na
// cwg363: na

namespace cwg364 { // cwg364: 2.7
  struct S {
    static void f(int);
    void f(char);
  };

  void g() {
    S::f('a');
    // expected-error@-1 {{call to non-static member function without an object argument}}
    S::f(0);
  }
} // namespace cwg364

namespace cwg366 { // cwg366: 2.7
#if "foo" // expected-error {{invalid token at start of a preprocessor expression}}
#endif
} // namespace cwg366

namespace cwg367 { // cwg367: 2.7
  static_assert(__enable_constant_folding(true ? throw 0 : 4), "");
  // expected-error@-1 {{expression is not an integral constant expression}}
  static_assert(__enable_constant_folding(true ? 4 : throw 0), "");
  static_assert(__enable_constant_folding(true ? *new int : 4), "");
  // expected-error@-1 {{expression is not an integral constant expression}}
  //   expected-note@-2 {{read of uninitialized object is not allowed in a constant expression}}
  static_assert(__enable_constant_folding(true ? 4 : *new int), "");
} // namespace cwg367

namespace cwg368 { // cwg368: 3.6
  template<typename T, T> struct S {}; // #cwg368-S
  template<typename T> int f(S<T, T()> *);
  // expected-error@-1 {{template argument for non-type template parameter is treated as function type 'T ()'}}
  //   expected-note@#cwg368-S {{template parameter is declared here}}
  template<typename T> int g(S<T, (T())> *); // #cwg368-g
  template<typename T> int g(S<T, true ? T() : T()> *); // #cwg368-g-2
  struct X {};
  int n = g<X>(0); // #cwg368-g-call
  // cxx98-17-error@#cwg368-g-call {{no matching function for call to 'g'}}
  //   cxx98-17-note@#cwg368-g {{candidate template ignored: substitution failure [with T = X]: a non-type template parameter cannot have type 'X' before C++20}}
  //   cxx98-17-note@#cwg368-g-2 {{candidate template ignored: substitution failure [with T = X]: a non-type template parameter cannot have type 'X' before C++20}}
  // cxx20-23-error@#cwg368-g-call {{call to 'g' is ambiguous}}
  //   cxx20-23-note@#cwg368-g {{candidate function [with T = cwg368::X]}}
  //   cxx20-23-note@#cwg368-g-2 {{candidate function [with T = cwg368::X]}}
} // namespace cwg368

// cwg370: na

namespace cwg372 { // cwg372: no
  namespace example1 {
    template<typename T> struct X {
    protected:
      typedef T Type; // #cwg372-ex1-Type
    };
    template<typename T> struct Y {};

    // FIXME: These two are valid; deriving from T1<T> gives Z1 access to
    // the protected member T1<T>::Type.
    template<typename T,
             template<typename> class T1,
             template<typename> class T2> struct Z1 :
      T1<T>,
      T2<typename T1<T>::Type> {};
      // expected-error@-1 {{'Type' is a protected member of 'cwg372::example1::X<int>'}}
      //   expected-note@#cwg372-z1 {{in instantiation of template class 'cwg372::example1::Z1<int, cwg372::example1::X, cwg372::example1::Y>' requested here}}
      //   expected-note@#cwg372-ex1-Type {{declared protected here}}

    template<typename T,
             template<typename> class T1,
             template<typename> class T2> struct Z2 :
      T2<typename T1<T>::Type>,
      // expected-error@-1 {{'Type' is a protected member of 'cwg372::example1::X<int>'}}
      //   expected-note@#cwg372-z2 {{in instantiation of template class 'cwg372::example1::Z2<int, cwg372::example1::X, cwg372::example1::Y>' requested here}}
      //   expected-note@#cwg372-ex1-Type {{declared protected here}}
      T1<T> {};

    Z1<int, X, Y> z1; // #cwg372-z1
    Z2<int, X, Y> z2; // #cwg372-z2
  }

  namespace example2 {
    struct X {
    private:
      typedef int Type; // #cwg372-ex2-Type
    };
    template<typename T> struct A {
      typename T::Type t;
      // expected-error@-1 {{'Type' is a private member of 'cwg372::example2::X'}}
      //   expected-note@#cwg372-ax {{in instantiation of template class 'cwg372::example2::A<cwg372::example2::X>' requested here}}
      //   expected-note@#cwg372-ex2-Type {{declared private here}}
    };
    A<X> ax; // #cwg372-ax
  }

  namespace example3 {
    struct A {
    protected:
      typedef int N; // #cwg372-N
    };

    template<typename T> struct B {};
    template<typename U> struct C : U, B<typename U::N> {};
    // expected-error@-1 {{'N' is a protected member of 'cwg372::example3::A'}}
    //   expected-note@#cwg372-x {{in instantiation of template class 'cwg372::example3::C<cwg372::example3::A>' requested here}}
    //   expected-note@#cwg372-N {{declared protected here}}
    template<typename U> struct D : B<typename U::N>, U {};
    // expected-error@-1 {{'N' is a protected member of 'cwg372::example3::A'}}
    //   expected-note@#cwg372-y {{in instantiation of template class 'cwg372::example3::D<cwg372::example3::A>' requested here}}
    //   expected-note@#cwg372-N {{declared protected here}}

    C<A> x; // #cwg372-x
    D<A> y; // #cwg372-y
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
      struct B {}; // #cwg372-B-std
    };
    struct D : A::B, A {};
    // expected-error@-1 {{'B' is a protected member of 'cwg372::std_example::A'}}
    //   expected-note@#cwg372-B-std {{declared protected here}}
  }

  // FIXME: This is valid: deriving from A::B gives access to A::B!
  namespace badwolf {
    class A {
    protected:
      struct B; // #cwg372-B
    };
    struct A::B : A {};
    struct C : A::B {};
    // expected-error@-1 {{'B' is a protected member of 'cwg372::badwolf::A'}}
    //   expected-note@#cwg372-B {{declared protected here}}
  }
} // namespace cwg372

namespace cwg373 { // cwg373: 5
  namespace X { int cwg373; }
  struct cwg373 { // #cwg373-struct
    void f() {
      using namespace cwg373::X;
      int k = cwg373;
      // expected-error@-1 {{'cwg373' does not refer to a value}}
      //   expected-note@#cwg373-struct {{declared here}}
      namespace Y = cwg373::X;
      k = Y::cwg373;
    }
  };

  struct A { struct B {}; }; // #cwg373-A
  namespace X = A::B;
  // expected-error@-1 {{expected namespace name}}
  //   expected-note@#cwg373-A {{'A' declared here}}
  using namespace A::B;
  // expected-error@-1 {{expected namespace name}}
  //   expected-note@#cwg373-A {{'A' declared here}}
} // namespace cwg373

namespace cwg374 { // cwg374: 7
                  // NB 2.9 c++11
  namespace N {
    template<typename T> void f();
    template<typename T> struct A { void f(); };
  }
  template<> void N::f<char>() {}
  template<> void N::A<char>::f() {}
  template<> struct N::A<int> {};
} // namespace cwg374

// cwg375: dup 345
// cwg376: na

namespace cwg377 { // cwg377: 2.7
  enum E {
  // expected-error@-1 {{enumeration values exceed range of largest integer}}
    a = -__LONG_LONG_MAX__ - 1,
    // cxx98-error@-1 {{'long long' is a C++11 extension}}
    b = 2 * (unsigned long long)__LONG_LONG_MAX__
    // cxx98-error@-1 {{'long long' is a C++11 extension}}
    // cxx98-error@-2 {{'long long' is a C++11 extension}}
  };
} // namespace cwg377

// cwg378: dup 276
// cwg379: na

namespace cwg381 { // cwg381: 2.7
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
    e.B::a = 0;
    /* expected-error@-1 {{ambiguous conversion from derived class 'E' to base class 'cwg381::B':
    struct cwg381::E -> C -> B
    struct cwg381::E -> D -> B}} */
    F f;
    f.A::a = 1;
  }
} // namespace cwg381

namespace cwg382 { // cwg382: 2.7 c++11
  // FIXME: Should we allow this in C++98 mode?
  struct A { typedef int T; };
  typename A::T t;
  // cxx98-error@-1 {{'typename' occurs outside of a template}}
  typename cwg382::A a;
  // cxx98-error@-1 {{'typename' occurs outside of a template}}
  typename A b;
  // expected-error@-1 {{expected a qualified name after 'typename'}}
} // namespace cwg382

namespace cwg383 { // cwg383: 2.7
  struct A { A &operator=(const A&); };
  struct B { ~B(); };
  union C { C &operator=(const C&); };
  union D { ~D(); };
  static_assert(!__is_pod(A) && !__is_pod(B) && !__is_pod(C) && !__is_pod(D), "");
} // namespace cwg383

namespace cwg384 { // cwg384: 2.7
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
} // namespace cwg384

namespace cwg385 { // cwg385: 2.8
  struct A { protected: void f(); };
  struct B : A { using A::f; };
  struct C : A { void g(B b) { b.f(); } };
  void h(B b) { b.f(); }

  struct D { int n; }; // #cwg385-n
  struct E : protected D {}; // #cwg385-E
  struct F : E { friend int i(E); };
  int i(E e) { return e.n; }
  // expected-error@-1 {{'n' is a protected member of 'cwg385::D'}}
  //   expected-note@#cwg385-E {{constrained by protected inheritance here}}
  //   expected-note@#cwg385-n {{member is declared here}}
} // namespace cwg385

namespace cwg386 { // cwg386: no
namespace example1 {
namespace N1 {
// Binds name 'f' in N1. Target scope is N1.
template<typename T> void f( T* x ) {
  // ... other stuff ...
  delete x;
}
}

namespace N2 {
// Bind name 'f' in N2. When a single search find this declaration,
// it's replaced with N1::f declaration.
using N1::f;

// According to _N4988_.[dcl.meaning]/3.3:
// `f<int>` is not a qualified-id, so its target scope is N2.
// `f<int>` is a template-id, so 'f' undergoes (unqualified) lookup.
// Search performed by unqualified lookup finds N1::f via using-declaration,
// but this result is not considered, because it's not nominable in N2,
// which is because its target scope is N1.
// So unqualified lookup doesn't find anything, making this declaration ill-formed.
template<> void f<int>( int* );
// expected-error@-1 {{no function template matches function template specialization 'f'}}

class Test {
  ~Test() { }
  // According to _N4988_.[dcl.meaning]/2.2:
  // `f<>` is a template-id and not a template declaration,
  // so its terminal name 'f' undergoes (unqualified) lookup.
  // Search in N2 performed by unqualified lookup finds
  // (single) N1::f declaration via using-declaration.
  // N1::f is replaced with N1::f<> specialization after deduction,
  // and this is the result of the unqualified lookup.
  // This friend declaration correspond to the result of the lookup.
  // All lookup results target the same scope, which is N1,
  // so target scope of this friend declaration is also N1.
  // FIXME: This is well-formed.
  friend void f<>( Test* x );
  // expected-error@-1 {{no function template matches function template specialization 'f'}}
};
}
} // namespace example1

namespace example2 {
namespace N1 {
// Binds name 'f' in N1. Target scope is N1.
void f(); // #cwg386-ex2-N1-f
}

namespace N2 {
// Bind name 'f' in N2. When a single search finds this declaration,
// it's replaced with N1::f declaration.
using N1::f; // #cwg386-ex2-using
class A {
  // According to _N4988_.[dcl.meaning]/2.2:
  // `N2::f` is a qualified-id, so its terminal name 'f' undergoes (qualified) lookup.
  // Search in N2 performed by qualified lookup finds N1::f via using-declaration,
  // which is the (only) result of qualified lookup.
  // This friend declaration corresponds to the result of the lookup.
  // All lookup results target the same scope, which is N1,
  // so target scope of this friend declaration is also N1.
  // FIXME: This is well-formed.
  friend void N2::f();
  // expected-error@-1 {{cannot befriend target of using declaration}}
  //   expected-note@#cwg386-ex2-N1-f {{target of using declaration}}
  //   expected-note@#cwg386-ex2-using {{using declaration}}
};
}
} // namespace example2
} // namespace cwg386

namespace cwg387 { // cwg387: 2.8
  namespace old {
    template<typename T> class number {
      number(int); // #cwg387-number-ctor
      friend number gcd(number &x, number &y) {}
    };

    void g() {
      number<double> a(3);
      // expected-error@-1 {{calling a private constructor of class 'cwg387::old::number<double>'}}
      //   expected-note@#cwg387-number-ctor {{implicitly declared private here}}
      number<double> b(4);
      // expected-error@-1 {{calling a private constructor of class 'cwg387::old::number<double>'}}
      //   expected-note@#cwg387-number-ctor {{implicitly declared private here}}
      a = gcd(a, b);
      b = gcd(3, 4);
      // expected-error@-1 {{use of undeclared identifier 'gcd'}}
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
      b = gcd(3, 4);
      // expected-error@-1 {{use of undeclared identifier 'gcd'}}
    }
  }
} // namespace cwg387

// FIXME: cwg388 needs libc++abi test

namespace cwg389 { // cwg389: no
  struct S {
    typedef struct {} A;
    typedef enum {} B;
    typedef struct {} const C; // #cwg389-C
    typedef enum {} const D; // #cwg389-D
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

  typedef struct {} *WithoutLinkage1; // #cwg389-no-link-1
  typedef enum {} const WithoutLinkage2; // #cwg389-no-link-2
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

  typedef T<WithoutLinkage1> BadArg1;
  // expected-error@-1 {{template argument uses unnamed type}}
  //   expected-note@#cwg389-no-link-1 {{unnamed type used in template argument was declared here}}
  typedef T<WithoutLinkage2> BadArg2;
  // expected-error@-1 {{template argument uses unnamed type}}
  //   expected-note@#cwg389-no-link-2 {{unnamed type used in template argument was declared here}}
  typedef T<WithoutLinkage3> BadArg3;
  // expected-error@-1 {{template argument uses unnamed type}}
  //   expected-note@#cwg389-C {{unnamed type used in template argument was declared here}}
  typedef T<WithoutLinkage4> BadArg4;
  // expected-error@-1 {{template argument uses unnamed type}}
  //   expected-note@#cwg389-D {{unnamed type used in template argument was declared here}}
  typedef T<WithoutLinkage5> BadArg5;
  // expected-error@-1 {{template argument uses unnamed type}}
  //   expected-note@#cwg389-C {{unnamed type used in template argument was declared here}}
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
    extern WithoutLinkage1 cwg389_withoutLinkage1;
    extern WithoutLinkage2 cwg389_withoutLinkage2;
    extern WithoutLinkage3 cwg389_withoutLinkage3;
    extern WithoutLinkage4 cwg389_withoutLinkage4;
    extern WithoutLinkage5 cwg389_withoutLinkage5;
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

    use(&cwg389_withoutLinkage1); use(&cwg389_withoutLinkage2);
    use(&cwg389_withoutLinkage3); use(&cwg389_withoutLinkage4);
    use(&cwg389_withoutLinkage5);

    use(&withoutLinkageDef1); use(&withoutLinkageDef2); use(&withoutLinkageDef3);
    use(&withoutLinkageDef4); use(&withoutLinkageDef5);
  }

  void local() {
    // FIXME: This is ill-formed.
    extern WithoutLinkage1 withoutLinkageLocal;
  }
} // namespace cwg389

namespace cwg390 { // cwg390: 3.3
  template<typename T>
  struct A {
    A() { f(); }
    // expected-warning@-1 {{call to pure virtual member function 'f' has undefined behavior; overrides of 'f' in subclasses are not available in the constructor of 'cwg390::A<int>'}}
    //   expected-note@#cwg390-A-int {{in instantiation of member function 'cwg390::A<int>::A' requested here}}
    //   expected-note@#cwg390-f {{'f' declared here}}
    virtual void f() = 0; // #cwg390-f
    virtual ~A() = 0;
  };
  template<typename T> A<T>::~A() { T::error; }
  // expected-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
  //   expected-note@#cwg390-A-int {{in instantiation of member function 'cwg390::A<int>::~A' requested here}}
  template<typename T> void A<T>::f() { T::error; } // ok, not odr-used
  struct B : A<int> { // #cwg390-A-int
    void f() {}
  } b;
} // namespace cwg390

namespace cwg391 { // cwg391: 2.8 c++11
  // FIXME: Should this apply to C++98 too?
  class A { A(const A&); }; // #cwg391-A
  A fa();
  const A &a = fa();
  // cxx98-error@-1 {{C++98 requires an accessible copy constructor for class 'cwg391::A' when binding a reference to a temporary; was private}}
  //   cxx98-note@#cwg391-A {{implicitly declared private here}}

  struct B { B(const B&) = delete; }; // #cwg391-B
  // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  B fb();
  const B &b = fb();
  // cxx98-error@-1 {{copying variable of type 'B' invokes deleted constructor}}
  //   cxx98-note@#cwg391-B {{'B' has been explicitly marked deleted here}}

  template<typename T>
  struct C {
    C(const C&) { T::error; }
  };
  C<int> fc();
  const C<int> &c = fc();
} // namespace cwg391

// cwg392 is in cwg392.cpp

namespace cwg393 { // cwg393: 2.7
template <typename T>
struct S {};

void f1(S<int (*)[]>);
void f2(S<int (&)[]>);
void g(int(*S<int>::*)[]);

template<typename T>
void sp_assert_convertible( T* ) {}

template<typename T, typename U>
void h() {
  T (*p) [] = (U(*)[])0;
  sp_assert_convertible<T[]>( (U(*)[])0 );
}
} // namespace cwg393

// cwg394: na

namespace cwg395 { // cwg395: 3.0
  struct S {
    template <typename T, int N>(&operator T())[N];
    // expected-error@-1 {{cannot specify any part of a return type in the declaration of a conversion function}}
    template <typename T, int N> operator(T (&)[N])();
    // expected-error@-1 {{expected ')'}}
    //   expected-note@-2 {{to match this '('}}
    // expected-error@-3 {{a type specifier is required for all declarations}}
    template <typename T> operator T *() const { return 0; }
    template <typename T, typename U> operator T U::*() const { return 0; }
    template <typename T, typename U> operator T (U::*)()() const { return 0; }
    // expected-error@-1 {{a type specifier is required for all declarations}}
    // expected-error@-2 {{conversion function cannot have any parameters}}
    // expected-error@-3 {{cannot specify any part of a return type in the declaration of a conversion function}}
    // expected-error@-4 {{conversion function cannot convert to a function type}}

  };

  struct null1_t {
    template <class T, class U> struct ptr_mem_fun_t {
      typedef T (U::*type)();
    };

    template <class T, class U>
    operator typename ptr_mem_fun_t<T, U>::type() const { // #cwg395-conv-func
      return 0;
    }
  } null1;
  int (S::*p)() = null1;
  // expected-error@-1 {{no viable conversion from 'struct null1_t' to 'int (S::*)()'}}
  //   expected-note@#cwg395-conv-func {{candidate template ignored: couldn't infer template argument 'T'}}

  template <typename T> using id = T;
  // cxx98-error@-1 {{alias declarations are a C++11 extension}}

  struct T {
    template <typename T, int N> operator id<T[N]> &();
    template <typename T, typename U> operator id<T (U::*)()>() const;
  };

  struct null2_t {
    template<class T, class U> using ptr_mem_fun_t = T (U::*)();
    // cxx98-error@-1 {{alias declarations are a C++11 extension}}
    template<class T, class U> operator ptr_mem_fun_t<T, U>() const { return 0; };
  } null2;
  int (S::*q)() = null2;
} // namespace cwg395

namespace cwg396 { // cwg396: 3.0
  void f() {
    auto int a();
    // since-cxx11-error@-1 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
    // expected-error@-2 {{illegal storage class on function}}
    int (i); // #cwg396-i
    auto int (i);
    // since-cxx11-error@-1 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
    // expected-error@-2 {{redefinition of 'i'}}
    //   expected-note@#cwg396-i {{previous definition is here}}
  }
} // namespace cwg396

// cwg397: sup 1823

namespace cwg398 { // cwg398: 2.7
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
    template <class T> void f(typename T::Y *) {} // #cwg398-f
    template <class T> void g(X<T::N> *) {} // #cwg398-g
    template <class T> void h(Z<T::template TT> *) {} // #cwg398-h
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
      f<A>(0);
      // expected-error@-1 {{no matching function for call to 'f'}}
      //   expected-note@#cwg398-f {{candidate template ignored: substitution failure [with T = A]: no type named 'Y' in 'cwg398::example2::A'}}
      f<B>(0);
      // expected-error@-1 {{no matching function for call to 'f'}}
      //   expected-note@#cwg398-f {{candidate template ignored: substitution failure [with T = B]: typename specifier refers to non-type member 'Y' in 'cwg398::example2::B'}}
      g<C>(0);
      // expected-error@-1 {{no matching function for call to 'g'}}
      //   expected-note@#cwg398-g {{candidate template ignored: substitution failure [with T = C]: missing 'typename' prior to dependent type name 'C::N'}}
      h<D>(0);
      // expected-error@-1 {{no matching function for call to 'h'}}
      //   expected-note@#cwg398-h {{candidate template ignored: substitution failure [with T = D]: 'TT' following the 'template' keyword does not refer to a template}}
    }
  }
} // namespace cwg398

namespace cwg399 { // cwg399: 11
                  // NB: reuse cwg244 test
  struct B {}; // #cwg399-B
  struct D : B {};

  D D_object;
  typedef B B_alias;
  B* B_ptr = &D_object;

  void f() {
    D_object.~B();
    // expected-error@-1 {{destructor type 'cwg399::B' in object destruction expression does not match the type 'D' of the object being destroyed}}
    //   expected-note@#cwg399-B {{type 'cwg399::B' found by destructor name lookup}}
    D_object.B::~B();
    D_object.D::~B(); // FIXME: Missing diagnostic for this.
    B_ptr->~B();
    B_ptr->~B_alias();
    B_ptr->B_alias::~B();
    B_ptr->B_alias::~B_alias();
    B_ptr->cwg399::~B();
    // expected-error@-1 {{no member named '~B' in namespace 'cwg399'}}
    B_ptr->cwg399::~B_alias();
    // expected-error@-1 {{no member named '~B' in namespace 'cwg399'}}
  }

  template<typename T, typename U>
  void f(T *B_ptr, U D_object) {
    D_object.~B(); // FIXME: Missing diagnostic for this.
    D_object.B::~B();
    D_object.D::~B(); // FIXME: Missing diagnostic for this.
    B_ptr->~B();
    B_ptr->~B_alias();
    B_ptr->B_alias::~B();
    B_ptr->B_alias::~B_alias();
    B_ptr->cwg399::~B();
    // expected-error@-1 {{'cwg399' does not refer to a type name in pseudo-destructor expression; expected the name of type 'T'}}
    B_ptr->cwg399::~B_alias();
    // expected-error@-1 {{'cwg399' does not refer to a type name in pseudo-destructor expression; expected the name of type 'T'}}
  }
  template void f<B, D>(B*, D);

  namespace N {
    template<typename T> struct E {};
    typedef E<int> F;
  }
  void g(N::F f) {
    typedef N::F G; // #cwg399-G
    f.~G();
    f.G::~E();
    // expected-error@-1 {{ISO C++ requires the name after '::~' to be found in the same scope as the name before '::~'}}
    f.G::~F();
    // expected-error@-1 {{undeclared identifier 'F' in destructor name}}
    f.G::~G();
    // This is technically ill-formed; E is looked up in 'N::' and names the
    // class template, not the injected-class-name of the class. But that's
    // probably a bug in the standard.
    f.N::F::~E();
    // expected-error@-1 {{ISO C++ requires the name after '::~' to be found in the same scope as the name before '::~'}}
    // This is valid; we look up the second F in the same scope in which we
    // found the first one, that is, 'N::'.
    f.N::F::~F();
    // This is technically ill-formed; G is looked up in 'N::' and is not found.
    // Rejecting this seems correct, but most compilers accept, so we do also.
    f.N::F::~G();
    // expected-error@-1 {{qualified destructor name only found in lexical scope; omit the qualifier to find this type name by unqualified lookup}}
    //   expected-note@#cwg399-G {{type 'G' (aka 'E<int>') found by destructor name lookup}}
  }

  // Bizarrely, compilers perform lookup in the scope for qualified destructor
  // names, if the nested-name-specifier is non-dependent. Ensure we diagnose
  // this.
  namespace QualifiedLookupInScope {
    namespace N {
      template <typename> struct S { struct Inner {}; };
    }
    template <typename U> void f(typename N::S<U>::Inner *p) {
      typedef typename N::S<U>::Inner T;
      p->::cwg399::QualifiedLookupInScope::N::S<U>::Inner::~T();
      // expected-error@-1 {{no type named 'T' in 'cwg399::QualifiedLookupInScope::N::S<int>'}}
      //   expected-note@#cwg399-f {{in instantiation of function template specialization 'cwg399::QualifiedLookupInScope::f<int>' requested here}}
    }
    template void f<int>(N::S<int>::Inner *); // #cwg399-f

    template <typename U> void g(U *p) {
      typedef U T;
      p->T::~T();
      p->U::~T();
      p->::cwg399::QualifiedLookupInScope::N::S<int>::Inner::~T();
      // expected-error@-1 {{'T' does not refer to a type name in pseudo-destructor expression; expected the name of type 'U'}}
    }
    template void g(N::S<int>::Inner *);
  }
} // namespace cwg399
