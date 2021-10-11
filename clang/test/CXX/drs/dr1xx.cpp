// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr100 { // dr100: yes
  template<const char (*)[4]> struct A {}; // expected-note 0-1{{declared here}}
  template<const char (&)[4]> struct B {}; // expected-note 0-1{{declared here}}
  template<const char *> struct C {}; // expected-note 0-1{{declared here}}
  template<const char &> struct D {}; // expected-note 0-1{{declared here}}
  A<&"foo"> a; // #100a
  B<"bar"> b; // #100b
  C<"baz"> c; // #100c
  D<*"quux"> d; // #100d
#if __cplusplus < 201703L
  // expected-error@#100a {{does not refer to any declaration}}
  // expected-error@#100b {{does not refer to any declaration}}
  // expected-error@#100c {{does not refer to any declaration}}
  // expected-error@#100d {{does not refer to any declaration}}
#else
  // expected-error@#100a {{pointer to string literal is not allowed in a template argument}}
  // expected-error@#100b {{reference to string literal is not allowed in a template argument}}
  // expected-error@#100c {{pointer to subobject of string literal is not allowed in a template argument}}
  // expected-error@#100d {{reference to subobject of string literal is not allowed in a template argument}}
#endif
}

namespace dr101 { // dr101: 3.5
  extern "C" void dr101_f();
  typedef unsigned size_t;
  namespace X {
    extern "C" void dr101_f();
    typedef unsigned size_t;
  }
  using X::dr101_f;
  using X::size_t;
  extern "C" void dr101_f();
  typedef unsigned size_t;
}

namespace dr102 { // dr102: yes
  namespace A {
    template<typename T> T f(T a, T b) { return a + b; } // expected-error {{neither visible in the template definition nor found by argument-dependent lookup}}
  }
  namespace B {
    struct S {};
  }
  B::S operator+(B::S, B::S); // expected-note {{should be declared prior to the call site or in namespace 'dr102::B'}}
  template B::S A::f(B::S, B::S); // expected-note {{in instantiation of}}
}

// dr103: na
// dr104: na lib
// dr105: na

namespace dr106 { // dr106: sup 540
  typedef int &r1;
  typedef r1 &r1;
  typedef const r1 r1; // expected-warning {{has no effect}}
  typedef const r1 &r1; // expected-warning {{has no effect}}

  typedef const int &r2;
  typedef r2 &r2;
  typedef const r2 r2; // expected-warning {{has no effect}}
  typedef const r2 &r2; // expected-warning {{has no effect}}
}

namespace dr107 { // dr107: yes
  struct S {};
  extern "C" S operator+(S, S) { return S(); }
}

namespace dr108 { // dr108: yes
  template<typename T> struct A {
    struct B { typedef int X; };
    B::X x; // expected-error {{missing 'typename'}}
    struct C : B { X x; }; // expected-error {{unknown type name}}
  };
  template<> struct A<int>::B { int X; };
}

namespace dr109 { // dr109: yes
  struct A { template<typename T> void f(T); };
  template<typename T> struct B : T {
    using T::template f; // expected-error {{'template' keyword not permitted here}}
    using T::template f<int>; // expected-error {{'template' keyword not permitted here}} expected-error {{using declaration cannot refer to a template specialization}}
    // FIXME: We shouldn't suggest using the 'template' keyword in a location where it's not valid.
    using T::f<int>; // expected-error {{use 'template' keyword}} expected-error {{using declaration cannot refer to a template specialization}}
    void g() { this->f<int>(123); } // expected-error {{use 'template' keyword}}
  };
}

namespace dr111 { // dr111: dup 535
  struct A { A(); A(volatile A&, int = 0); A(A&, const char * = "foo"); };
  struct B : A { B(); }; // expected-note +{{would lose const qualifier}} expected-note {{requires 0 arguments}}
  const B b1;
  B b2(b1); // expected-error {{no matching constructor}}
}

namespace dr112 { // dr112: yes
  struct T { int n; };
  typedef T Arr[1];

  const T a1[1] = {};
  volatile T a2[1] = {};
  const Arr a3 = {};
  volatile Arr a4 = {};
  template<const volatile T*> struct X {};
  X<a1> x1;
  X<a2> x2;
  X<a3> x3;
  X<a4> x4;
#if __cplusplus < 201103L
  // expected-error@-5 {{internal linkage}} expected-note@-10 {{here}}
  // expected-error@-4 {{internal linkage}} expected-note@-9 {{here}}
#else
  // FIXME: Test this somehow.
#endif
}

namespace dr113 { // dr113: yes
  extern void (*p)();
  void f() {
    no_such_function(); // expected-error {{undeclared}}
    p();
  }
  void g();
  void (*p)() = &g;
}

namespace dr114 { // dr114: yes
  struct A {
    virtual void f(int) = 0; // expected-note {{unimplemented}}
  };
  struct B : A {
    template<typename T> void f(T);
    void g() { f(0); }
  } b; // expected-error {{abstract}}
}

namespace dr115 { // dr115: yes
  template<typename T> int f(T); // expected-note +{{}}
  template<typename T> int g(T); // expected-note +{{}}
  template<typename T> int g(T, int); // expected-note +{{}}

  int k1 = f(&f); // expected-error {{no match}}
  int k2 = f(&f<int>);
  int k3 = f(&g<int>); // expected-error {{no match}}

  void h() {
    (void)&f; // expected-error {{address of overloaded function 'f' cannot be cast to type 'void'}}
    (void)&f<int>;
    (void)&g<int>; // expected-error {{address of overloaded function 'g' cannot be cast to type 'void'}}

    &f; // expected-error {{reference to overloaded function could not be resolved}}
    &f<int>; // expected-warning {{unused}}
    &g<int>; // expected-error {{reference to overloaded function could not be resolved}}
  }

  struct S {
    template<typename T> static int f(T);
    template<typename T> static int g(T);
    template<typename T> static int g(T, int);
  } s;

  int k4 = f(&s.f); // expected-error {{non-constant pointer to member}}
  int k5 = f(&s.f<int>);
  int k6 = f(&s.g<int>); // expected-error {{non-constant pointer to member}}

  void i() {
    (void)&s.f; // expected-error {{non-constant pointer to member}}
    (void)&s.f<int>;
    (void)&s.g<int>; // expected-error {{non-constant pointer to member}}

    &s.f; // expected-error {{non-constant pointer to member}}
    &s.f<int>; // expected-warning {{unused}}
    &s.g<int>; // expected-error {{non-constant pointer to member}}
  }

  struct T {
    template<typename T> int f(T);
    template<typename T> int g(T);
    template<typename T> int g(T, int);
  } t;

  int k7 = f(&s.f); // expected-error {{non-constant pointer to member}}
  int k8 = f(&s.f<int>);
  int k9 = f(&s.g<int>); // expected-error {{non-constant pointer to member}}

  void j() {
    (void)&s.f; // expected-error {{non-constant pointer to member}}
    (void)&s.f<int>;
    (void)&s.g<int>; // expected-error {{non-constant pointer to member}}

    &s.f; // expected-error {{non-constant pointer to member}}
    &s.f<int>; // expected-warning {{unused}}
    &s.g<int>; // expected-error {{non-constant pointer to member}}
  }

#if __cplusplus >= 201103L
  // Special case kicks in only if a template argument list is specified.
  template<typename T=int> void with_default(); // expected-note +{{}}
  int k10 = f(&with_default); // expected-error {{no matching function}}
  int k11 = f(&with_default<>);
  void k() {
    (void)&with_default; // expected-error {{overloaded function}}
    (void)&with_default<>;
    &with_default; // expected-error {{overloaded function}}
    &with_default<>; // expected-warning {{unused}}
  }
#endif
}

namespace dr116 { // dr116: yes
  template<int> struct A {};
  template<int N> void f(A<N>) {} // expected-note {{previous}}
  template<int M> void f(A<M>) {} // expected-error {{redefinition}}
  template<typename T> void f(A<sizeof(T)>) {} // expected-note {{previous}}
  template<typename U> void f(A<sizeof(U)>) {} // expected-error {{redefinition}}
}

// dr117: na
// dr118 is in its own file.
// dr119: na
// dr120: na

namespace dr121 { // dr121: yes
  struct X {
    template<typename T> struct Y {};
  };
  template<typename T> struct Z {
    X::Y<T> x;
    T::Y<T> y; // expected-error +{{}}
  };
  Z<X> z;
}

namespace dr122 { // dr122: yes
  template<typename T> void f();
  void g() { f<int>(); }
}

// dr123: na
// dr124: dup 201

// dr125: yes
struct dr125_A { struct dr125_B {}; }; // expected-note {{here}}
dr125_A::dr125_B dr125_C();
namespace dr125_B { dr125_A dr125_C(); }
namespace dr125 {
  struct X {
    friend dr125_A::dr125_B (::dr125_C)(); // ok
    friend dr125_A (::dr125_B::dr125_C)(); // ok
    friend dr125_A::dr125_B::dr125_C(); // expected-error {{did you mean the constructor name 'dr125_B'?}}
    // expected-error@-1 {{missing exception specification}}
  };
}

namespace dr126 { // dr126: partial
  // FIXME: We do not yet generate correct code for this change:
  // eg:
  //   catch (void*&) should catch void* but not int*
  //   catch (void*) and catch (void*const&) should catch both
  // Likewise:
  //   catch (Base *&) should catch Base* but not Derived*
  //   catch (Base *) should catch both
  // In each case, we emit the same code for both catches.
  //
  // The ABI does not let us represent the language rule in the unwind tables.
  // So, when catching by non-const (or volatile) reference to pointer, we
  // should compare the exception type to the caught type and only accept an
  // exact match.
#if __cplusplus <= 201402L
  struct C {};
  struct D : C {};
  struct E : private C { friend class A; friend class B; };
  struct F : protected C {};
  struct G : C {};
  struct H : D, G {};

  struct A {
    virtual void cp() throw(C*);
    virtual void dp() throw(C*);
    virtual void ep() throw(C*); // expected-note {{overridden}}
    virtual void fp() throw(C*); // expected-note {{overridden}}
    virtual void gp() throw(C*);
    virtual void hp() throw(C*); // expected-note {{overridden}}

    virtual void cr() throw(C&);
    virtual void dr() throw(C&);
    virtual void er() throw(C&); // expected-note {{overridden}}
    virtual void fr() throw(C&); // expected-note {{overridden}}
    virtual void gr() throw(C&);
    virtual void hr() throw(C&); // expected-note {{overridden}}

    virtual void pv() throw(void*);

#if __cplusplus >= 201103L
    virtual void np() throw(C*);
    virtual void npm() throw(int C::*);
    virtual void nr() throw(C*&); // expected-note {{overridden}}
    virtual void ncr() throw(C*const&);
#endif

    virtual void ref1() throw(C *const&);
    virtual void ref2() throw(C *);

    virtual void v() throw(int);
    virtual void w() throw(const int);
    virtual void x() throw(int*); // expected-note {{overridden}}
    virtual void y() throw(const int*);
    virtual void z() throw(int); // expected-note {{overridden}}
  };
  struct B : A {
    virtual void cp() throw(C*);
    virtual void dp() throw(D*);
    virtual void ep() throw(E*); // expected-error {{more lax}}
    virtual void fp() throw(F*); // expected-error {{more lax}}
    virtual void gp() throw(G*);
    virtual void hp() throw(H*); // expected-error {{more lax}}

    virtual void cr() throw(C&);
    virtual void dr() throw(D&);
    virtual void er() throw(E&); // expected-error {{more lax}}
    virtual void fr() throw(F&); // expected-error {{more lax}}
    virtual void gr() throw(G&);
    virtual void hr() throw(H&); // expected-error {{more lax}}

    virtual void pv() throw(C*);

#if __cplusplus >= 201103L
    using nullptr_t = decltype(nullptr);
    virtual void np() throw(nullptr_t);
    virtual void npm() throw(nullptr_t&);
    virtual void nr() throw(nullptr_t); // expected-error {{more lax}}
    virtual void ncr() throw(nullptr_t);
#endif

    virtual void ref1() throw(D *const &);
    virtual void ref2() throw(D *);

    virtual void v() throw(const int);
    virtual void w() throw(int);
    virtual void x() throw(const int*); // expected-error {{more lax}}
    virtual void y() throw(int*); // ok
    virtual void z() throw(long); // expected-error {{more lax}}
  };
#else
  void f() throw(int); // expected-error {{ISO C++17 does not allow}} expected-note {{use 'noexcept}}
#endif
}

namespace dr127 { // dr127: yes
  __extension__ typedef __decltype(sizeof(0)) size_t;
  template<typename T> struct A {
    A() { throw 0; }
    void *operator new(size_t, const char * = 0);
    void operator delete(void *, const char *) { T::error; } // expected-error 2{{no members}}
    void operator delete(void *) { T::error; }
  };
  A<void> *p = new A<void>; // expected-note {{instantiat}}
  A<int> *q = new ("") A<int>; // expected-note {{instantiat}}
}

namespace dr128 { // dr128: yes
  enum E1 { e1 } x = e1;
  enum E2 { e2 } y = static_cast<E2>(x), z = static_cast<E2>(e1);
}

// dr129: dup 616
// dr130: na

namespace dr131 { // dr131: sup P1949
  const char *a_with_\u0e8c = "\u0e8c";
  const char *b_with_\u0e8d = "\u0e8d";
  const char *c_with_\u0e8e = "\u0e8e";
}

namespace dr132 { // dr132: no
  void f() {
    extern struct {} x; // ok
    extern struct S {} y; // FIXME: This is invalid.
  }
  static enum { E } e;
}

// dr133: dup 87
// dr134: na

namespace dr135 { // dr135: yes
  struct A {
    A f(A a) { return a; }
    friend A g(A a) { return a; }
    static A h(A a) { return a; }
  };
}

namespace dr136 { // dr136: 3.4
  void f(int, int, int = 0); // expected-note {{previous declaration is here}}
  void g(int, int, int); // expected-note {{previous declaration is here}}
  struct A {
    friend void f(int, int = 0, int); // expected-error {{friend declaration specifying a default argument must be the only declaration}}
    friend void g(int, int, int = 0); // expected-error {{friend declaration specifying a default argument must be the only declaration}}
    friend void h(int, int, int = 0); // expected-error {{friend declaration specifying a default argument must be a definition}}
    friend void i(int, int, int = 0) {} // expected-note {{previous declaration is here}}
    friend void j(int, int, int = 0) {}
    operator int();
  };
  void i(int, int, int); // expected-error {{friend declaration specifying a default argument must be the only declaration}}
  void q() {
    j(A(), A()); // ok, has default argument
  }
  extern "C" void k(int, int, int, int); // expected-note 2{{previous declaration is here}}
  namespace NSA {
  struct A {
    friend void dr136::k(int, int, int, int = 0); // expected-error {{friend declaration specifying a default argument must be the only declaration}}
  };
  }
  namespace NSB {
  struct A {
    friend void dr136::k(int, int, int = 0, int); // expected-error {{missing default argument on parameter}} expected-error {{must be the only declaration}}
  };
  }
  struct B {
    void f(int); // expected-note {{previous declaration is here}}
  };
  struct C {
    friend void B::f(int = 0); // expected-error {{friend declaration specifying a default argument must be the only declaration}}
  };
}

namespace dr137 { // dr137: yes
  extern void *p;
  extern const void *cp;
  extern volatile void *vp;
  extern const volatile void *cvp;
  int *q = static_cast<int*>(p);
  int *qc = static_cast<int*>(cp); // expected-error {{casts away qualifiers}}
  int *qv = static_cast<int*>(vp); // expected-error {{casts away qualifiers}}
  int *qcv = static_cast<int*>(cvp); // expected-error {{casts away qualifiers}}
  const int *cq = static_cast<const int*>(p);
  const int *cqc = static_cast<const int*>(cp);
  const int *cqv = static_cast<const int*>(vp); // expected-error {{casts away qualifiers}}
  const int *cqcv = static_cast<const int*>(cvp); // expected-error {{casts away qualifiers}}
  const volatile int *cvq = static_cast<const volatile int*>(p);
  const volatile int *cvqc = static_cast<const volatile int*>(cp);
  const volatile int *cvqv = static_cast<const volatile int*>(vp);
  const volatile int *cvqcv = static_cast<const volatile int*>(cvp);
}

namespace dr139 { // dr139: yes
  namespace example1 {
    typedef int f; // expected-note {{previous}}
    struct A {
      friend void f(A &); // expected-error {{different kind of symbol}}
    };
  }

  namespace example2 {
    typedef int f;
    namespace N {
      struct A {
        friend void f(A &);
        operator int();
        void g(A a) { int i = f(a); } // ok, f is typedef not friend function
      };
    }
  }
}

namespace dr140 { // dr140: yes
  void f(int *const) {} // expected-note {{previous}}
  void f(int[3]) {} // expected-error {{redefinition}}
  void g(const int);
  void g(int n) { n = 2; }
}

namespace dr141 { // dr141: yes
  template<typename T> void f();
  template<typename T> struct S { int n; }; // expected-note {{'::dr141::S<int>::n' declared here}}
  struct A : S<int> {
    template<typename T> void f();
    template<typename T> struct S {};
  } a;
  struct B : S<int> {} b;
  void g() {
    a.f<int>();
    (void)a.S<int>::n; // expected-error {{no member named 'n' in 'dr141::A::S<int>'; did you mean '::dr141::S<int>::n'?}}
#if __cplusplus < 201103L
    // expected-error@-2 {{ambiguous}}
    // expected-note@-11 {{lookup from the current scope}}
    // expected-note@-9 {{lookup in the object type}}
#endif
    b.f<int>(); // expected-error {{no member}} expected-error +{{}}
    (void)b.S<int>::n;
  }
  template<typename T> struct C {
    T t;
    void g() {
      t.f<int>(); // expected-error {{use 'template'}}
    }
    void h() {
      (void)t.S<int>::n; // ok
    }
    void i() {
      (void)t.S<int>(); // ok!
    }
  };
  void h() { C<B>().h(); } // ok
  struct X {
    template<typename T> void S();
  };
  void i() { C<X>().i(); } // ok!!
}

namespace dr142 { // dr142: yes
  class B { // expected-note +{{here}}
  public:
    int mi; // expected-note +{{here}}
    static int si; // expected-note +{{here}}
  };
  class D : private B { // expected-note +{{here}}
  };
  class DD : public D {
    void f();
  };
  void DD::f() {
    mi = 3; // expected-error {{private member}}
    si = 3; // expected-error {{private member}}
    B b_old; // expected-error {{private member}}
    dr142::B b;
    b.mi = 3;
    b.si = 3;
    B::si = 3; // expected-error {{private member}}
    dr142::B::si = 3;
    B *bp1_old = this; // expected-error {{private member}} expected-error {{private base class}}
    dr142::B *bp1 = this; // expected-error {{private base class}}
    B *bp2_old = (B*)this; // expected-error 2{{private member}}
    dr142::B *bp2 = (dr142::B*)this;
    bp2->mi = 3;
  }
}

namespace dr143 { // dr143: yes
  namespace A { struct X; }
  namespace B { void f(A::X); }
  namespace A {
    struct X { friend void B::f(X); };
  }
  void g(A::X x) {
    f(x); // expected-error {{undeclared identifier 'f'}}
  }
}

namespace dr145 { // dr145: yes
  void f(bool b) {
#if __cplusplus <= 201402L
    ++b; // expected-warning {{deprecated}}
    b++; // expected-warning {{deprecated}}
#else
    ++b; // expected-error {{increment}}
    b++; // expected-error {{increment}}
#endif
  }
}

namespace dr147 { // dr147: yes
  namespace example1 {
    template<typename> struct A {
      template<typename T> A(T);
    };
    // Per core issue 1435, this is ill-formed because A<int>::A<int> does not
    // name the injected-class-name. (A<int>::A does, though.)
    template<> template<> A<int>::A<int>(int) {} // expected-error {{out-of-line constructor for 'A' cannot have template arguments}}
    template<> template<> A<float>::A(float) {}
  }
  namespace example2 {
    struct A { A(); };
    struct B : A { B(); };
    A::A a1; // expected-error {{is a constructor}}
    B::A a2;
  }
  namespace example3 {
    template<typename> struct A {
      template<typename T> A(T);
      static A a;
    };
    template<> A<int>::A<int>(A<int>::a); // expected-error {{is a constructor}}
  }
}

namespace dr148 { // dr148: yes
  struct A { int A::*p; };
  int check1[__is_pod(int(A::*)) ? 1 : -1];
  int check2[__is_pod(A) ? 1 : -1];
}

// dr149: na

namespace dr151 { // dr151: yes
  struct X {};
  typedef int X::*p;
#if __cplusplus < 201103L
#define fold(x) (__builtin_constant_p(0) ? (x) : (x))
#else
#define fold
#endif
  int check[fold(p() == 0) ? 1 : -1];
#undef fold
}

namespace dr152 { // dr152: yes
  struct A {
    A(); // expected-note 0-2{{not viable}}
    explicit A(const A&); // expected-note 1-2{{not a candidate}}
  };
  A a1 = A();
#if __cplusplus <= 201402L
  // expected-error@-2 {{no matching constructor}}
#endif
  A a2((A()));

  A &f();
  A a3 = f(); // expected-error {{no matching constructor}}
  A a4(f());
}

// dr153: na

namespace dr154 { // dr154: yes
  union { int a; }; // expected-error {{must be declared 'static'}}
  namespace {
    union { int b; };
  }
  static union { int c; };
}

namespace dr155 { // dr155: dup 632
  struct S { int n; } s = { { 1 } }; // expected-warning {{braces around scalar initializer}}
}

// dr158 is in its own file.

namespace dr159 { // dr159: 3.5
  namespace X { void f(); }
  void f();
  void dr159::f() {} // expected-warning {{extra qualification}}
  void dr159::X::f() {}
}

// dr160: na

namespace dr161 { // dr161: yes
  class A {
  protected:
    struct B { int n; } b; // expected-note 2{{here}}
    static B bs;
    void f(); // expected-note {{here}}
    static void sf();
  };
  struct C : A {};
  struct D : A {
    void g(C c) {
      (void)b.n;
      B b1;
      C::B b2; // ok, accessible as a member of A
      (void)&C::b; // expected-error {{protected}}
      (void)&C::bs;
      (void)c.b; // expected-error {{protected}}
      (void)c.bs;
      f();
      sf();
      c.f(); // expected-error {{protected}}
      c.sf();
      A::f();
      D::f();
      A::sf();
      C::sf();
      D::sf();
    }
  };
}

namespace dr162 { // dr162: no
  struct A {
    char &f(char);
    static int &f(int);

    void g() {
      int &a = (&A::f)(0); // FIXME: expected-error {{could not be resolved}}
      char &b = (&A::f)('0'); // expected-error {{could not be resolved}}
    }
  };

  int &c = (&A::f)(0); // FIXME: expected-error {{could not be resolved}}
  char &d = (&A::f)('0'); // expected-error {{could not be resolved}}
}

// dr163: na

namespace dr164 { // dr164: yes
  void f(int);
  template <class T> int g(T t) { return f(t); }

  enum E { e };
  int f(E);

  int k = g(e);
}

namespace dr165 { // dr165: no
  namespace N {
    struct A { friend struct B; };
    void f() { void g(); }
  }
  // FIXME: dr1477 says this is ok, dr165 says it's ill-formed
  struct N::B {};
  // FIXME: dr165 says this is ill-formed, but the argument in dr1477 says it's ok
  void N::g() {}
}

namespace dr166 { // dr166: yes
  namespace A { class X; }

  template<typename T> int f(T t) { return t.n; }
  int g(A::X);
  template<typename T> int h(T t) { return t.n; } // expected-error {{private}}
  int i(A::X);

  namespace A {
    class X {
      friend int f<X>(X);
      friend int dr166::g(X);
      friend int h(X);
      friend int i(X);
      int n; // expected-note 2{{here}}
    };

    int h(X x) { return x.n; }
    int i(X x) { return x.n; }
  }

  template int f(A::X);
  int g(A::X x) { return x.n; }
  template int h(A::X); // expected-note {{instantiation}}
  int i(A::X x) { return x.n; } // expected-error {{private}}
}

// dr167: sup 1012

namespace dr168 { // dr168: no
  extern "C" typedef int (*p)();
  extern "C++" typedef int (*q)();
  struct S {
    static int f();
  };
  p a = &S::f; // FIXME: this should fail.
  q b = &S::f;
}

namespace dr169 { // dr169: yes
  template<typename> struct A { int n; };
  struct B {
    template<typename> struct C;
    template<typename> void f();
    template<typename> static int n; // expected-error 0-1{{extension}}
  };
  struct D : A<int>, B {
    using A<int>::n;
    using B::C<int>; // expected-error {{using declaration cannot refer to a template specialization}}
    using B::f<int>; // expected-error {{using declaration cannot refer to a template specialization}}
    using B::n<int>; // expected-error {{using declaration cannot refer to a template specialization}}
  };
}

namespace { // dr171: yes
  int dr171a;
}
int dr171b; // expected-note {{here}}
namespace dr171 {
  extern "C" void dr171a();
  extern "C" void dr171b(); // expected-error {{conflicts}}
}

namespace dr172 { // dr172: yes
  enum { zero };
  int check1[-1 < zero ? 1 : -1];

  enum { x = -1, y = (unsigned int)-1 };
  int check2[sizeof(x) > sizeof(int) ? 1 : -1];

  enum { a = (unsigned int)-1 / 2 };
  int check3a[sizeof(a) == sizeof(int) ? 1 : -1];
  int check3b[-a < 0 ? 1 : -1];

  enum { b = (unsigned int)-1 / 2 + 1 };
  int check4a[sizeof(b) == sizeof(unsigned int) ? 1 : -1];
  int check4b[-b > 0 ? 1 : -1];

  enum { c = (unsigned long)-1 / 2 };
  int check5a[sizeof(c) == sizeof(long) ? 1 : -1];
  int check5b[-c < 0 ? 1 : -1];

  enum { d = (unsigned long)-1 / 2 + 1 };
  int check6a[sizeof(d) == sizeof(unsigned long) ? 1 : -1];
  int check6b[-d > 0 ? 1 : -1];

  enum { e = (unsigned long long)-1 / 2 }; // expected-error 0-1{{extension}}
  int check7a[sizeof(e) == sizeof(long) ? 1 : -1]; // expected-error 0-1{{extension}}
  int check7b[-e < 0 ? 1 : -1];

  enum { f = (unsigned long long)-1 / 2 + 1 }; // expected-error 0-1{{extension}}
  int check8a[sizeof(f) == sizeof(unsigned long) ? 1 : -1]; // expected-error 0-1{{extension}}
  int check8b[-f > 0 ? 1 : -1];
}

namespace dr173 { // dr173: yes
  int check[('0' + 1 == '1' && '0' + 2 == '2' && '0' + 3 == '3' &&
             '0' + 4 == '4' && '0' + 5 == '5' && '0' + 6 == '6' &&
             '0' + 7 == '7' && '0' + 8 == '8' && '0' + 9 == '9') ? 1 : -1];
}

// dr174: sup 1012

namespace dr175 { // dr175: yes
  struct A {}; // expected-note {{here}}
  struct B : private A {}; // expected-note {{constrained by private inheritance}}
  struct C : B {
    A a; // expected-error {{private}}
    dr175::A b;
  };
}

namespace dr176 { // dr176: yes
  template<typename T> class Y;
  template<> class Y<int> {
    void f() {
      typedef Y A; // expected-note {{here}}
      typedef Y<char> A; // expected-error {{different types ('Y<char>' vs 'Y<int>')}}
    }
  };

  template<typename T> struct Base {}; // expected-note 2{{found}}
  template<typename T> struct Derived : public Base<T> {
    void f() {
      typedef typename Derived::template Base<T> A;
      typedef typename Derived::Base A;
    }
  };
  template struct Derived<int>;

  template<typename T> struct Derived2 : Base<int>, Base<char> {
    typename Derived2::Base b; // expected-error {{found in multiple base classes}}
    typename Derived2::Base<double> d;
  };

  template<typename T> class X { // expected-note {{here}}
    X *p1;
    X<T> *p2;
    X<int> *p3;
    dr176::X *p4; // expected-error {{requires template arguments}}
  };
}

namespace dr177 { // dr177: yes
  struct B {};
  struct A {
    A(A &); // expected-note 0-1{{not viable: expects an lvalue}}
    A(const B &); // expected-note 0-1{{not viable: no known conversion from 'A' to}}
  };
  B b;
  A a = b;
#if __cplusplus <= 201402L
  // expected-error@-2 {{no viable constructor copying variable}}
#endif

  struct C { C(C&); }; // expected-note {{not viable: expects an lvalue for 1st argument}}
  struct D : C {};
  struct E { operator D(); };
  E e;
  C c = e; // expected-error {{no viable constructor copying variable of type 'D'}}
}

namespace dr178 { // dr178: yes
  int check[int() == 0 ? 1 : -1];
#if __cplusplus >= 201103L
  static_assert(int{} == 0, "");
  struct S { int a, b; };
  static_assert(S{1}.b == 0, "");
  struct T { constexpr T() : n() {} int n; };
  static_assert(T().n == 0, "");
  struct U : S { constexpr U() : S() {} };
  static_assert(U().b == 0, "");
#endif
}

namespace dr179 { // dr179: yes
  void f();
  int n = &f - &f; // expected-error {{arithmetic on pointers to the function type 'void ()'}}
}

namespace dr180 { // dr180: yes
  template<typename T> struct X : T, T::some_base {
    X() : T::some_type_that_might_be_T(), T::some_base() {}
    friend class T::some_class;
    void f() {
      enum T::some_enum e;
    }
  };
}

namespace dr181 { // dr181: yes
  namespace X {
    template <template X<class T> > struct A { }; // expected-error +{{}}
    template <template X<class T> > void f(A<X>) { } // expected-error +{{}}
  }

  namespace Y {
    template <template <class T> class X> struct A { };
    template <template <class T> class X> void f(A<X>) { }
  }
}

namespace dr182 { // dr182: yes
  template <class T> struct C {
    void f();
    void g();
  };

  template <class T> void C<T>::f() {}
  template <class T> void C<T>::g() {}

  class A {
    class B {};
    void f();
  };

  template void C<A::B>::f();
  template <> void C<A::B>::g();

  void A::f() {
    C<B> cb;
    cb.f();
  }
}

namespace dr183 { // dr183: sup 382
  template<typename T> struct A {};
  template<typename T> struct B {
    typedef int X;
  };
  template<> struct A<int> {
#if __cplusplus <= 199711
    typename B<int>::X x; // expected-error {{'typename' occurs outside of a template}}
#else
    typename B<int>::X x;
#endif
  };
}

namespace dr184 { // dr184: yes
  template<typename T = float> struct B {};

  template<template<typename TT = float> class T> struct A {
    void f();
    void g();
  };

  template<template<typename TT> class T> void A<T>::f() { // expected-note {{here}}
    T<> t; // expected-error {{too few template arguments}}
  }

  template<template<typename TT = char> class T> void A<T>::g() {
    T<> t;
    typedef T<> X;
    typedef T<char> X;
  }

  void h() { A<B>().g(); }
}

// dr185 FIXME: add codegen test

namespace dr187 { // dr187: sup 481
  const int Z = 1;
  template<int X = Z, int Z = X> struct A;
  typedef A<> T;
  typedef A<1, 1> T;
}

namespace dr188 { // dr188: yes
  char c[10];
  int check[sizeof(0, c) == 10 ? 1 : -1];
}

// dr190 FIXME: add codegen test for tbaa

// dr193 FIXME: add codegen test

namespace dr194 { // dr194: yes
  struct A {
    A();
    void A(); // expected-error {{constructor cannot have a return type}}
  };
  struct B {
    void B(); // expected-error {{constructor cannot have a return type}}
    B();
  };
  struct C {
    inline explicit C(int) {}
  };
}

namespace dr195 { // dr195: yes
  void f();
  int *p = (int*)&f; // expected-error 0-1{{extension}}
  void (*q)() = (void(*)())&p; // expected-error 0-1{{extension}}
}

namespace dr197 { // dr197: yes
  char &f(char);

  template <class T> void g(T t) {
    char &a = f(1);
    char &b = f(T(1)); // expected-error {{unrelated type 'int'}}
    char &c = f(t); // expected-error {{unrelated type 'int'}}
  }

  void f(int);

  enum E { e };
  int &f(E);

  void h() {
    g('a');
    g(2);
    g(e); // expected-note {{in instantiation of}}
  }
}

namespace dr198 { // dr198: yes
  struct A {
    int n;
    struct B {
      int m[sizeof(n)];
#if __cplusplus < 201103L
      // expected-error@-2 {{invalid use of non-static data member}}
#endif
      int f() { return n; }
      // expected-error@-1 {{use of non-static data member 'n' of 'A' from nested type 'B'}}
    };
    struct C;
    struct D;
  };
  struct A::C {
    int m[sizeof(n)];
#if __cplusplus < 201103L
    // expected-error@-2 {{invalid use of non-static data member}}
#endif
    int f() { return n; }
    // expected-error@-1 {{use of non-static data member 'n' of 'A' from nested type 'C'}}
  };
  struct A::D : A {
    int m[sizeof(n)];
#if __cplusplus < 201103L
    // expected-error@-2 {{invalid use of non-static data member}}
#endif
    int f() { return n; }
  };
}

// dr199 FIXME: add codegen test
