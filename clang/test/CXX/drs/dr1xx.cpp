// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98,cxx98-11,cxx98-14,cxx98-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,cxx98-11,cxx98-14,cxx98-17,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,cxx98-14,cxx98-17,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx17,cxx98-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr100 { // dr100: yes
  template<const char (*)[4]> struct A {}; // #dr100-A
  template<const char (&)[4]> struct B {}; // #dr100-B
  template<const char *> struct C {}; // #dr100-C
  template<const char &> struct D {}; // #dr100-D
  A<&"foo"> a; // #dr100-a
  // cxx98-14-error@#dr100-a {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#dr100-A {{template parameter is declared here}}
  // since-cxx17-error@#dr100-a {{pointer to string literal is not allowed in a template argument}}
  B<"bar"> b; // #dr100-b
  // cxx98-14-error@#dr100-b {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#dr100-B {{template parameter is declared here}}
  // since-cxx17-error@#dr100-b {{reference to string literal is not allowed in a template argument}}
  C<"baz"> c; // #dr100-c
  // cxx98-14-error@#dr100-c {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#dr100-C {{template parameter is declared here}}
  // since-cxx17-error@#dr100-c {{pointer to subobject of string literal is not allowed in a template argument}}
  D<*"quux"> d; // #dr100-d
  // cxx98-14-error@#dr100-d {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#dr100-D {{template parameter is declared here}}
  // since-cxx17-error@#dr100-d {{reference to subobject of string literal is not allowed in a template argument}}
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
    template<typename T> T f(T a, T b) { return a + b; }
    // expected-error@-1 {{call to function 'operator+' that is neither visible in the template definition nor found by argument-dependent lookup}}
    // expected-note@#dr102-instantiation {{in instantiation of function template specialization 'dr102::A::f<dr102::B::S>' requested here}}
    // expected-note@#dr102-operator-plus {{'operator+' should be declared prior to the call site or in namespace 'dr102::B'}}
  }
  namespace B {
    struct S {};
  }
  B::S operator+(B::S, B::S); // #dr102-operator-plus
  template B::S A::f(B::S, B::S); // #dr102-instantiation
}

// dr103: na
// dr104: na lib
// dr105: na

namespace dr106 { // dr106: sup 540
  typedef int &r1;
  typedef r1 &r1;
  typedef const r1 r1;
  // expected-warning@-1 {{'const' qualifier on reference type 'r1' (aka 'int &') has no effect}}
  typedef const r1 &r1;
  // expected-warning@-1 {{'const' qualifier on reference type 'r1' (aka 'int &') has no effect}}

  typedef const int &r2;
  typedef r2 &r2;
  typedef const r2 r2;
  // expected-warning@-1 {{'const' qualifier on reference type 'r2' (aka 'const int &') has no effect}}
  typedef const r2 &r2;
  // expected-warning@-1 {{'const' qualifier on reference type 'r2' (aka 'const int &') has no effect}}
}

namespace dr107 { // dr107: yes
  struct S {};
  extern "C" S operator+(S, S) { return S(); }
}

namespace dr108 { // dr108: 2.9
  template<typename T> struct A {
    struct B { typedef int X; };
    B::X x;
    // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name B::X; implicit 'typename' is a C++20 extension}}
    struct C : B { X x; };
    // expected-error@-1 {{unknown type name 'X'}}
  };
  template<> struct A<int>::B { int X; };
}

namespace dr109 { // dr109: yes
  struct A { template<typename T> void f(T); };
  template<typename T> struct B : T {
    using T::template f;
    // expected-error@-1 {{'template' keyword not permitted here}}
    using T::template f<int>;
    // expected-error@-1 {{'template' keyword not permitted here}}
    // expected-error@-2 {{using declaration cannot refer to a template specialization}}
    // FIXME: We shouldn't suggest using the 'template' keyword in a location where it's not valid.
    using T::f<int>;
    // expected-error@-1 {{use 'template' keyword to treat 'f' as a dependent template name}}
    // expected-error@-2 {{using declaration cannot refer to a template specialization}}
    void g() { this->f<int>(123); }
    // expected-error@-1 {{use 'template' keyword to treat 'f' as a dependent template name}}
  };
}

namespace dr111 { // dr111: dup 535
  struct A { A(); A(volatile A&, int = 0); A(A&, const char * = "foo"); };
  struct B : A { B(); }; // #dr111-B
  const B b1;
  B b2(b1);
  // expected-error@-1 {{no matching constructor for initialization of 'B'}}
  // expected-note@#dr111-B {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const B') would lose const qualifier}}
  // expected-note@#dr111-B {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
}

namespace dr112 { // dr112: yes
  struct T { int n; };
  typedef T Arr[1];

  const T a1[1] = {}; // #dr112-a1
  volatile T a2[1] = {};
  const Arr a3 = {}; // #dr112-a3
  volatile Arr a4 = {};
  template<const volatile T*> struct X {};
  // FIXME: Test this somehow in C++11 and on.
  X<a1> x1;
  // cxx98-error@-1 {{non-type template argument referring to object 'a1' with internal linkage is a C++11 extension}}
  // cxx98-note@#dr112-a1 {{non-type template argument refers to object here}}
  X<a2> x2;
  X<a3> x3;
  // cxx98-error@-1 {{non-type template argument referring to object 'a3' with internal linkage is a C++11 extension}}
  // cxx98-note@#dr112-a3 {{non-type template argument refers to object here}}
  X<a4> x4;
}

namespace dr113 { // dr113: yes
  extern void (*p)();
  void f() {
    no_such_function();
    // expected-error@-1 {{use of undeclared identifier 'no_such_function'}}
    p();
  }
  void g();
  void (*p)() = &g;
}

namespace dr114 { // dr114: yes
  struct A {
    virtual void f(int) = 0; // #dr114-A-f
  };
  struct B : A {
    template<typename T> void f(T);
    void g() { f(0); }
  } b;
  // expected-error@-1 {{variable type 'struct B' is an abstract class}}
  // expected-note@#dr114-A-f {{unimplemented pure virtual method 'f' in 'B'}}
}

namespace dr115 { // dr115: 3.0
  template<typename T> int f(T); // #dr115-f
  template<typename T> int g(T); // #dr115-g
  template<typename T> int g(T, int); // #dr115-g-int

  int k1 = f(&f);
  // expected-error@-1 {{no matching function for call to 'f'}}
  // expected-note@#dr115-f {{candidate template ignored: couldn't infer template argument 'T'}}
  int k2 = f(&f<int>);
  int k3 = f(&g<int>);
  // expected-error@-1 {{no matching function for call to 'f'}}
  // expected-note@#dr115-f {{candidate template ignored: couldn't infer template argument 'T'}}

  void h() {
    (void)&f;
    // expected-error@-1 {{address of overloaded function 'f' cannot be cast to type 'void'}}
    // expected-note@#dr115-f {{candidate function template}}
    (void)&f<int>;
    (void)&g<int>;
    // expected-error@-1 {{address of overloaded function 'g' cannot be cast to type 'void'}}
    // expected-note@#dr115-g-int {{candidate function template}}
    // expected-note@#dr115-g {{candidate function template}}

    &f;
    // expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
    // expected-note@#dr115-f {{possible target for call}}
    &f<int>;
    // expected-warning@-1 {{expression result unused}}
    &g<int>;
    // expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
    // expected-note@#dr115-g-int {{possible target for call}}
    // expected-note@#dr115-g {{possible target for call}}
  }

  struct S {
    template<typename T> static int f(T);
    template<typename T> static int g(T);
    template<typename T> static int g(T, int);
  } s;

  int k4 = f(&s.f);
  // expected-error@-1 {{cannot create a non-constant pointer to member function}}
  int k5 = f(&s.f<int>);
  int k6 = f(&s.g<int>);
  // expected-error@-1 {{cannot create a non-constant pointer to member function}}

  void i() {
    (void)&s.f;
    // expected-error@-1 {{cannot create a non-constant pointer to member function}}
    (void)&s.f<int>;
    (void)&s.g<int>;
    // expected-error@-1 {{cannot create a non-constant pointer to member function}}

    &s.f;
    // expected-error@-1 {{cannot create a non-constant pointer to member function}}
    &s.f<int>;
    // expected-warning@-1 {{expression result unused}}
    &s.g<int>;
    // expected-error@-1 {{cannot create a non-constant pointer to member function}}
  }

  struct T {
    template<typename T> int f(T);
    template<typename T> int g(T);
    template<typename T> int g(T, int);
  } t;

  int k7 = f(&s.f);
  // expected-error@-1 {{cannot create a non-constant pointer to member function}}
  int k8 = f(&s.f<int>);
  int k9 = f(&s.g<int>);
  // expected-error@-1 {{cannot create a non-constant pointer to member function}}

  void j() {
    (void)&s.f;
    // expected-error@-1 {{cannot create a non-constant pointer to member function}}
    (void)&s.f<int>;
    (void)&s.g<int>;
    // expected-error@-1 {{cannot create a non-constant pointer to member function}}

    &s.f;
    // expected-error@-1 {{cannot create a non-constant pointer to member function}}
    &s.f<int>;
    // expected-warning@-1 {{expression result unused}}
    &s.g<int>;
    // expected-error@-1 {{cannot create a non-constant pointer to member function}}
  }

#if __cplusplus >= 201103L
  // Special case kicks in only if a template argument list is specified.
  template<typename T=int> void with_default(); // #dr115-with-default
  int k10 = f(&with_default);
  // expected-error@-1 {{no matching function for call to 'f'}}
  // expected-note@#dr115-f {{candidate template ignored: couldn't infer template argument 'T'}}
  int k11 = f(&with_default<>);
  void k() {
    (void)&with_default;
    // expected-error@-1 {{address of overloaded function 'with_default' cannot be cast to type 'void'}}
    // expected-note@#dr115-with-default {{candidate function template}}
    (void)&with_default<>;
    &with_default;
    // expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
    // expected-note@#dr115-with-default {{possible target for call}}
    &with_default<>;
    // expected-warning@-1 {{expression result unused}}
  }
#endif
}

namespace dr116 { // dr116: yes
  template<int> struct A {};
  template<int N> void f(A<N>) {} // #dr116-f-N
  template<int M> void f(A<M>) {}
  // expected-error@-1 {{redefinition of 'f'}}
  // expected-note@#dr116-f-N {{previous definition is here}}
  template<typename T> void f(A<sizeof(T)>) {} // #dr116-f-T
  template<typename U> void f(A<sizeof(U)>) {}
  // expected-error@-1 {{redefinition of 'f'}}
  // expected-note@#dr116-f-T {{previous definition is here}}
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
    T::Y<T> y;
    // expected-error@-1 {{use 'template' keyword to treat 'Y' as a dependent template name}}
    // cxx98-17-error@-2 {{missing 'typename' prior to dependent type name T::Y; implicit 'typename' is a C++20 extension}}
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
struct dr125_A { struct dr125_B {}; }; // #dr125_B
dr125_A::dr125_B dr125_C();
namespace dr125_B { dr125_A dr125_C(); }
namespace dr125 {
  struct X {
    friend dr125_A::dr125_B (::dr125_C)(); // ok
    friend dr125_A (::dr125_B::dr125_C)(); // ok
    friend dr125_A::dr125_B::dr125_C(); // #dr125_C
    // expected-error@#dr125_C {{missing return type for function 'dr125_C'; did you mean the constructor name 'dr125_B'?}}
    // cxx98-error@#dr125_C {{'dr125_B' is missing exception specification 'throw()'}}
    //   cxx98-note@#dr125_B {{previous declaration is here}}
    // since-cxx11-error@#dr125_C {{'dr125_B' is missing exception specification 'noexcept'}}
    //   since-cxx11-note@#dr125_B {{previous declaration is here}}
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
  struct C {};
  struct D : C {};
  struct E : private C { friend class A; friend class B; };
  struct F : protected C {};
  struct G : C {};
  struct H : D, G {};

#if __cplusplus <= 201402L
  struct A {
    virtual void cp() throw(C*);
    virtual void dp() throw(C*);
    virtual void ep() throw(C*); // #dr126-ep
    virtual void fp() throw(C*); // #dr126-fp
    virtual void gp() throw(C*);
    virtual void hp() throw(C*); // #dr126-hp

    virtual void cr() throw(C&);
    virtual void dr() throw(C&);
    virtual void er() throw(C&); // #dr126-er
    virtual void fr() throw(C&); // #dr126-fr
    virtual void gr() throw(C&);
    virtual void hr() throw(C&); // #dr126-hr

    virtual void pv() throw(void*);

    virtual void np() throw(C*);
    virtual void npm() throw(int C::*);
    virtual void nr() throw(C*&); // #dr126-nr
    virtual void ncr() throw(C*const&);

    virtual void ref1() throw(C *const&);
    virtual void ref2() throw(C *);

    virtual void v() throw(int);
    virtual void w() throw(const int);
    virtual void x() throw(int*); // #dr126-x
    virtual void y() throw(const int*);
    virtual void z() throw(int); // #dr126-z
  };
  struct B : A {
    virtual void cp() throw(C*);
    virtual void dp() throw(D*);
    virtual void ep() throw(E*);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx98-14-note@#dr126-ep {{overridden virtual function is here}}
    virtual void fp() throw(F*);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx98-14-note@#dr126-fp {{overridden virtual function is here}}
    virtual void gp() throw(G*);
    virtual void hp() throw(H*);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx98-14-note@#dr126-hp {{overridden virtual function is here}}

    virtual void cr() throw(C&);
    virtual void dr() throw(D&);
    virtual void er() throw(E&);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx98-14-note@#dr126-er {{overridden virtual function is here}}
    virtual void fr() throw(F&);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx98-14-note@#dr126-fr {{overridden virtual function is here}}
    virtual void gr() throw(G&);
    virtual void hr() throw(H&);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx98-14-note@#dr126-hr {{overridden virtual function is here}}

    virtual void pv() throw(C*);

#if __cplusplus >= 201103L
    using nullptr_t = decltype(nullptr);
    virtual void np() throw(nullptr_t);
    virtual void npm() throw(nullptr_t&);
    virtual void nr() throw(nullptr_t);
    // cxx11-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx11-14-note@#dr126-nr {{overridden virtual function is here}}
    virtual void ncr() throw(nullptr_t);
#endif // __cplusplus >= 201103L

    virtual void ref1() throw(D *const &);
    virtual void ref2() throw(D *);

    virtual void v() throw(const int);
    virtual void w() throw(int);
    virtual void x() throw(const int*);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx98-14-note@#dr126-x {{overridden virtual function is here}}
    virtual void y() throw(int*); // ok
    virtual void z() throw(long);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    // cxx98-14-note@#dr126-z {{overridden virtual function is here}}
  };
#endif // __cplusplus <= 201402L
  void f() throw(int);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
}

namespace dr127 { // dr127: 2.9
  __extension__ typedef __decltype(sizeof(0)) size_t;
  template<typename T> struct A {
    A() { throw 0; }
    void *operator new(size_t, const char * = 0);
    void operator delete(void *, const char *) { T::error; } // #dr127-delete-const-char
    // expected-error@#dr127-delete-const-char {{type 'void' cannot be used prior to '::' because it has no members}}
    //   expected-note@#dr127-p {{in instantiation of member function 'dr127::A<void>::operator delete' requested here}}
    // expected-error@#dr127-delete-const-char {{type 'int' cannot be used prior to '::' because it has no members}}
    //   expected-note@#dr127-q {{in instantiation of member function 'dr127::A<int>::operator delete' requested here}}
    void operator delete(void *) { T::error; }
  };
  A<void> *p = new A<void>; // #dr127-p
  A<int> *q = new ("") A<int>; // #dr127-q
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
  void f(int, int, int = 0); // #dr136-f
  void g(int, int, int); // #dr136-g
  struct A {
    friend void f(int, int = 0, int);
    // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
    // expected-note@#dr136-f {{previous declaration is here}}
    friend void g(int, int, int = 0);
    // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
    // expected-note@#dr136-g {{previous declaration is here}}
    friend void h(int, int, int = 0);
    // expected-error@-1 {{friend declaration specifying a default argument must be a definition}}
    friend void i(int, int, int = 0) {} // #dr136-A-i
    friend void j(int, int, int = 0) {}
    operator int();
  };
  void i(int, int, int);
  // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
  // expected-note@#dr136-A-i {{previous declaration is here}}
  void q() {
    j(A(), A()); // ok, has default argument
  }
  extern "C" void k(int, int, int, int); // #dr136-k 
  namespace NSA {
  struct A {
    friend void dr136::k(int, int, int, int = 0);
    // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
    // expected-note@#dr136-k {{previous declaration is here}}
  };
  }
  namespace NSB {
  struct A {
    friend void dr136::k(int, int, int = 0, int); // #dr136-friend-k
    // expected-error@#dr136-friend-k {{friend declaration specifying a default argument must be the only declaration}}
    //   expected-note@#dr136-k {{previous declaration is here}}
    // expected-error@#dr136-friend-k {{missing default argument on parameter}}
  };
  }
  struct B {
    void f(int); // #dr136-B-f
  };
  struct C {
    friend void B::f(int = 0);
    // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
    // expected-note@#dr136-B-f {{previous declaration is here}}
  };
}

namespace dr137 { // dr137: yes
  extern void *p;
  extern const void *cp;
  extern volatile void *vp;
  extern const volatile void *cvp;
  int *q = static_cast<int*>(p);
  int *qc = static_cast<int*>(cp);
  // expected-error@-1 {{static_cast from 'const void *' to 'int *' casts away qualifiers}}
  int *qv = static_cast<int*>(vp);
  // expected-error@-1 {{static_cast from 'volatile void *' to 'int *' casts away qualifiers}}
  int *qcv = static_cast<int*>(cvp);
  // expected-error@-1 {{static_cast from 'const volatile void *' to 'int *' casts away qualifiers}}
  const int *cq = static_cast<const int*>(p);
  const int *cqc = static_cast<const int*>(cp);
  const int *cqv = static_cast<const int*>(vp);
  // expected-error@-1 {{static_cast from 'volatile void *' to 'const int *' casts away qualifiers}}
  const int *cqcv = static_cast<const int*>(cvp);
  // expected-error@-1 {{static_cast from 'const volatile void *' to 'const int *' casts away qualifiers}}
  const volatile int *cvq = static_cast<const volatile int*>(p);
  const volatile int *cvqc = static_cast<const volatile int*>(cp);
  const volatile int *cvqv = static_cast<const volatile int*>(vp);
  const volatile int *cvqcv = static_cast<const volatile int*>(cvp);
}

namespace dr139 { // dr139: yes
  namespace example1 {
    typedef int f; // #dr139-typedef-f
    struct A {
      friend void f(A &);
      // expected-error@-1 {{redefinition of 'f' as different kind of symbol}}
      // expected-note@#dr139-typedef-f {{previous definition is here}}
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
  void f(int *const) {} // #dr140-f-first
  void f(int[3]) {}
  // expected-error@-1 {{redefinition of 'f'}}
  // expected-note@#dr140-f-first {{previous definition is here}}
  void g(const int);
  void g(int n) { n = 2; }
}

namespace dr141 { // dr141: 3.1
  template<typename T> void f();
  template<typename T> struct S { int n; }; // #dr141-S
  struct A : S<int> {
    template<typename T> void f();
    template<typename T> struct S {}; // #dr141-A-S
  } a;
  struct B : S<int> {} b;
  void g() {
    a.f<int>();
    (void)a.S<int>::n; // #dr141-a
    // cxx98-error@#dr141-a {{lookup of 'S' in member access expression is ambiguous; using member of 'struct A'}}
    //   cxx98-note@#dr141-A-S {{lookup in the object type 'struct A' refers here}}
    //   cxx98-note@#dr141-S {{lookup from the current scope refers here}}
    // expected-error@#dr141-a {{no member named 'n' in 'dr141::A::S<int>'; did you mean '::dr141::S<int>::n'?}}
    //   expected-note@#dr141-S {{'::dr141::S<int>::n' declared here}}
    // FIXME: we issue a useful diagnostic first, then some bogus ones.
    b.f<int>();
    // expected-error@-1 {{no member named 'f' in 'dr141::B'}}
    // expected-error@-2 +{{}}
    (void)b.S<int>::n;
  }
  template<typename T> struct C {
    T t;
    void g() {
      t.f<int>();
      // expected-error@-1 {{use 'template' keyword to treat 'f' as a dependent template name}}
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

namespace dr142 { // dr142: 2.8
  class B { // #dr142-B
  public:
    int mi; // #dr142-B-mi
    static int si; // #dr142-B-si
  };
  class D : private B { // #dr142-D
  };
  class DD : public D {
    void f();
  };
  void DD::f() {
    mi = 3;
    // expected-error@-1 {{'mi' is a private member of 'dr142::B'}}
    // expected-note@#dr142-D {{constrained by private inheritance here}}
    // expected-note@#dr142-B-mi {{member is declared here}}
    si = 3;
    // expected-error@-1 {{'si' is a private member of 'dr142::B'}}
    // expected-note@#dr142-D {{constrained by private inheritance here}}
    // expected-note@#dr142-B-si {{member is declared here}}
    B b_old;
    // expected-error@-1 {{'B' is a private member of 'dr142::B'}}
    // expected-note@#dr142-D {{constrained by private inheritance here}}
    // expected-note@#dr142-B {{member is declared here}}
    dr142::B b;
    b.mi = 3;
    b.si = 3;
    B::si = 3;
    // expected-error@-1 {{'B' is a private member of 'dr142::B'}}
    // expected-note@#dr142-D {{constrained by private inheritance here}}
    // expected-note@#dr142-B {{member is declared here}}
    dr142::B::si = 3;
    B *bp1_old = this; // #dr142-bp1_old
    // expected-error@#dr142-bp1_old {{'B' is a private member of 'dr142::B'}}
    //   expected-note@#dr142-D {{constrained by private inheritance here}}
    //   expected-note@#dr142-B {{member is declared here}}
    // expected-error@#dr142-bp1_old {{cannot cast 'dr142::DD' to its private base class 'B'}}
    //   expected-note@#dr142-D {{declared private here}}
    dr142::B *bp1 = this;
    // expected-error@-1 {{cannot cast 'dr142::DD' to its private base class 'dr142::B'}}
    // expected-note@#dr142-D {{declared private here}}
    B *bp2_old = (B*)this; // #dr142-bp2_old
    // expected-error@#dr142-bp2_old {{'B' is a private member of 'dr142::B'}}
    //   expected-note@#dr142-D {{constrained by private inheritance here}}
    //   expected-note@#dr142-B {{member is declared here}}
    // expected-error@#dr142-bp2_old {{'B' is a private member of 'dr142::B'}}
    //   expected-note@#dr142-D {{constrained by private inheritance here}}
    //   expected-note@#dr142-B {{member is declared here}}
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
    f(x);
    // expected-error@-1 {{use of undeclared identifier 'f'}}
  }
}

namespace dr145 { // dr145: yes
  void f(bool b) {
    ++b;
    // cxx98-14-warning@-1 {{incrementing expression of type bool is deprecated and incompatible with C++17}}
    // since-cxx17-error@-2 {{ISO C++17 does not allow incrementing expression of type bool}}
    b++;
    // cxx98-14-warning@-1 {{incrementing expression of type bool is deprecated and incompatible with C++17}}
    // since-cxx17-error@-2 {{ISO C++17 does not allow incrementing expression of type bool}}
  }
}

namespace dr147 { // dr147: yes
  namespace example1 {
    template<typename> struct A {
      template<typename T> A(T);
    };
    // Per core issue 1435, this is ill-formed because A<int>::A<int> does not
    // name the injected-class-name. (A<int>::A does, though.)
    template<> template<> A<int>::A<int>(int) {}
    // expected-error@-1 {{out-of-line constructor for 'A' cannot have template arguments}}
    template<> template<> A<float>::A(float) {}
  }
  namespace example2 {
    struct A { A(); };
    struct B : A { B(); };
    A::A a1;
    // expected-error@-1 {{qualified reference to 'A' is a constructor name rather than a type in this context}}
    B::A a2;
  }
  namespace example3 {
    template<typename> struct A {
      template<typename T> A(T);
      static A a;
    };
    template<> A<int>::A<int>(A<int>::a);
    // expected-error@-1 {{qualified reference to 'A' is a constructor name rather than a template name in this context}}
  }
}

namespace dr148 { // dr148: yes
  struct A { int A::*p; };
  int check1[__is_pod(int(A::*)) ? 1 : -1];
  int check2[__is_pod(A) ? 1 : -1];
}

// dr149: na

namespace dr151 { // dr151: 3.1
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
    A(); // #dr152-A-ctor
    explicit A(const A&); // #dr152-A-explicit-ctor
  };
  A a1 = A();
  // cxx98-14-error@-1 {{no matching constructor for initialization of 'A'}}
  // cxx98-14-note@#dr152-A-explicit-ctor {{explicit constructor is not a candidate}}
  // cxx98-14-note@#dr152-A-ctor {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  A a2((A()));

  A &f();
  A a3 = f();
  // expected-error@-1 {{no matching constructor for initialization of 'A'}}
  // expected-note@#dr152-A-explicit-ctor {{explicit constructor is not a candidate}}
  // expected-note@#dr152-A-ctor {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  A a4(f());
}

// dr153: na

namespace dr154 { // dr154: yes
  union { int a; };
  // expected-error@-1 {{nonymous unions at namespace or global scope must be declared 'static'}}
  namespace {
    union { int b; };
  }
  static union { int c; };
}

namespace dr155 { // dr155: dup 632
  struct S { int n; } s = { { 1 } };
  // expected-warning@-1 {{braces around scalar initializer}}
}

// dr158 is in its own file.

namespace dr159 { // dr159: 3.5
  namespace X { void f(); }
  void f();
  void dr159::f() {}
  // expected-warning@-1 {{extra qualification on member 'f'}}
  void dr159::X::f() {}
}

// dr160: na

namespace dr161 { // dr161: 3.1
  class A {
  protected:
    struct B { int n; } b; // #dr161-B
    static B bs;
    void f(); // #dr161-f
    static void sf();
  };
  struct C : A {};
  struct D : A {
    void g(C c) {
      (void)b.n;
      B b1;
      C::B b2; // ok, accessible as a member of A
      (void)&C::b;
      // expected-error@-1 {{'b' is a protected member of 'dr161::A'}}
      // expected-note@#dr161-B {{declared protected here}}
      (void)&C::bs;
      (void)c.b;
      // expected-error@-1 {{'b' is a protected member of 'dr161::A'}}
      // expected-note@#dr161-B {{declared protected here}}
      (void)c.bs;
      f();
      sf();
      c.f();
      // expected-error@-1 {{protected}}
      // expected-note@#dr161-f {{declared protected here}}
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
      int &a = (&A::f)(0);
      // FIXME: expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
      char &b = (&A::f)('0');
      // expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
    }
  };

  int &c = (&A::f)(0);
  // FIXME: expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
  char &d = (&A::f)('0');
  // expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
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

namespace dr166 { // dr166: 2.9
  namespace A { class X; }

  template<typename T> int f(T t) { return t.n; }
  int g(A::X);
  template<typename T> int h(T t) { return t.n; }
  // expected-error@-1 {{'n' is a private member of 'dr166::A::X'}}
  // expected-note@#dr166-h-instantiation {{in instantiation of function template specialization 'dr166::h<dr166::A::X>' requested here}}
  // expected-note@#dr166-X-n {{implicitly declared private here}}
  int i(A::X);

  namespace A {
    class X {
      friend int f<X>(X);
      friend int dr166::g(X);
      friend int h(X);
      friend int i(X);
      int n; // #dr166-X-n
    };

    int h(X x) { return x.n; }
    int i(X x) { return x.n; }
  }

  template int f(A::X);
  int g(A::X x) { return x.n; }
  template int h(A::X); // #dr166-h-instantiation
  int i(A::X x) { return x.n; }
  // expected-error@-1 {{'n' is a private member of 'dr166::A::X'}}
  // expected-note@#dr166-X-n {{implicitly declared private here}}
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
    template<typename> static int n;
    // cxx98-11-error@-1 {{variable templates are a C++14 extension}}
  };
  struct D : A<int>, B {
    using A<int>::n;
    using B::C<int>;
    // expected-error@-1 {{using declaration cannot refer to a template specialization}}
    using B::f<int>;
    // expected-error@-1 {{using declaration cannot refer to a template specialization}}
    using B::n<int>;
    // expected-error@-1 {{using declaration cannot refer to a template specialization}}
  };
}

namespace { // dr171: 3.4
  int dr171a;
}
int dr171b; // #dr171b-int
namespace dr171 {
  extern "C" void dr171a();
  extern "C" void dr171b();
  // expected-error@-1 {{declaration of 'dr171b' with C language linkage conflicts with declaration in global scope}}
  // expected-note@#dr171b-int {{declared in global scope here}}
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

  enum { e = (unsigned long long)-1 / 2 };
  // cxx98-error@-1 {{'long long' is a C++11 extension}}
  int check7a[sizeof(e) == sizeof(long) ? 1 : -1];
  int check7b[-e < 0 ? 1 : -1];

  enum { f = (unsigned long long)-1 / 2 + 1 };
  // cxx98-error@-1 {{'long long' is a C++11 extension}}
  int check8a[sizeof(f) == sizeof(unsigned long) ? 1 : -1];
  int check8b[-f > 0 ? 1 : -1];
}

namespace dr173 { // dr173: yes
  int check[('0' + 1 == '1' && '0' + 2 == '2' && '0' + 3 == '3' &&
             '0' + 4 == '4' && '0' + 5 == '5' && '0' + 6 == '6' &&
             '0' + 7 == '7' && '0' + 8 == '8' && '0' + 9 == '9') ? 1 : -1];
}

// dr174: sup 1012

namespace dr175 { // dr175: 2.8
  struct A {}; // #dr175-A
  struct B : private A {}; // #dr175-B
  struct C : B {
    A a;
    // expected-error@-1 {{'A' is a private member of 'dr175::A'}}
    // expected-note@#dr175-B {{constrained by private inheritance here}}
    // expected-note@#dr175-A {{member is declared here}}
    dr175::A b;
  };
}

namespace dr176 { // dr176: 3.1
  template<typename T> class Y;
  template<> class Y<int> {
    void f() {
      typedef Y A; // #dr176-A-first
      typedef Y<char> A;
      // expected-error@-1 {{typedef redefinition with different types ('Y<char>' vs 'Y<int>')}}
      // expected-note@#dr176-A-first {{previous definition is here}}
    }
  };

  template<typename T> struct Base {}; // #dr176-Base
  template<typename T> struct Derived : public Base<T> {
    void f() {
      typedef typename Derived::template Base<T> A;
      typedef typename Derived::Base A;
    }
  };
  template struct Derived<int>;

  template<typename T> struct Derived2 : Base<int>, Base<char> {
    typename Derived2::Base b;
    // expected-error@-1 {{member 'Base' found in multiple base classes of different types}}
    // expected-note@#dr176-Base {{member type 'dr176::Base<int>' found by ambiguous name lookup}}
    // expected-note@#dr176-Base {{member type 'dr176::Base<char>' found by ambiguous name lookup}}
    typename Derived2::Base<double> d;
  };

  template<typename T> class X { // #dr176-X
    X *p1;
    X<T> *p2;
    X<int> *p3;
    dr176::X *p4; // #dr176-p4
    // cxx98-14-error@#dr176-p4 {{use of class template 'dr176::X' requires template arguments}}
    //  cxx98-14-note@#dr176-X {{template is declared here}}
    // since-cxx17-error@#dr176-p4 {{use of class template 'X' requires template arguments; argument deduction not allowed in non-static class member}}
    //  since-cxx17-note@#dr176-X {{template is declared here}}
  };
}

namespace dr177 { // dr177: yes
  struct B {};
  struct A {
    A(A &); // #dr177-A-copy-ctor
    A(const B &); // #dr177-A-ctor-from-B
  };
  B b;
  A a = b;
  // cxx98-14-error@-1 {{no viable constructor copying variable of type 'A'}}
  // cxx98-14-note@#dr177-A-copy-ctor {{candidate constructor not viable: expects an lvalue for 1st argument}}
  // cxx98-14-note@#dr177-A-ctor-from-B {{candidate constructor not viable: no known conversion from 'A' to 'const B &' for 1st argument}}

  struct C { C(C&); }; // #dr177-C-copy-ctor
  struct D : C {};
  struct E { operator D(); };
  E e;
  C c = e;
  // expected-error@-1 {{no viable constructor copying variable of type 'D'}}
  // expected-note@#dr177-C-copy-ctor {{candidate constructor not viable: expects an lvalue for 1st argument}}
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
  int n = &f - &f;
  // expected-error@-1 {{arithmetic on pointers to the function type 'void ()'}}
}

namespace dr180 { // dr180: 2.8
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
    template <template X<class T> > struct A { };
    // expected-error@-1 +{{}}
    template <template X<class T> > void f(A<X>) { }
    // expected-error@-1 +{{}}
  }

  namespace Y {
    template <template <class T> class X> struct A { };
    template <template <class T> class X> void f(A<X>) { }
  }
}

namespace dr182 { // dr182: 14
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
    typename B<int>::X x;
    // cxx98-error@-1 {{'typename' occurs outside of a template}}
  };
}

namespace dr184 { // dr184: yes
  template<typename T = float> struct B {};

  template<template<typename TT = float> class T> struct A {
    void f();
    void g();
  };

  template<template<typename TT> class T> void A<T>::f() { // #dr184-T
    T<> t;
    // expected-error@-1 {{too few template arguments for template template parameter 'T'}}
    // expected-note@#dr184-T {{template is declared here}}
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

int dr191_j;
namespace dr191 { // dr191: yes
  namespace example1 {
    struct outer {
      static int i;
      struct inner {
        void f() {
          struct local {
            void g() {
              i = 5;
            }
          };
        }
      };
    };
  }

  namespace example2 {
    struct S {
      void f() {
        struct local2 {
          void g() {
            dr191_j = 5;
          }
        };
      }
    };
  }
}

// dr193 FIXME: add codegen test

namespace dr194 { // dr194: yes
  struct A {
    A();
    void A();
    // expected-error@-1 {{constructor cannot have a return type}}
  };
  struct B {
    void B();
    // expected-error@-1 {{constructor cannot have a return type}}
    B();
  };
  struct C {
    inline explicit C(int) {}
  };
}

namespace dr195 { // dr195: yes
  void f();
  int *p = (int*)&f;
  // cxx98-error@-1 {{cast between pointer-to-function and pointer-to-object is an extension}}
  void (*q)() = (void(*)())&p;
  // cxx98-error@-1 {{cast between pointer-to-function and pointer-to-object is an extension}}
}

namespace dr197 { // dr197: yes
  char &f(char);

  template <class T> void g(T t) {
    char &a = f(1);
    char &b = f(T(1));
    // expected-error@-1 {{non-const lvalue reference to type 'char' cannot bind to a value of unrelated type 'int'}}
    // expected-note@#dr197-g-e-call {{in instantiation of function template specialization 'dr197::g<dr197::E>' requested here}}
    char &c = f(t);
    // expected-error@-1 {{non-const lvalue reference to type 'char' cannot bind to a value of unrelated type 'int'}}
  }

  void f(int);

  enum E { e };
  int &f(E);

  void h() {
    g('a');
    g(2);
    g(e); // #dr197-g-e-call
  }
}

namespace dr198 { // dr198: yes
  struct A {
    int n;
    struct B {
      int m[sizeof(n)];
      // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
      int f() { return n; }
      // expected-error@-1 {{use of non-static data member 'n' of 'A' from nested type 'B'}}
    };
    struct C;
    struct D;
  };
  struct A::C {
    int m[sizeof(n)];
    // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
    int f() { return n; }
    // expected-error@-1 {{use of non-static data member 'n' of 'A' from nested type 'C'}}
  };
  struct A::D : A {
    int m[sizeof(n)];
    // cxx98-error@-1 {{invalid use of non-static data member 'n'}}
    int f() { return n; }
  };
}

// dr199 FIXME: add codegen test
