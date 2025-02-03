// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98,cxx98-11,cxx98-14,cxx98-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,cxx98-11,cxx98-14,cxx98-17,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,cxx98-14,cxx98-17,cxx11-14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx17,cxx98-17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx17 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

#if __cplusplus == 199711L
#define __enable_constant_folding(x) (__builtin_constant_p(x) ? (x) : (x))
#else
#define __enable_constant_folding
#endif

namespace cwg100 { // cwg100: 2.7
  template<const char (*)[4]> struct A {}; // #cwg100-A
  template<const char (&)[4]> struct B {}; // #cwg100-B
  template<const char *> struct C {}; // #cwg100-C
  template<const char &> struct D {}; // #cwg100-D
  A<&"foo"> a; // #cwg100-a
  // cxx98-14-error@#cwg100-a {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#cwg100-A {{template parameter is declared here}}
  // since-cxx17-error@#cwg100-a {{pointer to string literal is not allowed in a template argument}}
  B<"bar"> b; // #cwg100-b
  // cxx98-14-error@#cwg100-b {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#cwg100-B {{template parameter is declared here}}
  // since-cxx17-error@#cwg100-b {{reference to string literal is not allowed in a template argument}}
  C<"baz"> c; // #cwg100-c
  // cxx98-14-error@#cwg100-c {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#cwg100-C {{template parameter is declared here}}
  // since-cxx17-error@#cwg100-c {{pointer to subobject of string literal is not allowed in a template argument}}
  D<*"quux"> d; // #cwg100-d
  // cxx98-14-error@#cwg100-d {{non-type template argument does not refer to any declaration}}
  //   cxx98-14-note@#cwg100-D {{template parameter is declared here}}
  // since-cxx17-error@#cwg100-d {{reference to subobject of string literal is not allowed in a template argument}}
} // namespace cwg100

namespace cwg101 { // cwg101: 3.5
  extern "C" void cwg101_f();
  typedef unsigned size_t;
  namespace X {
    extern "C" void cwg101_f();
    typedef unsigned size_t;
  }
  using X::cwg101_f;
  using X::size_t;
  extern "C" void cwg101_f();
  typedef unsigned size_t;
} // namespace cwg101

namespace cwg102 { // cwg102: 2.7
  namespace A {
    template<typename T> T f(T a, T b) { return a + b; }
    // expected-error@-1 {{call to function 'operator+' that is neither visible in the template definition nor found by argument-dependent lookup}}
    //   expected-note@#cwg102-instantiation {{in instantiation of function template specialization 'cwg102::A::f<cwg102::B::S>' requested here}}
    //   expected-note@#cwg102-operator-plus {{'operator+' should be declared prior to the call site or in namespace 'cwg102::B'}}
  }
  namespace B {
    struct S {};
  }
  B::S operator+(B::S, B::S); // #cwg102-operator-plus
  template B::S A::f(B::S, B::S); // #cwg102-instantiation
} // namespace cwg102

// cwg103: na
// cwg104: na lib
// cwg105: na

namespace cwg106 { // cwg106: sup 540
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
} // namespace cwg106

namespace cwg107 { // cwg107: 2.7
  struct S {};
  extern "C" S operator+(S, S) { return S(); }
} // namespace cwg107

namespace cwg108 { // cwg108: 2.9
  template<typename T> struct A {
    struct B { typedef int X; };
    B::X x;
    // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name B::X; implicit 'typename' is a C++20 extension}}
    struct C : B { X x; };
    // expected-error@-1 {{unknown type name 'X'}}
  };
  template<> struct A<int>::B { int X; };
} // namespace cwg108

namespace cwg109 { // cwg109: 2.8
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
} // namespace cwg109

namespace cwg110 { // cwg110: 2.8
template <typename T>
void f(T);

class f {};

template <typename T>
void f(T, T);

class f g;
void (*h)(int) = static_cast<void(*)(int)>(f);
void (*i)(int, int) = static_cast<void(*)(int, int)>(f);
} // namespace cwg110

namespace cwg111 { // cwg111: dup 535
  struct A { A(); A(volatile A&, int = 0); A(A&, const char * = "foo"); };
  struct B : A { B(); }; // #cwg111-B
  const B b1;
  B b2(b1);
  // expected-error@-1 {{no matching constructor for initialization of 'B'}}
  //   expected-note@#cwg111-B {{candidate constructor (the implicit copy constructor) not viable: 1st argument ('const B') would lose const qualifier}}
  //   expected-note@#cwg111-B {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
} // namespace cwg111

namespace cwg112 { // cwg112: 3.1
  struct T { int n; };
  typedef T Arr[1];

  const T a1[1] = {}; // #cwg112-a1
  volatile T a2[1] = {};
  const Arr a3 = {}; // #cwg112-a3
  volatile Arr a4 = {};
  template<const volatile T*> struct X {};
  // FIXME: Test this somehow in C++11 and on.
  X<a1> x1;
  // cxx98-error@-1 {{non-type template argument referring to object 'a1' with internal linkage is a C++11 extension}}
  //   cxx98-note@#cwg112-a1 {{non-type template argument refers to object here}}
  X<a2> x2;
  X<a3> x3;
  // cxx98-error@-1 {{non-type template argument referring to object 'a3' with internal linkage is a C++11 extension}}
  //   cxx98-note@#cwg112-a3 {{non-type template argument refers to object here}}
  X<a4> x4;
} // namespace cwg112

namespace cwg113 { // cwg113: 2.7
  extern void (*p)();
  void f() {
    no_such_function();
    // expected-error@-1 {{use of undeclared identifier 'no_such_function'}}
    p();
  }
  void g();
  void (*p)() = &g;
} // namespace cwg113

namespace cwg114 { // cwg114: 2.7
  struct A {
    virtual void f(int) = 0; // #cwg114-A-f
  };
  struct B : A {
    template<typename T> void f(T);
    void g() { f(0); }
  } b;
  // expected-error@-1 {{variable type 'struct B' is an abstract class}}
  //   expected-note@#cwg114-A-f {{unimplemented pure virtual method 'f' in 'B'}}
} // namespace cwg114

namespace cwg115 { // cwg115: 3.0
  template<typename T> int f(T); // #cwg115-f
  template<typename T> int g(T); // #cwg115-g
  template<typename T> int g(T, int); // #cwg115-g-int

  int k1 = f(&f);
  // expected-error@-1 {{no matching function for call to 'f'}}
  //   expected-note@#cwg115-f {{candidate template ignored: couldn't infer template argument 'T'}}
  int k2 = f(&f<int>);
  int k3 = f(&g<int>);
  // expected-error@-1 {{no matching function for call to 'f'}}
  //   expected-note@#cwg115-f {{candidate template ignored: couldn't infer template argument 'T'}}

  void h() {
    (void)&f;
    // expected-error@-1 {{address of overloaded function 'f' cannot be cast to type 'void'}}
    //   expected-note@#cwg115-f {{candidate function template}}
    (void)&f<int>;
    (void)&g<int>;
    // expected-error@-1 {{address of overloaded function 'g' cannot be cast to type 'void'}}
    //   expected-note@#cwg115-g-int {{candidate function template}}
    //   expected-note@#cwg115-g {{candidate function template}}

    &f;
    // expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
    //   expected-note@#cwg115-f {{possible target for call}}
    &f<int>;
    // expected-warning@-1 {{expression result unused}}
    &g<int>;
    // expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
    //   expected-note@#cwg115-g-int {{possible target for call}}
    //   expected-note@#cwg115-g {{possible target for call}}
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
  template<typename T=int> void with_default(); // #cwg115-with-default
  int k10 = f(&with_default);
  // since-cxx11-error@-1 {{no matching function for call to 'f'}}
  //   since-cxx11-note@#cwg115-f {{candidate template ignored: couldn't infer template argument 'T'}}
  int k11 = f(&with_default<>);
  void k() {
    (void)&with_default;
    // since-cxx11-error@-1 {{address of overloaded function 'with_default' cannot be cast to type 'void'}}
    //   since-cxx11-note@#cwg115-with-default {{candidate function template}}
    (void)&with_default<>;
    &with_default;
    // since-cxx11-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
    //   since-cxx11-note@#cwg115-with-default {{possible target for call}}
    &with_default<>;
    // since-cxx11-warning@-1 {{expression result unused}}
  }
#endif
} // namespace cwg115

namespace cwg116 { // cwg116: 2.7
  template<int> struct A {};
  template<int N> void f(A<N>) {} // #cwg116-f-N
  template<int M> void f(A<M>) {}
  // expected-error@-1 {{redefinition of 'f'}}
  //   expected-note@#cwg116-f-N {{previous definition is here}}
  template<typename T> void f(A<sizeof(T)>) {} // #cwg116-f-T
  template<typename U> void f(A<sizeof(U)>) {}
  // expected-error@-1 {{redefinition of 'f'}}
  //   expected-note@#cwg116-f-T {{previous definition is here}}
} // namespace cwg116

// cwg117: na
// cwg118 is in cwg118.cpp
// cwg119: na
// cwg120: na

namespace cwg121 { // cwg121: 2.7
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
} // namespace cwg121

namespace cwg122 { // cwg122: 2.7
  template<typename T> void f();
  void g() { f<int>(); }
} // namespace cwg122

// cwg123: na
// cwg124 is in cwg124.cpp

// cwg125: 2.7
struct cwg125_A { struct cwg125_B {}; }; // #cwg125_B
cwg125_A::cwg125_B cwg125_C();
namespace cwg125_B { cwg125_A cwg125_C(); }
namespace cwg125 {
  struct X {
    friend cwg125_A::cwg125_B (::cwg125_C)(); // ok
    friend cwg125_A (::cwg125_B::cwg125_C)(); // ok
    friend cwg125_A::cwg125_B::cwg125_C(); // #cwg125_C
    // expected-error@#cwg125_C {{missing return type for function 'cwg125_C'; did you mean the constructor name 'cwg125_B'?}}
    // cxx98-error@#cwg125_C {{'cwg125_B' is missing exception specification 'throw()'}}
    //   cxx98-note@#cwg125_B {{previous declaration is here}}
    // since-cxx11-error@#cwg125_C {{'cwg125_B' is missing exception specification 'noexcept'}}
    //   since-cxx11-note@#cwg125_B {{previous declaration is here}}
  };
} // namespace cwg125

namespace cwg126 { // cwg126: partial
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
    virtual void ep() throw(C*); // #cwg126-ep
    virtual void fp() throw(C*); // #cwg126-fp
    virtual void gp() throw(C*);
    virtual void hp() throw(C*); // #cwg126-hp

    virtual void cr() throw(C&);
    virtual void dr() throw(C&);
    virtual void er() throw(C&); // #cwg126-er
    virtual void fr() throw(C&); // #cwg126-fr
    virtual void gr() throw(C&);
    virtual void hr() throw(C&); // #cwg126-hr

    virtual void pv() throw(void*);

    virtual void np() throw(C*);
    virtual void npm() throw(int C::*);
    virtual void nr() throw(C*&); // #cwg126-nr
    virtual void ncr() throw(C*const&);

    virtual void ref1() throw(C *const&);
    virtual void ref2() throw(C *);

    virtual void v() throw(int);
    virtual void w() throw(const int);
    virtual void x() throw(int*); // #cwg126-x
    virtual void y() throw(const int*);
    virtual void z() throw(int); // #cwg126-z
  };
  struct B : A {
    virtual void cp() throw(C*);
    virtual void dp() throw(D*);
    virtual void ep() throw(E*);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx98-14-note@#cwg126-ep {{overridden virtual function is here}}
    virtual void fp() throw(F*);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx98-14-note@#cwg126-fp {{overridden virtual function is here}}
    virtual void gp() throw(G*);
    virtual void hp() throw(H*);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx98-14-note@#cwg126-hp {{overridden virtual function is here}}

    virtual void cr() throw(C&);
    virtual void dr() throw(D&);
    virtual void er() throw(E&);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx98-14-note@#cwg126-er {{overridden virtual function is here}}
    virtual void fr() throw(F&);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx98-14-note@#cwg126-fr {{overridden virtual function is here}}
    virtual void gr() throw(G&);
    virtual void hr() throw(H&);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx98-14-note@#cwg126-hr {{overridden virtual function is here}}

    virtual void pv() throw(C*);

#if __cplusplus >= 201103L
    using nullptr_t = decltype(nullptr);
    virtual void np() throw(nullptr_t);
    virtual void npm() throw(nullptr_t&);
    virtual void nr() throw(nullptr_t);
    // cxx11-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx11-14-note@#cwg126-nr {{overridden virtual function is here}}
    virtual void ncr() throw(nullptr_t);
#endif // __cplusplus >= 201103L

    virtual void ref1() throw(D *const &);
    virtual void ref2() throw(D *);

    virtual void v() throw(const int);
    virtual void w() throw(int);
    virtual void x() throw(const int*);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx98-14-note@#cwg126-x {{overridden virtual function is here}}
    virtual void y() throw(int*); // ok
    virtual void z() throw(long);
    // cxx98-14-error@-1 {{exception specification of overriding function is more lax than base version}}
    //   cxx98-14-note@#cwg126-z {{overridden virtual function is here}}
  };
#endif // __cplusplus <= 201402L
  void f() throw(int);
  // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
  //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
} // namespace cwg126

namespace cwg127 { // cwg127: 2.9
  __extension__ typedef __decltype(sizeof(0)) size_t;
  template<typename T> struct A {
    A() { throw 0; }
    void *operator new(size_t, const char * = 0);
    void operator delete(void *, const char *) { T::error; } // #cwg127-delete-const-char
    // expected-error@#cwg127-delete-const-char {{type 'void' cannot be used prior to '::' because it has no members}}
    //   expected-note@#cwg127-p {{in instantiation of member function 'cwg127::A<void>::operator delete' requested here}}
    // expected-error@#cwg127-delete-const-char {{type 'int' cannot be used prior to '::' because it has no members}}
    //   expected-note@#cwg127-q {{in instantiation of member function 'cwg127::A<int>::operator delete' requested here}}
    void operator delete(void *) { T::error; }
  };
  A<void> *p = new A<void>; // #cwg127-p
  A<int> *q = new ("") A<int>; // #cwg127-q
} // namespace cwg127

namespace cwg128 { // cwg128: 2.7
  enum E1 { e1 } x = e1;
  enum E2 { e2 } y = static_cast<E2>(x), z = static_cast<E2>(e1);
} // namespace cwg128

// cwg129: dup 616
// cwg130: na

namespace cwg131 { // cwg131: sup P1949
  const char *a_with_\u0e8c = "\u0e8c";
  const char *b_with_\u0e8d = "\u0e8d";
  const char *c_with_\u0e8e = "\u0e8e";
} // namespace cwg131

namespace cwg132 { // cwg132: no
  void f() {
    extern struct {} x; // ok
    extern struct S {} y; // FIXME: This is invalid.
  }
  static enum { E } e;
} // namespace cwg132

// cwg133: dup 87
// cwg134: na

namespace cwg135 { // cwg135: 2.7
  struct A {
    A f(A a) { return a; }
    friend A g(A a) { return a; }
    static A h(A a) { return a; }
  };
} // namespace cwg135

namespace cwg136 { // cwg136: 3.4
  void f(int, int, int = 0); // #cwg136-f
  void g(int, int, int); // #cwg136-g
  struct A {
    friend void f(int, int = 0, int);
    // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
    //   expected-note@#cwg136-f {{previous declaration is here}}
    friend void g(int, int, int = 0);
    // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
    //   expected-note@#cwg136-g {{previous declaration is here}}
    friend void h(int, int, int = 0);
    // expected-error@-1 {{friend declaration specifying a default argument must be a definition}}
    friend void i(int, int, int = 0) {} // #cwg136-A-i
    friend void j(int, int, int = 0) {}
    operator int();
  };
  void i(int, int, int);
  // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
  //   expected-note@#cwg136-A-i {{previous declaration is here}}
  void q() {
    j(A(), A()); // ok, has default argument
  }
  extern "C" void k(int, int, int, int); // #cwg136-k
  namespace NSA {
  struct A {
    friend void cwg136::k(int, int, int, int = 0);
    // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
    //   expected-note@#cwg136-k {{previous declaration is here}}
  };
  }
  namespace NSB {
  struct A {
    friend void cwg136::k(int, int, int = 0, int); // #cwg136-friend-k
    // expected-error@#cwg136-friend-k {{friend declaration specifying a default argument must be the only declaration}}
    //   expected-note@#cwg136-k {{previous declaration is here}}
    // expected-error@#cwg136-friend-k {{missing default argument on parameter}}
  };
  }
  struct B {
    void f(int); // #cwg136-B-f
  };
  struct C {
    friend void B::f(int = 0);
    // expected-error@-1 {{friend declaration specifying a default argument must be the only declaration}}
    //   expected-note@#cwg136-B-f {{previous declaration is here}}
  };
} // namespace cwg136

namespace cwg137 { // cwg137: 2.7
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
} // namespace cwg137

namespace cwg138 { // cwg138: partial
namespace example1 {
void foo(); // #cwg138-ex1-foo
namespace A {
  using example1::foo; // #cwg138-ex1-using
  class X {
    static const int i = 10;
    // This friend declaration is using neither qualified-id nor template-id,
    // so name 'foo' is not looked up, which means the using-declaration has no effect.
    // Target scope of this declaration is A, so this is grating friendship to
    // (hypothetical) A::foo instead of 'example1::foo' using declaration refers to.
    // A::foo corresponds to example1::foo named by the using declaration,
    // and since A::foo is a different entity, they potentially conflict.
    // FIXME: This is ill-formed, but not for the reason diagnostic says.
    friend void foo();
    // expected-error@-1 {{cannot befriend target of using declaration}}
    //   expected-note@#cwg138-ex1-foo {{target of using declaration}}
    //   expected-note@#cwg138-ex1-using {{using declaration}}
  };
}
} // namespace example1

namespace example2 {
void f();
void g();
class B {
  void g();
};
class A : public B {
  static const int i = 10;
  void f();
  // Both friend declaration are not using qualified-ids or template-ids,
  // so 'f' and 'g' are not looked up, which means that presence of A::f
  // and base B have no effect.
  // Both target scope of namespace 'example2', and grant friendship to
  // example2::f and example2::g respectively.
  friend void f();
  friend void g();
};
void f() {
  int i2 = A::i;
}
void g() {
  int i3 = A::i;
}
} // namespace example2

namespace example3 {
struct Base {
private:
  static const int i = 10; // #cwg138-ex3-Base-i
  
public:
  struct Data;
  // Elaborated type specifier is not the sole constituent of declaration,
  // so 'Data' undergoes unqualified type-only lookup, which finds Base::Data.
  friend class Data;

  struct Data {
    void f() {
      int i2 = Base::i;
    }
  };
};
struct Data {
  void f() {  
    int i2 = Base::i;
    // expected-error@-1 {{'i' is a private member of 'cwg138::example3::Base'}}
    //   expected-note@#cwg138-ex3-Base-i {{declared private here}}
  }
};
} // namespace example3
} // namespace cwg138

namespace cwg139 { // cwg139: 2.7
  namespace example1 {
    typedef int f; // #cwg139-typedef-f
    struct A {
      friend void f(A &);
      // expected-error@-1 {{redefinition of 'f' as different kind of symbol}}
      //   expected-note@#cwg139-typedef-f {{previous definition is here}}
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
} // namespace cwg139

namespace cwg140 { // cwg140: 2.7
  void f(int *const) {} // #cwg140-f-first
  void f(int[3]) {}
  // expected-error@-1 {{redefinition of 'f'}}
  //   expected-note@#cwg140-f-first {{previous definition is here}}
  void g(const int);
  void g(int n) { n = 2; }
} // namespace cwg140

namespace cwg141 { // cwg141: 3.1
  template<typename T> void f();
  template<typename T> struct S { int n; }; // #cwg141-S
  struct A : S<int> {
    template<typename T> void f();
    template<typename T> struct S {}; // #cwg141-A-S
  } a;
  struct B : S<int> {} b;
  void g() {
    a.f<int>();
    (void)a.S<int>::n; // #cwg141-a
    // cxx98-error@#cwg141-a {{lookup of 'S' in member access expression is ambiguous; using member of 'struct A'}}
    //   cxx98-note@#cwg141-A-S {{lookup in the object type 'struct A' refers here}}
    //   cxx98-note@#cwg141-S {{lookup from the current scope refers here}}
    // expected-error@#cwg141-a {{no member named 'n' in 'cwg141::A::S<int>'; did you mean '::cwg141::S<int>::n'?}}
    //   expected-note@#cwg141-S {{'::cwg141::S<int>::n' declared here}}
    // FIXME: we issue a useful diagnostic first, then some bogus ones.
    b.f<int>();
    // expected-error@-1 {{no member named 'f' in 'cwg141::B'}}
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
} // namespace cwg141

namespace cwg142 { // cwg142: 2.8
  class B { // #cwg142-B
  public:
    int mi; // #cwg142-B-mi
    static int si; // #cwg142-B-si
  };
  class D : private B { // #cwg142-D
  };
  class DD : public D {
    void f();
  };
  void DD::f() {
    mi = 3;
    // expected-error@-1 {{'mi' is a private member of 'cwg142::B'}}
    //   expected-note@#cwg142-D {{constrained by private inheritance here}}
    //   expected-note@#cwg142-B-mi {{member is declared here}}
    si = 3;
    // expected-error@-1 {{'si' is a private member of 'cwg142::B'}}
    //   expected-note@#cwg142-D {{constrained by private inheritance here}}
    //   expected-note@#cwg142-B-si {{member is declared here}}
    B b_old;
    // expected-error@-1 {{'B' is a private member of 'cwg142::B'}}
    //   expected-note@#cwg142-D {{constrained by private inheritance here}}
    //   expected-note@#cwg142-B {{member is declared here}}
    cwg142::B b;
    b.mi = 3;
    b.si = 3;
    B::si = 3;
    // expected-error@-1 {{'B' is a private member of 'cwg142::B'}}
    //   expected-note@#cwg142-D {{constrained by private inheritance here}}
    //   expected-note@#cwg142-B {{member is declared here}}
    cwg142::B::si = 3;
    B *bp1_old = this; // #cwg142-bp1_old
    // expected-error@#cwg142-bp1_old {{'B' is a private member of 'cwg142::B'}}
    //   expected-note@#cwg142-D {{constrained by private inheritance here}}
    //   expected-note@#cwg142-B {{member is declared here}}
    // expected-error@#cwg142-bp1_old {{cannot cast 'cwg142::DD' to its private base class 'B'}}
    //   expected-note@#cwg142-D {{declared private here}}
    cwg142::B *bp1 = this;
    // expected-error@-1 {{cannot cast 'cwg142::DD' to its private base class 'cwg142::B'}}
    //   expected-note@#cwg142-D {{declared private here}}
    B *bp2_old = (B*)this; // #cwg142-bp2_old
    // expected-error@#cwg142-bp2_old {{'B' is a private member of 'cwg142::B'}}
    //   expected-note@#cwg142-D {{constrained by private inheritance here}}
    //   expected-note@#cwg142-B {{member is declared here}}
    // expected-error@#cwg142-bp2_old {{'B' is a private member of 'cwg142::B'}}
    //   expected-note@#cwg142-D {{constrained by private inheritance here}}
    //   expected-note@#cwg142-B {{member is declared here}}
    cwg142::B *bp2 = (cwg142::B*)this;
    bp2->mi = 3;
  }
} // namespace cwg142

namespace cwg143 { // cwg143: 2.7
  namespace A { struct X; }
  namespace B { void f(A::X); }
  namespace A {
    struct X { friend void B::f(X); };
  }
  void g(A::X x) {
    f(x);
    // expected-error@-1 {{use of undeclared identifier 'f'}}
  }
} // namespace cwg143

namespace cwg145 { // cwg145: 2.7
  void f(bool b) {
    ++b;
    // cxx98-14-warning@-1 {{incrementing expression of type bool is deprecated and incompatible with C++17}}
    // since-cxx17-error@-2 {{ISO C++17 does not allow incrementing expression of type bool}}
    b++;
    // cxx98-14-warning@-1 {{incrementing expression of type bool is deprecated and incompatible with C++17}}
    // since-cxx17-error@-2 {{ISO C++17 does not allow incrementing expression of type bool}}
  }
} // namespace cwg145

namespace cwg147 { // cwg147: 2.7
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
} // namespace cwg147

namespace cwg148 { // cwg148: 2.7
  struct A { int A::*p; };
  static_assert(__is_pod(int(A::*)), "");
  static_assert(__is_pod(A), "");
} // namespace cwg148

// cwg149: na

namespace cwg150 { // cwg150: 19
  namespace p1 {
    template <class T, class U = int>
    class ARG { };

    template <class X, template <class Y> class PARM>
    void f(PARM<X>) { }

    void g() {
      ARG<int> x;
      f(x);
    }
  } // namespace p1

  namespace p2 {
    template <template <class T, class U = int> class PARM>
    class C {
      PARM<int> pi;
    };
  } // namespace p2

  namespace n1 {
    struct Dense { static const unsigned int dim = 1; };

    template <template <typename> class View,
              typename Block>
    void operator+(float, View<Block> const&);

    template <typename Block,
              unsigned int Dim = Block::dim>
    class Lvalue_proxy { operator float() const; };

    void test_1d (void) {
      Lvalue_proxy<Dense> p;
      float b;
      b + p;
    }
  } // namespace n1
} // namespace cwg150

namespace cwg151 { // cwg151: 3.1
  struct X {};
  typedef int X::*p;
  static_assert(__enable_constant_folding(p() == 0), "");
} // namespace cwg151

namespace cwg152 { // cwg152: 2.7
  struct A {
    A(); // #cwg152-A-ctor
    explicit A(const A&); // #cwg152-A-explicit-ctor
  };
  A a1 = A();
  // cxx98-14-error@-1 {{no matching constructor for initialization of 'A'}}
  //   cxx98-14-note@#cwg152-A-explicit-ctor {{explicit constructor is not a candidate}}
  //   cxx98-14-note@#cwg152-A-ctor {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  A a2((A()));

  A &f();
  A a3 = f();
  // expected-error@-1 {{no matching constructor for initialization of 'A'}}
  //   expected-note@#cwg152-A-explicit-ctor {{explicit constructor is not a candidate}}
  //   expected-note@#cwg152-A-ctor {{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
  A a4(f());
} // namespace cwg152

// cwg153: na

namespace cwg154 { // cwg154: 2.7
  union { int a; };
  // expected-error@-1 {{nonymous unions at namespace or global scope must be declared 'static'}}
  namespace {
    union { int b; };
  }
  static union { int c; };
} // namespace cwg154

namespace cwg155 { // cwg155: dup 632
  struct S { int n; } s = { { 1 } };
  // expected-warning@-1 {{braces around scalar initializer}}
} // namespace cwg155

// cwg156: sup 1111
// cwg158 is in cwg158.cpp

namespace cwg159 { // cwg159: 3.5
  namespace X { void f(); }
  void f();
  void cwg159::f() {}
  // expected-warning@-1 {{extra qualification on member 'f'}}
  void cwg159::X::f() {}
} // namespace cwg159

// cwg160: na

namespace cwg161 { // cwg161: 3.1
  class A {
  protected:
    struct B { int n; } b; // #cwg161-B
    static B bs;
    void f(); // #cwg161-f
    static void sf();
  };
  struct C : A {};
  struct D : A {
    void g(C c) {
      (void)b.n;
      B b1;
      C::B b2; // ok, accessible as a member of A
      (void)&C::b;
      // expected-error@-1 {{'b' is a protected member of 'cwg161::A'}}
      //   expected-note@#cwg161-B {{declared protected here}}
      (void)&C::bs;
      (void)c.b;
      // expected-error@-1 {{'b' is a protected member of 'cwg161::A'}}
      //   expected-note@#cwg161-B {{declared protected here}}
      (void)c.bs;
      f();
      sf();
      c.f();
      // expected-error@-1 {{protected}}
      //   expected-note@#cwg161-f {{declared protected here}}
      c.sf();
      A::f();
      D::f();
      A::sf();
      C::sf();
      D::sf();
    }
  };
} // namespace cwg161

namespace cwg162 { // cwg162: 19
  struct A {
    char &f(char);
    static int &f(int);

    void g() {
      int &a = (&A::f)(0);
      char &b = (&A::f)('0');
      // expected-error@-1 {{non-const lvalue reference to type 'char' cannot bind to a value of unrelated type 'int'}}
    }
  };

  int &c = (&A::f)(0);
  char &d = (&A::f)('0');
  // expected-error@-1 {{non-const lvalue reference to type 'char' cannot bind to a value of unrelated type 'int'}}
} // namespace cwg162

// cwg163: na

namespace cwg164 { // cwg164: 2.7
  void f(int);
  template <class T> int g(T t) { return f(t); }

  enum E { e };
  int f(E);

  int k = g(e);
} // namespace cwg164

namespace cwg165 { // cwg165: no
  namespace N {
    struct A { friend struct B; };
    void f() { void g(); }
  }
  // FIXME: cwg1477 says this is ok, cwg165 says it's ill-formed
  struct N::B {};
  // FIXME: cwg165 says this is ill-formed, but the argument in cwg1477 says it's ok
  void N::g() {}
} // namespace cwg165

namespace cwg166 { // cwg166: 2.9
  namespace A { class X; }

  template<typename T> int f(T t) { return t.n; }
  int g(A::X);
  template<typename T> int h(T t) { return t.n; }
  // expected-error@-1 {{'n' is a private member of 'cwg166::A::X'}}
  //   expected-note@#cwg166-h-instantiation {{in instantiation of function template specialization 'cwg166::h<cwg166::A::X>' requested here}}
  //   expected-note@#cwg166-X-n {{implicitly declared private here}}
  int i(A::X);

  namespace A {
    class X {
      friend int f<X>(X);
      friend int cwg166::g(X);
      friend int h(X);
      friend int i(X);
      int n; // #cwg166-X-n
    };

    int h(X x) { return x.n; }
    int i(X x) { return x.n; }
  }

  template int f(A::X);
  int g(A::X x) { return x.n; }
  template int h(A::X); // #cwg166-h-instantiation
  int i(A::X x) { return x.n; }
  // expected-error@-1 {{'n' is a private member of 'cwg166::A::X'}}
  //   expected-note@#cwg166-X-n {{implicitly declared private here}}
} // namespace cwg166

// cwg167: sup 1012

namespace cwg168 { // cwg168: no
  extern "C" typedef int (*p)();
  extern "C++" typedef int (*q)();
  struct S {
    static int f();
  };
  p a = &S::f; // FIXME: this should fail.
  q b = &S::f;
} // namespace cwg168

namespace cwg169 { // cwg169: 3.4
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
} // namespace cwg169

namespace cwg170 { // cwg170: 3.1
#if __cplusplus >= 201103L
struct A {};
struct B : A { int i; };
struct C : A {};
struct D : C {};

constexpr int f(int A::*) { return 0; }
constexpr int g(int C::*) { return 0; }
constexpr int h(int D::*) { return 0; }

constexpr auto p = static_cast<int A::*>(&B::i);
constexpr auto q = f(p);
constexpr auto r = g(p);
// since-cxx11-error@-1 {{constexpr variable 'r' must be initialized by a constant expression}}
constexpr auto s = h(p);
// since-cxx11-error@-1 {{constexpr variable 's' must be initialized by a constant expression}}
#endif
} // namespace cwg170

namespace { // cwg171: 3.4
  int cwg171a;
}
int cwg171b; // #cwg171b-int
namespace cwg171 {
  extern "C" void cwg171a();
  extern "C" void cwg171b();
  // expected-error@-1 {{declaration of 'cwg171b' with C language linkage conflicts with declaration in global scope}}
  //   expected-note@#cwg171b-int {{declared in global scope here}}
} // namespace cwg171

namespace cwg172 { // cwg172: 2.7
  enum { zero };
  static_assert(-1 < zero, "");

  enum { x = -1, y = (unsigned int)-1 };
  static_assert(sizeof(x) > sizeof(int), "");

  enum { a = (unsigned int)-1 / 2 };
  static_assert(sizeof(a) == sizeof(int), "");
  static_assert(-a < 0, "");

  enum { b = (unsigned int)-1 / 2 + 1 };
  static_assert(sizeof(b) == sizeof(unsigned int), "");
  static_assert(-b > 0, "");

  enum { c = (unsigned long)-1 / 2 };
  static_assert(sizeof(c) == sizeof(long), "");
  static_assert(-c < 0, "");

  enum { d = (unsigned long)-1 / 2 + 1 };
  static_assert(sizeof(d) == sizeof(unsigned long), "");
  static_assert(-d > 0, "");

  enum { e = (unsigned long long)-1 / 2 };
  // cxx98-error@-1 {{'long long' is a C++11 extension}}
  static_assert(sizeof(e) == sizeof(long), "");
  static_assert(-e < 0, "");

  enum { f = (unsigned long long)-1 / 2 + 1 };
  // cxx98-error@-1 {{'long long' is a C++11 extension}}
  static_assert(sizeof(f) == sizeof(unsigned long), "");
  static_assert(-f > 0, "");
} // namespace cwg172

namespace cwg173 { // cwg173: 2.7
  static_assert('0' + 1 == '1' && '0' + 2 == '2' && '0' + 3 == '3' &&
                '0' + 4 == '4' && '0' + 5 == '5' && '0' + 6 == '6' &&
                '0' + 7 == '7' && '0' + 8 == '8' && '0' + 9 == '9', "");
} // namespace cwg173

// cwg174: sup 1012

namespace cwg175 { // cwg175: 2.8
  struct A {}; // #cwg175-A
  struct B : private A {}; // #cwg175-B
  struct C : B {
    A a;
    // expected-error@-1 {{'A' is a private member of 'cwg175::A'}}
    //   expected-note@#cwg175-B {{constrained by private inheritance here}}
    //   expected-note@#cwg175-A {{member is declared here}}
    cwg175::A b;
  };
} // namespace cwg175

namespace cwg176 { // cwg176: 3.1
  template<typename T> class Y;
  template<> class Y<int> {
    void f() {
      typedef Y A; // #cwg176-A-first
      typedef Y<char> A;
      // expected-error@-1 {{typedef redefinition with different types ('Y<char>' vs 'Y<int>')}}
      //   expected-note@#cwg176-A-first {{previous definition is here}}
    }
  };

  template<typename T> struct Base {}; // #cwg176-Base
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
    //   expected-note@#cwg176-Base {{member type 'cwg176::Base<int>' found by ambiguous name lookup}}
    //   expected-note@#cwg176-Base {{member type 'cwg176::Base<char>' found by ambiguous name lookup}}
    typename Derived2::Base<double> d;
  };

  template<typename T> class X { // #cwg176-X
    X *p1;
    X<T> *p2;
    X<int> *p3;
    cwg176::X *p4; // #cwg176-p4
    // cxx98-14-error@#cwg176-p4 {{use of class template 'cwg176::X' requires template arguments}}
    //  cxx98-14-note@#cwg176-X {{template is declared here}}
    // since-cxx17-error@#cwg176-p4 {{use of class template 'cwg176::X' requires template arguments; argument deduction not allowed in non-static class member}}
    //  since-cxx17-note@#cwg176-X {{template is declared here}}
  };
} // namespace cwg176

namespace cwg177 { // cwg177: 2.7
  struct B {};
  struct A {
    A(A &); // #cwg177-A-copy-ctor
    A(const B &); // #cwg177-A-ctor-from-B
  };
  B b;
  A a = b;
  // cxx98-14-error@-1 {{no viable constructor copying variable of type 'A'}}
  //   cxx98-14-note@#cwg177-A-copy-ctor {{candidate constructor not viable: expects an lvalue for 1st argument}}
  //   cxx98-14-note@#cwg177-A-ctor-from-B {{candidate constructor not viable: no known conversion from 'A' to 'const B &' for 1st argument}}

  struct C { C(C&); }; // #cwg177-C-copy-ctor
  struct D : C {};
  struct E { operator D(); };
  E e;
  C c = e;
  // expected-error@-1 {{no viable constructor copying variable of type 'D'}}
  //   expected-note@#cwg177-C-copy-ctor {{candidate constructor not viable: expects an lvalue for 1st argument}}
} // namespace cwg177

namespace cwg178 { // cwg178: 3.1
  static_assert(int() == 0, "");
#if __cplusplus >= 201103L
  static_assert(int{} == 0, "");
  struct S { int a, b; };
  static_assert(S{1}.b == 0, "");
  struct T { constexpr T() : n() {} int n; };
  static_assert(T().n == 0, "");
  struct U : S { constexpr U() : S() {} };
  static_assert(U().b == 0, "");
#endif
} // namespace cwg178

namespace cwg179 { // cwg179: 2.7
  void f();
  int n = &f - &f;
  // expected-error@-1 {{arithmetic on pointers to the function type 'void ()'}}
} // namespace cwg179

namespace cwg180 { // cwg180: 2.8
  template<typename T> struct X : T, T::some_base {
    X() : T::some_type_that_might_be_T(), T::some_base() {}
    friend class T::some_class;
    void f() {
      enum T::some_enum e;
    }
  };
} // namespace cwg180

namespace cwg181 { // cwg181: 2.7
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
} // namespace cwg181

namespace cwg182 { // cwg182: 14
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
} // namespace cwg182

namespace cwg183 { // cwg183: sup 382
  template<typename T> struct A {};
  template<typename T> struct B {
    typedef int X;
  };
  template<> struct A<int> {
    typename B<int>::X x;
    // cxx98-error@-1 {{'typename' occurs outside of a template}}
  };
} // namespace cwg183

namespace cwg184 { // cwg184: 2.7
  template<typename T = float> struct B {};

  template<template<typename TT = float> class T> struct A {
    void f();
    void g();
  };

  template<template<typename TT> class T> void A<T>::f() { // #cwg184-T
    T<> t;
    // expected-error@-1 {{too few template arguments for template template parameter 'T'}}
    //   expected-note@#cwg184-T {{template is declared here}}
  }

  template<template<typename TT = char> class T> void A<T>::g() {
    T<> t;
    typedef T<> X;
    typedef T<char> X;
  }

  void h() { A<B>().g(); }
} // namespace cwg184

// cwg185 is in cwg185.cpp

namespace cwg187 { // cwg187: sup 481
  const int Z = 1;
  template<int X = Z, int Z = X> struct A;
  typedef A<> T;
  typedef A<1, 1> T;
} // namespace cwg187

namespace cwg188 { // cwg188: 2.7
  char c[10];
  static_assert(sizeof(0, c) == 10, "");
} // namespace cwg188

namespace cwg190 { // cwg190: 19
struct A {
  int a;
  static double x;
  int b;
  void y();
  int c;
};

struct B {
  int a;
  void y();
  int b;
  static double x;
  int c;
};

static_assert(__is_layout_compatible(A, B), "");
} // namespace cwg190

int cwg191_j;
namespace cwg191 { // cwg191: 2.7
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
            cwg191_j = 5;
          }
        };
      }
    };
  }
} // namespace cwg191

namespace cwg192 { // cwg192: 2.7
struct S {
  void f(I i) { }
  // expected-error@-1 {{unknown type name 'I'}}
  typedef int I;
};
} // namespace cwg192

// cwg193 is in cwg193.cpp

namespace cwg194 { // cwg194: 2.7
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
} // namespace cwg194

namespace cwg195 { // cwg195: 2.7
  void f();
  int *p = (int*)&f;
  // cxx98-error@-1 {{cast between pointer-to-function and pointer-to-object is an extension}}
  void (*q)() = (void(*)())&p;
  // cxx98-error@-1 {{cast between pointer-to-function and pointer-to-object is an extension}}
} // namespace cwg195

namespace cwg197 { // cwg197: 2.7
  char &f(char);

  template <class T> void g(T t) {
    char &a = f(1);
    char &b = f(T(1));
    // expected-error@-1 {{non-const lvalue reference to type 'char' cannot bind to a value of unrelated type 'int'}}
    //   expected-note@#cwg197-g-e-call {{in instantiation of function template specialization 'cwg197::g<cwg197::E>' requested here}}
    char &c = f(t);
    // expected-error@-1 {{non-const lvalue reference to type 'char' cannot bind to a value of unrelated type 'int'}}
  }

  void f(int);

  enum E { e };
  int &f(E);

  void h() {
    g('a');
    g(2);
    g(e); // #cwg197-g-e-call
  }
} // namespace cwg197

namespace cwg198 { // cwg198: 2.9
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
} // namespace cwg198

// cwg199 is in cwg199.cpp
