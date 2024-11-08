// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98-23,cxx98-11,cxx98-14,cxx98-17,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx98-23,cxx98-11,cxx98-14,cxx98-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,cxx98-23,cxx98-14,cxx98-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,cxx98-23,since-cxx17,cxx98-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,cxx98-23,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,cxx98-23,since-cxx23,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx26,since-cxx23,since-cxx20,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

// FIXME: This is included to avoid a diagnostic with no source location
// pointing at the implicit operator new. We can't match such a diagnostic
// with -verify.
__extension__ typedef __SIZE_TYPE__ size_t;
void *operator new(size_t); // #cwg5xx-global-operator-new
// cxx98-error@-1 {{'operator new' is missing exception specification 'throw(std::bad_alloc)'}}
#if __cplusplus > 201402L
namespace std {
  enum class align_val_t : size_t {};
}
void *operator new(size_t, std::align_val_t); // #cwg5xx-global-operator-new-aligned
#endif

namespace std {
  struct type_info;
}

namespace cwg500 { // cwg500: dup 372
  class D;
  class A {
    class B;
    class C;
    friend class D;
  };
  class A::B {};
  class A::C : public A::B {};
  class D : public A::B {};
}

namespace cwg501 { // cwg501: yes
  struct A {
    friend void f() {}
    void g() {
      void (*p)() = &f;
      // expected-error@-1 {{use of undeclared identifier 'f'}}
    }
  };
}

namespace cwg502 { // cwg502: yes
  struct Q {};
  template<typename T> struct A {
    enum E { e = 1 };
    void q1() { f(e); }
    void q2() { Q arr[sizeof(E)]; f(arr); }
    void q3() { Q arr[e]; f(arr); }
    void sanity() { Q arr[1]; f(arr); }
    // expected-error@-1 {{use of undeclared identifier 'f'}}
  };
  int f(A<int>::E);
  template<int N> int f(Q (&)[N]);
  template struct A<int>;
}

namespace cwg505 { // cwg505: yes
  const char *exts = "\e\(\{\[\%";
  // expected-error@-1 {{use of non-standard escape character '\e'}}
  // expected-error@-2 {{use of non-standard escape character '\('}}
  // expected-error@-3 {{use of non-standard escape character '\{'}}
  // expected-error@-4 {{use of non-standard escape character '\['}}
  // expected-error@-5 {{use of non-standard escape character '\%'}}
  const char *unknown = "\Q";
  // expected-error@-1 {{unknown escape sequence '\Q'}}
}

namespace cwg506 { // cwg506: yes
  struct NonPod { ~NonPod(); };
  void f(...);
  void g(NonPod np) { f(np); }
  // cxx98-error@-1 {{cannot pass object of non-POD type 'NonPod' through variadic function; call will abort at runtime}}
  // since-cxx11-error@-2 {{cannot pass object of non-trivial type 'NonPod' through variadic function; call will abort at runtime}}
}

// FIXME: Add tests here once CWG260 is resolved.
// cwg507: dup 260

// cwg508: na
// cwg509: na
// cwg510: na

namespace cwg512 { // cwg512: yes
  struct A { // #cwg512-A
    A(int); // #cwg512-A-ctor
  };
  union U { A a; };
  // cxx98-error@-1 {{union member 'a' has a non-trivial default constructor}}
  //   cxx98-note@#cwg512-A {{because type 'cwg512::A' has no default constructor}}
  //   cxx98-note@#cwg512-A-ctor {{implicit default constructor suppressed by user-declared constructor}}
}

// cwg513: na

namespace cwg514 { // cwg514: yes
  namespace A { extern int x, y; }
  int A::x = y;
}

namespace cwg515 { // cwg515: sup 1017
  // FIXME: cwg1017 reverses the wording of cwg515, but the current draft has
  // cwg515's wording, with a different fix for cwg1017.

  struct X { int n; };
  template<typename T> struct Y : T {
    int f() { return X::n; }
  };
  int k = Y<X>().f();

  struct A { int a; };
  struct B { void f() { int k = sizeof(A::a); } };
  // cxx98-error@-1 {{invalid use of non-static data member 'a'}}
}

// cwg516: na

namespace cwg517 { // cwg517: no
  // This is NDR, but we should diagnose it anyway.
  template<typename T> struct S {};
  template<typename T> int v = 0;
  // cxx98-11-error@-1 {{variable templates are a C++14 extension}}

  template struct S<int*>;
  template int v<int*>;

  S<char&> s;
  int k = v<char&>;

  // FIXME: These are both ill-formed.
  template<typename T> struct S<T*> {};
  template<typename T> int v<T*> = 0;

  // FIXME: These are both ill-formed.
  template<typename T> struct S<T&> {};
  template<typename T> int v<T&> = 0;
}

namespace cwg518 { // cwg518: yes c++11
  enum E { e, };
  // cxx98-error@-1 {{commas at the end of enumerator lists are a C++11 extension}}
}

// cwg519 is in cwg519.cpp
// cwg520: na

// cwg521: no
// FIXME: The wording here is broken. It's not reasonable to expect a
// diagnostic here. Once the relevant DR gets a number, mark this as a dup.

namespace cwg522 { // cwg522: yes
  struct S {};
  template<typename T> void b1(volatile T &);
  template<typename T> void b2(volatile T * const *);
  template<typename T> void b2(volatile T * const S::*);
  template<typename T> void b2(volatile T * const S::* const *);
  template<typename T> void b2a(volatile T *S::* const *); // #cwg522-b2a

  template<typename T> struct Base {};
  struct Derived : Base<int> {};
  template<typename T> void b3(Base<T>);
  template<typename T> void b3(Base<T> *);

  void test(int n, const int cn, int **p, int *S::*pm) {
    int *a[3], *S::*am[3];
    const Derived cd = Derived();
    Derived d[3];

    b1(n);
    b1(cn);
    b2(p);
    b2(pm);
    b2(a);
    b2(am);
    b2a(am);
    // expected-error@-1 {{no matching function for call to 'b2a'}}
    //   expected-note@#cwg522-b2a {{candidate template ignored: deduced type 'volatile int *cwg522::S::*const *' of 1st parameter does not match adjusted type 'int *cwg522::S::**' of argument}}
    b3(d);
    b3(cd);
  }
}

namespace cwg524 { // cwg524: yes
  template<typename T> void f(T a, T b) { operator+(a, b); }
  // expected-error@-1 {{call to function 'operator+' that is neither visible in the template definition nor found by argument-dependent lookup}}
  //   expected-note@#cwg524-f-N-S {{in instantiation of function template specialization 'cwg524::f<cwg524::N::S>' requested here}}
  //   expected-note@#cwg524-operator-plus {{'operator+' should be declared prior to the call site or in namespace 'cwg524::N'}}

  struct S {};
  void operator+(S, S);
  template void f(S, S);

  namespace N { struct S {}; }
  void operator+(N::S, N::S); // #cwg524-operator-plus
  template void f(N::S, N::S); // #cwg524-f-N-S
}

namespace cwg525 { // cwg525: yes
  namespace before {
    // Note, the example was correct prior to the change; instantiation is
    // required for cases like this:
    template <class T> struct D { operator T*(); };
    void g(D<double> ppp) {
      delete ppp;
    }
  }
  namespace after {
    template <class T> struct D { typename T::error e; };
    // expected-error@-1 {{type 'double' cannot be used prior to '::' because it has no members}}
    //   expected-note@#cwg525-ppp {{in instantiation of template class 'cwg525::after::D<double>' requested here}}
    void g(D<double> *ppp) {
      delete ppp; // #cwg525-ppp
    }
  }
}

namespace cwg526 { // cwg526: yes
  template<int> struct S {};
  template<int N> void f1(S<N> s);
  template<int N> void f2(S<(N)> s); // #cwg526-f2
  template<int N> void f3(S<+N> s); // #cwg526-f3
  template<int N> void g1(int (&)[N]);
  template<int N> void g2(int (&)[(N)]); // #cwg526-g2
  template<int N> void g3(int (&)[+N]); // #cwg526-g3

  void test(int (&a)[3], S<3> s) {
    f1(s);
    f2(s);
    // expected-error@-1 {{no matching function for call to 'f2'}}
    //   expected-note@#cwg526-f2 {{candidate template ignored: couldn't infer template argument 'N'}}
    f3(s);
    // expected-error@-1 {{no matching function for call to 'f3'}}
    //   expected-note@#cwg526-f3 {{candidate template ignored: couldn't infer template argument 'N'}}
    g1(a);
    g2(a);
    // expected-error@-1 {{no matching function for call to 'g2'}}
    //   expected-note@#cwg526-g2 {{candidate template ignored: couldn't infer template argument 'N'}}
    g3(a);
    // expected-error@-1 {{no matching function for call to 'g3'}}
    //   expected-note@#cwg526-g3 {{candidate template ignored: couldn't infer template argument 'N'}}
  }

  template<int N> struct X {
    typedef int type;
    X<N>::type v1;
    X<(N)>::type v2;
    // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name X<(N)>::type; implicit 'typename' is a C++20 extension}}
    X<+N>::type v3;
    // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name X<+N>::type; implicit 'typename' is a C++20 extension}}
  };
}

namespace cwg527 { // cwg527: na
  // This DR is meaningless. It removes a required diagnostic from the case
  // where a not-externally-visible object is odr-used but not defined, which
  // requires a diagnostic for a different reason.
  extern struct { int x; } a; // FIXME: We should reject this, per cwg389.
  static struct { int x; } b;
  extern "C" struct { int x; } c;
  namespace { extern struct { int x; } d; }
  typedef struct { int x; } *P;
  struct E { static P e; }; // FIXME: We should reject this, per cwg389.
  namespace { struct F { static P f; }; }

  int ax = a.x, bx = b.x, cx = c.x, dx = d.x, ex = E::e->x, fx = F::f->x;
}

namespace cwg528 { // cwg528: 2.7

struct S; // #cwg528-S

void f() {
  typeid(S);
  // expected-error@-1 {{'typeid' of incomplete type 'S'}}
  //   expected-note@#cwg528-S {{forward declaration of 'cwg528::S'}}
}

} // namespace cwg528

namespace cwg530 { // cwg530: yes
  template<int*> struct S { enum { N = 1 }; };
  template<void(*)()> struct T { enum { N = 1 }; };
  int n;
  void f();
  int a[S<&n>::N];
  int b[T<&f>::N];
}

namespace cwg531 { // cwg531: partial
  namespace good {
    template<typename T> struct A {
      void f(T) { T::error; }
      template<typename U> void g(T, U) { T::error; }
      struct B { typename T::error error; };
      template<typename U> struct C { typename T::error error; };
      static T n;
    };
    template<typename T> T A<T>::n = T::error;

    template<> void A<int>::f(int) {}
    template<> template<typename U> void A<int>::g(int, U) {}
    template<> struct A<int>::B {};
    template<> template<typename U> struct A<int>::C {};
    template<> int A<int>::n = 0;

    void use(A<int> a) {
      a.f(a.n);
      a.g(0, 0);
      A<int>::B b;
      A<int>::C<int> c;
    }

    template<> struct A<char> {
      void f(char);
      template<typename U> void g(char, U);
      struct B;
      template<typename U> struct C;
      static char n;
    };

    void A<char>::f(char) {}
    template<typename U> void A<char>::g(char, U) {}
    struct A<char>::B {};
    template<typename U> struct A<char>::C {};
    char A<char>::n = 0;
  }

  namespace bad {
    template<typename T> struct A {
      void f(T) { T::error; }
      template<typename U> void g(T, U) { T::error; }
      struct B { typename T::error error; };
      template<typename U> struct C { typename T::error error; }; // #cwg531-C
      static T n;
    };
    template<typename T> T A<T>::n = T::error;

    void A<int>::f(int) {}
    // expected-error@-1 {{template specialization requires 'template<>'}}
    template<typename U> void A<int>::g(int, U) {}
    // expected-error@-1 {{template parameter list matching the non-templated nested type 'cwg531::bad::A<int>' should be empty}}
    struct A<int>::B {};
    // expected-error@-1 {{template specialization requires 'template<>'}}
    template<typename U> struct A<int>::C {};
    // expected-error@-1 {{template parameter list matching the non-templated nested type 'cwg531::bad::A<int>' should be empty}}
    // expected-error@-2 {{redefinition of 'C' as different kind of symbol}}
    //   expected-note@#cwg531-C {{previous definition is here}}
    int A<int>::n = 0;
    // expected-error@-1 {{template specialization requires 'template<>'}}

    template<> struct A<char> { // #cwg531-A-char
      void f(char);
      template<typename U> void g(char, U);
      struct B; // #cwg531-B
      template<typename U> struct C;
      static char n;
    };

    template<> void A<char>::f(char) {}
    // expected-error@-1 {{no function template matches function template specialization 'f'}}
    template<> template<typename U> void A<char>::g(char, U) {}
    // expected-error@-1 {{extraneous template parameter list in template specialization}}
    //   expected-note@#cwg531-A-char {{'template<>' header not required for explicitly-specialized class 'cwg531::bad::A<char>' declared here}}
    template<> struct A<char>::B {};
    // expected-error@-1 {{extraneous 'template<>' in declaration of struct 'B'}}
    // expected-error@-2 {{specialization of member 'cwg531::bad::A<char>::B' does not specialize an instantiated member}}
    //  expected-note@#cwg531-B {{attempt to specialize declaration here}}
    template<> template<typename U> struct A<char>::C {};
    // expected-error@-1 {{extraneous template parameter list in template specialization}}
    //   expected-note@#cwg531-A-char {{'template<>' header not required for explicitly-specialized class 'cwg531::bad::A<char>' declared here}}
    template<> char A<char>::n = 0;
    // expected-error@-1 {{extraneous 'template<>' in declaration of variable 'n'}}
  }

  namespace nested {
    template<typename T> struct A {
      template<typename U> struct B;
    };
    template<> template<typename U> struct A<int>::B {
      void f();
      void g();
      template<typename V> void h();
      template<typename V> void i();
    };
    template<> template<typename U> void A<int>::B<U>::f() {}
    template<typename U> void A<int>::B<U>::g() {}
    // expected-error@-1 {{template parameter list matching the non-templated nested type 'cwg531::nested::A<int>' should be empty ('template<>')}}

    template<> template<typename U> template<typename V> void A<int>::B<U>::h() {}
    template<typename U> template<typename V> void A<int>::B<U>::i() {}
    // expected-error@-1 {{template parameter list matching the non-templated nested type 'cwg531::nested::A<int>' should be empty ('template<>')}}

#if __cplusplus <= 201703L
    // FIXME: All of those declarations shouldn't crash in C++20 mode.
    template<> template<> void A<int>::B<int>::f() {}
    template<> template<> template<typename V> void A<int>::B<int>::h() {}
    template<> template<> template<> void A<int>::B<int>::h<int>() {}

    template<> void A<int>::B<char>::f() {}
    // cxx98-17-error@-1 {{template specialization requires 'template<>'}}
    template<> template<typename V> void A<int>::B<char>::h() {}
    // cxx98-17-error@-1 {{template parameter list matching the non-templated nested type 'cwg531::nested::A<int>::B<char>' should be empty ('template<>')}}
#endif
  }
}

// PR8130
namespace cwg532 { // cwg532: 3.5
  struct A { };

  template<class T> struct B {
    template<class R> int &operator*(R&);
  };

  template<class T, class R> float &operator*(T&, R&);
  void test() {
    A a;
    B<A> b;
    int &ir = b * a;
  }
}

// cwg533: na

namespace cwg534 { // cwg534: 2.9
  struct S {};
  template<typename T> void operator+(S, T);
  template<typename T> void operator+<T*>(S, T*) {}
  // expected-error@-1 {{function template partial specialization is not allowed}}
}

namespace cwg535 { // cwg535: yes
  class X { private: X(const X&); };
  struct A {
    X x;
    template<typename T> A(T&);
  };
  struct B : A {
    X y;
    B(volatile A&);
  };

  extern A a1;
  A a2(a1); // ok, uses constructor template

  extern volatile B b1;
  B b2(b1); // ok, uses converting constructor

  void f() { throw a1; }

#if __cplusplus >= 201103L
  struct C {
    constexpr C() : n(0) {}
    template<typename T> constexpr C(T&t) : n(t.n == 0 ? throw 0 : 0) {}
    int n;
  };
  constexpr C c() { return C(); }
  // ok, copy is elided
  constexpr C x = c();
#endif
}

// cwg536: na
// cwg537: na
// cwg538: na

// cwg539: yes
const cwg539(
// expected-error@-1 {{a type specifier is required for all declarations}}
    const a) {
    // expected-error@-1 {{unknown type name 'a'}}
  const b;
  // expected-error@-1 {{a type specifier is required for all declarations}}
  new const;
  // expected-error@-1 {{expected a type}}
  try {} catch (const n) {}
  // expected-error@-1 {{unknown type name 'n'}}
  try {} catch (const) {}
  // expected-error@-1 {{expected a type}}
  if (const n = 0) {}
  // expected-error@-1 {{a type specifier is required for all declarations}}
  switch (const n = 0) {}
  // expected-error@-1 {{a type specifier is required for all declarations}}
  while (const n = 0) {}
  // expected-error@-1 {{a type specifier is required for all declarations}}
  for (const n = 0;
  // expected-error@-1 {{a type specifier is required for all declarations}}
       const m = 0; ) {}
       // expected-error@-1 {{a type specifier is required for all declarations}}
  sizeof(const);
  // expected-error@-1 {{a type specifier is required for all declarations}}
  struct S {
    const n;
    // expected-error@-1 {{a type specifier is required for all declarations}}
    operator const();
    // expected-error@-1 {{expected a type}}
  };
#if __cplusplus >= 201103L
  int arr[3];
  // FIXME: The extra braces here are to avoid the parser getting too
  // badly confused when recovering here. We should fix this recovery.
  { for (const n
  // since-cxx11-error@-1 {{unknown type name 'n'}}
  //   since-cxx11-note@-2 {{}}
         : arr) ; {} }
         // since-cxx11-error@-1 +{{}}
  (void) [](const) {};
  // since-cxx11-error@-1 {{a type specifier is required for all declarations}}
  (void) [](const n) {};
  // since-cxx11-error@-1 {{unknown type name 'n'}}
  enum E : const {};
  // since-cxx11-error@-1 {{expected a type}}
  using T = const;
  // since-cxx11-error@-1 {{expected a type}}
  auto f() -> const;
  // since-cxx11-error@-1 {{expected a type}}
#endif
}

namespace cwg540 { // cwg540: yes
  typedef int &a;
  typedef const a &a;
  // expected-warning@-1 {{'const' qualifier on reference type 'a' (aka 'int &') has no effect}}
  typedef const int &b;
  typedef b &b;
  typedef const a &c; // #cwg540-typedef-a-c
  // expected-warning@-1 {{'const' qualifier on reference type 'a' (aka 'int &') has no effect}}
  typedef const b &c; // #cwg540-typedef-b-c
  // expected-error@#cwg540-typedef-b-c {{typedef redefinition with different types ('const int &' vs 'int &')}}
  //   expected-note@#cwg540-typedef-a-c {{previous definition is here}}
  // expected-warning@#cwg540-typedef-b-c {{'const' qualifier on reference type 'b' (aka 'const int &') has no effect}}
}

namespace cwg541 { // cwg541: yes
  template<int> struct X { typedef int type; };
  template<typename T> struct S {
    int f(T);

    int g(int);
    T g(bool);

    int h();
    int h(T);

    void x() {
      // These are type-dependent expressions, even though we could
      // determine that all calls have type 'int'.
      X<sizeof(f(0))>::type a;
      // expected-error@-1 {{expected ';' after expression}}
      // expected-error@-2 {{use of undeclared identifier 'a'}}
      X<sizeof(g(0))>::type b;
      // expected-error@-1 {{expected ';' after expression}}
      // expected-error@-2 {{use of undeclared identifier 'b'}}
      X<sizeof(h(0))>::type b;
      // expected-error@-1 {{expected ';' after expression}}
      // expected-error@-2 {{use of undeclared identifier 'b'}}

      typename X<sizeof(f(0))>::type a;
      typename X<sizeof(h(0))>::type b;
    }
  };
}

namespace cwg542 { // cwg542: yes
#if __cplusplus >= 201103L
  // In C++20 A and B are no longer aggregates and thus the constructor is
  // called, which fails.
  struct A { A() = delete; int n; }; // #cwg542-A
  // ok, constructor not called
  A a[32] = {}; // #cwg542-a
  // since-cxx20-error@-1 {{call to deleted constructor of 'A'}}
  //   since-cxx20-note@#cwg542-A {{'A' has been explicitly marked deleted here}}
  //   since-cxx20-note@#cwg542-a {{in implicit initialization of array element 0 with omitted initializer}}

  struct B {
    int n;
  private:
    B() = default; // #cwg542-B-ctor
  };
  B b[32] = {}; // ok, constructor not called
  // since-cxx20-error@-1 {{calling a private constructor of class 'cwg542::B'}}
  //   since-cxx20-note@#cwg542-B-ctor {{declared private here}}
#endif
}

namespace cwg543 { // cwg543: 3.0
  // In C++98+CWG543, this is valid because value-initialization doesn't call a
  // trivial default constructor, so we never notice that defining the
  // constructor would be ill-formed.
  //
  // In C++11+CWG543, this is ill-formed, because the default constructor is
  // deleted, and value-initialization *does* call a deleted default
  // constructor, even if it is trivial.
  struct A {
    const int n; // #cwg543-A-n
  };
  A a = A();
  // since-cxx11-error@-1 {{call to implicitly-deleted default constructor of 'A'}}
  //   since-cxx11-note@#cwg543-A-n {{default constructor of 'A' is implicitly deleted because field 'n' of const-qualified type 'const int' would not be initialized}}
}

namespace cwg544 { // cwg544: yes
  int *n;

  template<class T> struct A { int n; };
  template<class T> struct B : A<T> { int get(); };
  template<> int B<int>::get() { return n; }
  int k = B<int>().get();
}

namespace cwg546 { // cwg546: yes
  template<typename T> struct A { void f(); };
  template struct A<int>;
  template<typename T> void A<T>::f() { T::error; }
}

namespace cwg547 { // cwg547: 3.2
  template<typename T> struct X;
  template<typename T> struct X<T() const> {};
  template<typename T, typename C> X<T> f(T C::*) { return X<T>(); }

  struct S { void f() const; };
  X<void() const> x = f(&S::f);
}

namespace cwg548 { // cwg548: dup 482
  template<typename T> struct S {};
  template<typename T> void f() {}
  template struct cwg548::S<int>;
  template void cwg548::f<int>();
}

// cwg550: dup 393

namespace cwg551 { // cwg551: yes c++11
  // FIXME: This obviously should apply in C++98 mode too.
  template<typename T> void f() {}
  template inline void f<int>();
  // since-cxx11-error@-1 {{explicit instantiation cannot be 'inline'}}

  template<typename T> inline void g() {}
  template inline void g<int>();
  // since-cxx11-error@-1 {{explicit instantiation cannot be 'inline'}}

  template<typename T> struct X {
    void f() {}
  };
  template inline void X<int>::f();
  // since-cxx11-error@-1 {{explicit instantiation cannot be 'inline'}}
}

namespace cwg552 { // cwg552: yes
  template<typename T, typename T::U> struct X {};
  struct Y { typedef int U; };
  X<Y, 0> x;
}

// cwg553: 2.7
struct cwg553_class {
  friend void *operator new(size_t, cwg553_class);
};
namespace cwg553 {
  cwg553_class c;
  // Contrary to the apparent intention of the DR, operator new is not actually
  // looked up with a lookup mechanism that performs ADL; the standard says it
  // "is looked up in global scope", where it is not visible.
  void *p = new (c) int;
  // expected-error@-1 {{no matching function for call to 'operator new'}}
  //   since-cxx17-note@#cwg5xx-global-operator-new-aligned {{candidate function not viable: no known conversion from 'cwg553_class' to 'std::align_val_t' for 2nd argument}}
  //   expected-note@#cwg5xx-global-operator-new {{candidate function not viable: requires 1 argument, but 2 were provided}}

  struct namespace_scope {
    friend void *operator new(size_t, namespace_scope);
    // expected-error@-1 {{'operator new' cannot be declared inside a namespace}}
  };
}

// cwg554: na

namespace cwg555 { // cwg555: 2.8
typedef int I;
typedef const int CI;
typedef volatile int VI;
void f(int *a, CI *b, VI *c) {
  a->~I();
  a->~CI();
  a->~VI();
  a->I::~I();
  a->CI::~CI();
  a->VI::~VI();

  a->CI::~VI(); // allowed by changes to [expr.id.prim.qual]/2 introduced in P1131R2

  b->~I();
  b->~CI();
  b->~VI();
  b->I::~I();
  b->CI::~CI();
  b->VI::~VI();

  c->~I();
  c->~CI();
  c->~VI();
  c->I::~I();
  c->CI::~CI();
  c->VI::~VI();
}

void g(int &a, CI &b, VI &c) {
  a.~I();
  a.~CI();
  a.~VI();
  a.I::~I();
  a.CI::~CI();
  a.VI::~VI();

  a.CI::~VI(); // allowed by changes to [expr.id.prim.qual]/2 introduced in P1131R2

  b.~I();
  b.~CI();
  b.~VI();
  b.I::~I();
  b.CI::~CI();
  b.VI::~VI();

  c.~I();
  c.~CI();
  c.~VI();
  c.I::~I();
  c.CI::~CI();
  c.VI::~VI();
}
} // namespace cwg555

// cwg556: na

namespace cwg557 { // cwg557: 3.1
  template<typename T> struct S {
    friend void f(S<T> *);
    friend void g(S<S<T> > *);
  };
  void x(S<int> *p, S<S<int> > *q) {
    f(p);
    g(q);
  }
}

namespace cwg558 { // cwg558: 2.9
  wchar_t a = L'\uD7FF';
  wchar_t b = L'\xD7FF';
  wchar_t c = L'\uD800';
  // expected-error@-1 {{invalid universal character}}
  wchar_t d = L'\xD800';
  wchar_t e = L'\uDFFF';
  // expected-error@-1 {{invalid universal character}}
  wchar_t f = L'\xDFFF';
  wchar_t g = L'\uE000';
  wchar_t h = L'\xE000';
}

template<typename> struct cwg559 { typedef int T; cwg559::T u; }; // cwg559: yes

namespace cwg560 { // cwg560: 16

template <class T>
struct Outer {
  struct Inner {
    Inner* self();
  };
};
template <class T>
Outer<T>::Inner* Outer<T>::Inner::self() { return this; }
// cxx98-17-error@-1 {{missing 'typename' prior to dependent type name Outer<T>::Inner; implicit 'typename' is a C++20 extension}}

} // namespace cwg560

namespace cwg561 { // cwg561: yes
  template<typename T> void f(int);
  template<typename T> void g(T t) {
    f<T>(t);
  }
  namespace {
    struct S {};
    template<typename T> static void f(S);
  }
  void h(S s) {
    g(s);
  }
}

// cwg562: na
// cwg563 is in cwg563.cpp

namespace cwg564 { // cwg564: yes
  extern "C++" void f(int);
  void f(int); // ok
  extern "C++" { extern int n; }
  int n; // ok
}

namespace cwg565 { // cwg565: yes
  namespace N {
    template<typename T> int f(T); // #cwg565-f
  }
  using N::f; // #cwg565-using
  template<typename T> int f(T*);
  template<typename T> void f(T);
  template<typename T, int = 0> int f(T);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename T> int f(T, int = 0);
  template<typename T> int f(T);
  // expected-error@-1 {{declaration conflicts with target of using declaration already in scope}}
  //   expected-note@#cwg565-f {{target of using declaration}}
  //   expected-note@#cwg565-using {{using declaration}}
}

namespace cwg566 { // cwg566: yes
#if __cplusplus >= 201103L
  static_assert(int(-3.99) == -3, "");
#endif
}

// cwg567: na

namespace cwg568 { // cwg568: 3.0 c++11
  // FIXME: This is a DR issue against C++98, so should probably apply there
  // too.
  struct x { int y; };
  class trivial : x {
    x y;
  public:
    int n;
  };
  static_assert(__is_trivial(trivial), "");

  struct std_layout {
    std_layout();
    std_layout(const std_layout &);
    ~std_layout();
  private:
    int n;
  };
  static_assert(__is_standard_layout(std_layout), "");

  struct aggregate {
    int x;
    int y;
    trivial t;
    std_layout sl;
  };
  aggregate aggr = {};

  void f(...);
  void g(trivial t) { f(t); }
  // cxx98-error@-1 {{cannot pass object of non-POD type 'trivial' through variadic function; call will abort at runtime}}

  void jump() {
    goto x;
    // cxx98-error@-1 {{cannot jump from this goto statement to its label}}
    //   cxx98-note@#cwg568-t {{jump bypasses initialization of non-POD variable}}
    trivial t; // #cwg568-t
  x: ;
  }
}

namespace cwg569 { // cwg569: yes c++11
  // FIXME: This is a DR issue against C++98, so should probably apply there
  // too.
  ;;;;;
  // cxx98-error@-1 {{C++11 extension}}
}

namespace cwg570 { // cwg570: dup 633
  int n;
  int &r = n; // #cwg570-r
  int &r = n;
  // expected-error@-1 {{redefinition of 'r'}}
  //   expected-note@#cwg570-r {{previous definition is here}}
}

// cwg571 is in cwg571.cpp

namespace cwg572 { // cwg572: yes
  enum E { a = 1, b = 2 };
  static_assert(a + b == 3, "");
}

namespace cwg573 { // cwg573: no
  void *a;
  int *b = reinterpret_cast<int*>(a);
  void (*c)() = reinterpret_cast<void(*)()>(a);
  // cxx98-error@-1 {{cast between pointer-to-function and pointer-to-object is an extension}}
  void *d = reinterpret_cast<void*>(c);
  // cxx98-error@-1 {{cast between pointer-to-function and pointer-to-object is an extension}}
  void f() { delete a; }
  // cxx98-23-error@-1 {{cannot delete expression with pointer-to-'void' type 'void *'}}
  // since-cxx26-error@-2 {{cannot delete pointer to incomplete type 'void'}}
  int n = d - a;
  // expected-error@-1 {{arithmetic on pointers to void}}
  // FIXME: This is ill-formed.
  template<void*> struct S;
  template<int*> struct T;
}

namespace cwg574 { // cwg574: 3.0
  struct A {
    A &operator=(const A&) const; // #cwg574-A-copy-assign
  };
  struct B {
    B &operator=(const B&) volatile; // #cwg574-B-copy-assign
  };
#if __cplusplus >= 201103L
  struct C {
    C &operator=(const C&) &; // #cwg574-C-copy-assign
  };
  struct D {
    D &operator=(const D&) &&; // #cwg574-D-copy-assign
  };
  void test(C c, D d) {
    c = c;
    C() = c;
    // since-cxx11-error@-1 {{no viable overloaded '='}}
    //   since-cxx11-note@#cwg574-C-copy-assign {{candidate function not viable: expects an lvalue for object argument}}
    d = d;
    // since-cxx11-error@-1 {{no viable overloaded '='}}
    //   since-cxx11-note@#cwg574-D-copy-assign {{candidate function not viable: expects an rvalue for object argument}}
    D() = d;
  }
#endif
  struct Test {
    friend A &A::operator=(const A&);
    // expected-error@-1 {{friend declaration of 'operator=' does not match any declaration in 'cwg574::A'}}
    //   expected-note@#cwg574-A-copy-assign {{candidate function has different qualifiers (expected unqualified but found 'const')}}
    friend B &B::operator=(const B&);
    // expected-error@-1 {{friend declaration of 'operator=' does not match any declaration in 'cwg574::B'}}
    //   expected-note@#cwg574-B-copy-assign {{candidate function has different qualifiers (expected unqualified but found 'volatile')}}
#if __cplusplus >= 202302L
    friend C &C::operator=(const C&);
    // since-cxx23-error@-1 {{conflicting types for 'operator='}}
    //   since-cxx23-note@#cwg574-C-copy-assign {{previous declaration is here}}
    friend D &D::operator=(const D&);
    // since-cxx23-error@-1 {{conflicting types for 'operator='}}
    //   since-cxx23-note@#cwg574-D-copy-assign {{previous declaration is here}}
#elif __cplusplus >= 201103L
    // FIXME: We shouldn't produce the 'cannot overload' diagnostics here.
    friend C &C::operator=(const C&); // #cwg574-test-C
    // since-cxx11-error@#cwg574-test-C {{cannot overload}}
    //   since-cxx11-note@#cwg574-C-copy-assign {{previous declaration is here}}
    // since-cxx11-error@#cwg574-test-C {{friend declaration of 'operator=' does not match any declaration in 'cwg574::C'}}
    //   since-cxx11-note@#cwg574-C-copy-assign {{candidate function}}
    friend D &D::operator=(const D&); // #cwg574-test-D
    // since-cxx11-error@#cwg574-test-D {{cannot overload a member function without a ref-qualifier with a member function with ref-qualifier '&&'}}
    //   since-cxx11-note@#cwg574-D-copy-assign {{previous declaration is here}}
    // since-cxx11-error@#cwg574-test-D {{friend declaration of 'operator=' does not match any declaration in 'cwg574::D'}}
    //   since-cxx11-note@#cwg574-D-copy-assign {{candidate function}}
#endif
  };
}

namespace cwg575 { // cwg575: yes
  template<typename T, typename U = typename T::type> void a(T); void a(...);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename T, typename T::type U = 0> void b(T); void b(...);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename T, int U = T::value> void c(T); void c(...);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename T> void d(T, int = T::value); void d(...);
  // expected-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
  //   expected-note@#cwg575-d {{in instantiation of default function argument expression for 'd<int>' required here}}
  void x() {
    a(0);
    b(0);
    c(0);
    d(0); // #cwg575-d
  }

  template<typename T = int&> void f(T* = 0);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename T = int> void f(T = 0);
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  void g() { f<>(); }

  template<typename T> T &h(T *);
  template<typename T> T *h(T *);
  void *p = h((void*)0);
}

namespace cwg576 { // cwg576: 3.5
  typedef void f() {}
  // expected-error@-1 {{function definition declared 'typedef'}}
  void f(typedef int n);
  // expected-error@-1 {{invalid storage class specifier in function declarator}}
  void f(char c) { typedef int n; }
}

namespace cwg577 { // cwg577: 3.5
  typedef void V;
  typedef const void CV;
  void a(void);
  void b(const void);
  // expected-error@-1 {{'void' as parameter must not have type qualifiers}}
  void c(V);
  void d(CV);
  // expected-error@-1 {{'void' as parameter must not have type qualifiers}}
  void (*e)(void) = c;
  void (*f)(const void);
  // expected-error@-1 {{'void' as parameter must not have type qualifiers}}
  void (*g)(V) = a;
  void (*h)(CV);
  // expected-error@-1 {{'void' as parameter must not have type qualifiers}}
  template<typename T> void i(T); // #cwg577-i
  template<typename T> void j(void (*)(T)); // #cwg577-j
  void k() {
    a();
    c();
    i<void>();
    // expected-error@-1 {{no matching function for call to 'i'}}
    //   expected-note@#cwg577-i {{candidate function template not viable: requires 1 argument, but 0 were provided}}
    i<const void>();
    // expected-error@-1 {{no matching function for call to 'i'}}
    //   expected-note@#cwg577-i {{candidate function template not viable: requires 1 argument, but 0 were provided}}
    j<void>(0);
    // expected-error@-1 {{no matching function for call to 'j'}}
    //   expected-note@#cwg577-j {{candidate template ignored: substitution failure [with T = void]: argument may not have 'void' type}}
    j<const void>(0);
    // expected-error@-1 {{no matching function for call to 'j'}}
    //   expected-note@#cwg577-j {{candidate template ignored: substitution failure [with T = const void]: argument may not have 'void' type}}
  }
}

namespace cwg580 { // cwg580: partial
  class C;
  struct A { static C c; };
  struct B { static C c; };
  class C {
    C(); // #cwg580-C-ctor
    ~C(); // #cwg580-C-dtor

    typedef int I; // #cwg580-I
    template<int> struct X;
    template<int> friend struct Y;
    template<int> void f();
    template<int> friend void g();
    friend struct A;
  };

  template<C::I> struct C::X {};
  template<C::I> struct Y {};
  template<C::I> struct Z {};
  // expected-error@-1 {{'I' is a private member of 'cwg580::C'}}
  //   expected-note@#cwg580-I {{implicitly declared private here}}

  struct C2 {
    class X {
      struct A;
      typedef int I;
      friend struct A;
    };
    class Y {
      // FIXME: We incorrectly accept this
      // because we think C2::Y::A<...> might
      // instantiate to C2::X::A
      template<X::I> struct A {};
    };
  };

  template<C::I> void C::f() {}
  template<C::I> void g() {}
  template<C::I> void h() {}
  // expected-error@-1 {{'I' is a private member of 'cwg580::C'}}
  //   expected-note@#cwg580-I {{implicitly declared private here}}

  C A::c;
  C B::c; // #cwg580-c
  // expected-error@#cwg580-c {{calling a private constructor of class 'cwg580::C'}}
  //   expected-note@#cwg580-C-ctor {{implicitly declared private here}}
  // expected-error@#cwg580-c {{variable of type 'C' has private destructor}}
  //   expected-note@#cwg580-C-dtor {{implicitly declared private here}}
}

// cwg582: na

namespace cwg583 { // cwg583: 4
  // see n3624
  int *p;
  bool b1 = p < 0;
  // expected-error@-1 {{ordered comparison between pointer and zero ('int *' and 'int')}}
  bool b2 = p > 0;
  // expected-error@-1 {{ordered comparison between pointer and zero ('int *' and 'int')}}
  bool b3 = p <= 0;
  // expected-error@-1 {{ordered comparison between pointer and zero ('int *' and 'int')}}
  bool b4 = p >= 0;
  // expected-error@-1 {{ordered comparison between pointer and zero ('int *' and 'int')}}
}

// cwg584: na

namespace cwg585 { // cwg585: 3.0
  template<typename> struct T; // #cwg585-struct-T
  struct A {
    friend T;
    // cxx98-14-error@-1 {{a type specifier is required for all declarations}}
    // cxx98-14-error@-2 {{friends can only be classes or functions}}
    // since-cxx17-error@-3 {{use of class template 'T' requires template arguments; argument deduction not allowed in friend declaration}}
    //   since-cxx17-note@#cwg585-struct-T {{template is declared here}}
    // FIXME: It's not clear whether the standard allows this or what it means,
    // but the CWG585 writeup suggests it as an alternative.
    template<typename U> friend T<U>;
    // expected-error@-1 {{friend type templates must use an elaborated type}}
  };
  template<template<typename> class T> struct B { // #cwg585-template-T
    friend T;
    // cxx98-14-error@-1 {{a type specifier is required for all declarations}}
    // cxx98-14-error@-2 {{friends can only be classes or functions}}
    // since-cxx17-error@-3 {{use of template template parameter 'T' requires template arguments; argument deduction not allowed in friend declaration}}
    //   since-cxx17-note@#cwg585-template-T {{template is declared here}}
    template<typename U> friend T<U>;
    // expected-error@-1 {{friend type templates must use an elaborated type}}
  };
}

// cwg586: na

namespace cwg587 { // cwg587: 3.2
  template<typename T> void f(bool b, const T x, T y) {
    const T *p = &(b ? x : y);
  }
  struct S {};
  template void f(bool, const int, int);
  template void f(bool, const S, S);
}

namespace cwg588 { // cwg588: yes
  struct A { int n; }; // #cwg588-A
  template<typename T> int f() {
    struct S : A, T { int f() { return n; } } s;
    int a = s.f();
    int b = s.n;
    // expected-error@-1 {{member 'n' found in multiple base classes of different types}}
    //   expected-note@#cwg588-k {{in instantiation of function template specialization 'cwg588::f<cwg588::B>' requested here}}
    //   expected-note@#cwg588-A {{member found by ambiguous name lookup}}
    //   expected-note@#cwg588-B {{member found by ambiguous name lookup}}
  }
  struct B { int n; }; // #cwg588-B
  int k = f<B>(); // #cwg588-k
}

namespace cwg589 { // cwg589: yes
  struct B { };
  struct D : B { };
  D f();
  extern const B &b;
  bool a;
  const B *p = &(a ? f() : b);
  // expected-error@-1 {{taking the address of a temporary object of type 'const B'}}
  const B *q = &(a ? D() : b);
  // expected-error@-1 {{taking the address of a temporary object of type 'const B'}}
}

namespace cwg590 { // cwg590: yes
  template<typename T> struct A {
    struct B {
      struct C {
        A<T>::B::C f(A<T>::B::C); // ok, no 'typename' required.
      };
    };
  };
  template<typename T> typename A<T>::B::C A<T>::B::C::f(A<T>::B::C) {}
}

namespace cwg591 { // cwg591: no
  template<typename T> struct A {
    typedef int M;
    struct B {
      typedef void M;
      struct C;
    };
  };

  template<typename T> struct A<T>::B::C : A<T> {
    // FIXME: Should find member of non-dependent base class A<T>.
    M m;
    // expected-error@-1 {{field has incomplete type 'M' (aka 'void'}}
  };
}

// cwg592: na
// cwg593 is in cwg593.cpp
// cwg594: na

namespace cwg595 { // cwg595: dup 1330
  template<class T> struct X {
    void f() throw(T) {}
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    //   since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
  };
  struct S {
    X<S> xs;
  };
}

// cwg597: na

namespace cwg598 { // cwg598: yes
  namespace N {
    void f(int);
    void f(char);
    // Not found by ADL.
    void g(void (*)(int));
    void h(void (*)(int));

    namespace M {
      struct S {};
      int &h(void (*)(S));
    }
    void i(M::S);
    void i();
  }
  int &g(void(*)(char));
  int &r = g(N::f);
  int &s = h(N::f);
  // expected-error@-1 {{use of undeclared identifier 'h'}}
  int &t = h(N::i);
}

namespace cwg599 { // cwg599: partial
  typedef int Fn();
  struct S { operator void*(); };
  struct T { operator Fn*(); };
  struct U { operator int*(); operator void*(); }; // #cwg599-U
  struct V { operator int*(); operator Fn*(); };
  void f(void *p, void (*q)(), S s, T t, U u, V v) {
    delete p;
    // cxx98-23-error@-1 {{cannot delete expression with pointer-to-'void' type 'void *'}}
    // since-cxx26-error@-2 {{cannot delete pointer to incomplete type 'void'}}
    delete q;
    // expected-error@-1 {{cannot delete expression of type 'void (*)()'}}
    delete s;
    // cxx98-23-error@-1 {{cannot delete expression with pointer-to-'void' type 'void *'}}
    // since-cxx26-error@-2 {{cannot delete pointer to incomplete type 'void'}}
    delete t;
    // expected-error@-1 {{cannot delete expression of type 'T'}}
    // FIXME: This is valid, but is rejected due to a non-conforming GNU
    // extension allowing deletion of pointers to void.
    delete u;
    // expected-error@-1 {{ambiguous conversion of delete expression of type 'U' to a pointer}}
    //   expected-note@#cwg599-U {{conversion to pointer type 'int *'}}
    //   expected-note@#cwg599-U {{conversion to pointer type 'void *'}}
    delete v;
  }
}
