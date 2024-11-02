// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

// FIXME: This is included to avoid a diagnostic with no source location
// pointing at the implicit operator new. We can't match such a diagnostic
// with -verify.
__extension__ typedef __SIZE_TYPE__ size_t;
void *operator new(size_t); // expected-error 0-1{{missing exception spec}} expected-note{{candidate}}
#if __cplusplus > 201402L
namespace std {
  enum class align_val_t : size_t {};
}
void *operator new(size_t, std::align_val_t); // expected-note{{candidate}}
#endif

namespace dr500 { // dr500: dup 372
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

namespace dr501 { // dr501: yes
  struct A {
    friend void f() {}
    void g() {
      void (*p)() = &f; // expected-error {{undeclared identifier}}
    }
  };
}

namespace dr502 { // dr502: yes
  struct Q {};
  template<typename T> struct A {
    enum E { e = 1 };
    void q1() { f(e); }
    void q2() { Q arr[sizeof(E)]; f(arr); }
    void q3() { Q arr[e]; f(arr); }
    void sanity() { Q arr[1]; f(arr); } // expected-error {{undeclared identifier 'f'}}
  };
  int f(A<int>::E);
  template<int N> int f(Q (&)[N]);
  template struct A<int>;
}

namespace dr505 { // dr505: yes
  const char *exts = "\e\(\{\[\%"; // expected-error 5{{use of non-standard escape}}
  const char *unknown = "\Q"; // expected-error {{unknown escape sequence}}
}

namespace dr506 { // dr506: yes
  struct NonPod { ~NonPod(); };
  void f(...);
  void g(NonPod np) { f(np); } // expected-error {{cannot pass}}
}

// FIXME: Add tests here once DR260 is resolved.
// dr507: dup 260

// dr508: na
// dr509: na
// dr510: na

namespace dr512 { // dr512: yes
  struct A {
    A(int);
  };
  union U { A a; };
#if __cplusplus < 201103L
  // expected-error@-2 {{has a non-trivial default constructor}}
  // expected-note@-6 {{no default constructor}}
  // expected-note@-6 {{suppressed by user-declared constructor}}
#endif
}

// dr513: na

namespace dr514 { // dr514: yes
  namespace A { extern int x, y; }
  int A::x = y;
}

namespace dr515 { // dr515: sup 1017
  // FIXME: dr1017 reverses the wording of dr515, but the current draft has
  // dr515's wording, with a different fix for dr1017.

  struct X { int n; };
  template<typename T> struct Y : T {
    int f() { return X::n; }
  };
  int k = Y<X>().f();

  struct A { int a; };
  struct B { void f() { int k = sizeof(A::a); } };
#if __cplusplus < 201103L
  // expected-error@-2 {{invalid use of non-static data member}}
#endif
}

// dr516: na

namespace dr517 { // dr517: no
  // This is NDR, but we should diagnose it anyway.
  template<typename T> struct S {};
  template<typename T> int v = 0; // expected-error 0-1{{extension}}

  template struct S<int*>;
  template int v<int*>;

  S<char&> s;
  int k = v<char&>;

  // FIXME: These are both ill-formed.
  template<typename T> struct S<T*> {};
  template<typename T> int v<T*> = 0; // expected-error 0-1{{extension}}

  // FIXME: These are both ill-formed.
  template<typename T> struct S<T&> {};
  template<typename T> int v<T&> = 0; // expected-error 0-1{{extension}}
}

namespace dr518 { // dr518: yes c++11
  enum E { e, };
#if __cplusplus < 201103L
  // expected-error@-2 {{C++11 extension}}
#endif
}

namespace dr519 { // dr519: yes
// FIXME: Add a codegen test.
#if __cplusplus >= 201103L
#define fold(x) (__builtin_constant_p(x) ? (x) : (x))
  int test[fold((int*)(void*)0) ? -1 : 1];
#undef fold
#endif
}

// dr520: na

// dr521: no
// FIXME: The wording here is broken. It's not reasonable to expect a
// diagnostic here. Once the relevant DR gets a number, mark this as a dup.

namespace dr522 { // dr522: yes
  struct S {};
  template<typename T> void b1(volatile T &);
  template<typename T> void b2(volatile T * const *);
  template<typename T> void b2(volatile T * const S::*);
  template<typename T> void b2(volatile T * const S::* const *);
  template<typename T> void b2a(volatile T *S::* const *); // expected-note {{candidate template ignored: deduced type 'volatile int *dr522::S::*const *' of 1st parameter does not match adjusted type 'int *dr522::S::**' of argument}}

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
    b2a(am); // expected-error {{no matching function}}
    b3(d);
    b3(cd);
  }
}

namespace dr524 { // dr524: yes
  template<typename T> void f(T a, T b) { operator+(a, b); } // expected-error {{call}}

  struct S {};
  void operator+(S, S);
  template void f(S, S);

  namespace N { struct S {}; }
  void operator+(N::S, N::S); // expected-note {{should be declared}}
  template void f(N::S, N::S); // expected-note {{instantiation}}
}

namespace dr525 { // dr525: yes
  namespace before {
    // Note, the example was correct prior to the change; instantiation is
    // required for cases like this:
    template <class T> struct D { operator T*(); };
    void g(D<double> ppp) {
      delete ppp;
    }
  }
  namespace after {
    template <class T> struct D { typename T::error e; }; // expected-error {{prior to '::'}}
    void g(D<double> *ppp) {
      delete ppp; // expected-note {{instantiation of}}
    }
  }
}

namespace dr526 { // dr526: yes
  template<int> struct S {};
  template<int N> void f1(S<N> s);
  template<int N> void f2(S<(N)> s); // expected-note {{couldn't infer}}
  template<int N> void f3(S<+N> s); // expected-note {{couldn't infer}}
  template<int N> void g1(int (&)[N]);
  template<int N> void g2(int (&)[(N)]); // expected-note {{couldn't infer}}
  template<int N> void g3(int (&)[+N]); // expected-note {{couldn't infer}}

  void test(int (&a)[3], S<3> s) {
    f1(s);
    f2(s); // expected-error {{no matching}}
    f3(s); // expected-error {{no matching}}
    g1(a);
    g2(a); // expected-error {{no matching}}
    g3(a); // expected-error {{no matching}}
  }

  template<int N> struct X {
    typedef int type;
    X<N>::type v1;
    X<(N)>::type v2;
    X<+N>::type v3;
#if __cplusplus <= 201703L
    // expected-error@-3 {{implicit 'typename' is a C++20 extension}}
    // expected-error@-3 {{implicit 'typename' is a C++20 extension}}
#endif
  };
}

namespace dr527 { // dr527: na
  // This DR is meaningless. It removes a required diagnostic from the case
  // where a not-externally-visible object is odr-used but not defined, which
  // requires a diagnostic for a different reason.
  extern struct { int x; } a; // FIXME: We should reject this, per dr389.
  static struct { int x; } b;
  extern "C" struct { int x; } c;
  namespace { extern struct { int x; } d; }
  typedef struct { int x; } *P;
  struct E { static P e; }; // FIXME: We should reject this, per dr389.
  namespace { struct F { static P f; }; }

  int ax = a.x, bx = b.x, cx = c.x, dx = d.x, ex = E::e->x, fx = F::f->x;
}

namespace dr530 { // dr530: yes
  template<int*> struct S { enum { N = 1 }; };
  template<void(*)()> struct T { enum { N = 1 }; };
  int n;
  void f();
  int a[S<&n>::N];
  int b[T<&f>::N];
}

namespace dr531 { // dr531: partial
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
      template<typename U> struct C { typename T::error error; }; // expected-note {{here}}
      static T n;
    };
    template<typename T> T A<T>::n = T::error;

    void A<int>::f(int) {} // expected-error {{requires 'template<>'}}
    template<typename U> void A<int>::g(int, U) {} // expected-error {{should be empty}}
    struct A<int>::B {}; // expected-error {{requires 'template<>'}}
    template<typename U> struct A<int>::C {}; // expected-error {{should be empty}} expected-error {{different kind of symbol}}
    int A<int>::n = 0; // expected-error {{requires 'template<>'}}

    template<> struct A<char> { // expected-note 2{{here}}
      void f(char);
      template<typename U> void g(char, U);
      struct B; // expected-note {{here}}
      template<typename U> struct C;
      static char n;
    };

    template<> void A<char>::f(char) {} // expected-error {{no function template matches}}
    // FIXME: This is ill-formed; -pedantic-errors should reject.
    template<> template<typename U> void A<char>::g(char, U) {} // expected-warning {{extraneous template parameter list}}
    template<> struct A<char>::B {}; // expected-error {{extraneous 'template<>'}} expected-error {{does not specialize}}
    // FIXME: This is ill-formed; -pedantic-errors should reject.
    template<> template<typename U> struct A<char>::C {}; // expected-warning {{extraneous template parameter list}}
    template<> char A<char>::n = 0; // expected-error {{extraneous 'template<>'}}
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
    template<typename U> void A<int>::B<U>::g() {} // expected-error {{should be empty}}

    template<> template<typename U> template<typename V> void A<int>::B<U>::h() {}
    template<typename U> template<typename V> void A<int>::B<U>::i() {} // expected-error {{should be empty}}

#if __cplusplus <= 201703L
    // FIXME: All of those declarations shouldn't crash in C++20 mode.
    template<> template<> void A<int>::B<int>::f() {}
    template<> template<> template<typename V> void A<int>::B<int>::h() {}
    template<> template<> template<> void A<int>::B<int>::h<int>() {}

    template<> void A<int>::B<char>::f() {} // expected-error {{requires 'template<>'}}
    template<> template<typename V> void A<int>::B<char>::h() {} // expected-error {{should be empty}}
#endif
  }
}

// PR8130
namespace dr532 { // dr532: 3.5
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

// dr533: na

namespace dr534 { // dr534: yes
  struct S {};
  template<typename T> void operator+(S, T);
  template<typename T> void operator+<T*>(S, T*) {} // expected-error {{function template partial spec}}
}

namespace dr535 { // dr535: yes
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

// dr537: na
// dr538: na

// dr539: yes
const dr539( // expected-error {{a type specifier is required}}
    const a) { // expected-error {{unknown type name 'a'}}
  const b; // expected-error {{a type specifier is required}}
  new const; // expected-error {{expected a type}}
  try {} catch (const n) {} // expected-error {{unknown type name 'n'}}
  try {} catch (const) {} // expected-error {{expected a type}}
  if (const n = 0) {} // expected-error {{a type specifier is required}}
  switch (const n = 0) {} // expected-error {{a type specifier is required}}
  while (const n = 0) {} // expected-error {{a type specifier is required}}
  for (const n = 0; // expected-error {{a type specifier is required}}
       const m = 0; ) {} // expected-error {{a type specifier is required}}
  sizeof(const); // expected-error {{a type specifier is required}}
  struct S {
    const n; // expected-error {{a type specifier is required}}
    operator const(); // expected-error {{expected a type}}
  };
#if __cplusplus >= 201103L
  int arr[3];
  // FIXME: The extra braces here are to avoid the parser getting too
  // badly confused when recovering here. We should fix this recovery.
  { for (const n // expected-error {{unknown type name 'n'}} expected-note {{}}
         : arr) ; {} } // expected-error +{{}}
  (void) [](const) {}; // expected-error {{a type specifier is required}}
  (void) [](const n) {}; // expected-error {{unknown type name 'n'}}
  enum E : const {}; // expected-error {{expected a type}}
  using T = const; // expected-error {{expected a type}}
  auto f() -> const; // expected-error {{expected a type}}
#endif
}

namespace dr540 { // dr540: yes
  typedef int &a;
  typedef const a &a; // expected-warning {{has no effect}}
  typedef const int &b;
  typedef b &b;
  typedef const a &c; // expected-note {{previous}} expected-warning {{has no effect}}
  typedef const b &c; // expected-error {{different}} expected-warning {{has no effect}}
}

namespace dr541 { // dr541: yes
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
      X<sizeof(f(0))>::type a; // expected-error +{{}}
      X<sizeof(g(0))>::type b; // expected-error +{{}}
      X<sizeof(h(0))>::type b; // expected-error +{{}}

      typename X<sizeof(f(0))>::type a;
      typename X<sizeof(h(0))>::type b;
    }
  };
}

namespace dr542 { // dr542: yes
#if __cplusplus >= 201103L
  // In C++20 A and B are no longer aggregates and thus the constructor is
  // called, which fails.
  struct A { A() = delete; int n; };
  A a[32] = {}; // ok, constructor not called
#if __cplusplus > 201703L
  // expected-error@-2 {{call to deleted constructor}}
  // expected-note@-3 {{in implicit initialization}}
  // expected-note@-5 {{marked deleted here}}
#endif

  struct B {
    int n;
  private:
    B() = default;
  };
  B b[32] = {}; // ok, constructor not called
#if __cplusplus > 201703L
  // expected-error@-2 {{calling a private constructor}}
  // expected-note@-5 {{declared private here}}
#endif
#endif
}

namespace dr543 { // dr543: yes
  // In C++98+DR543, this is valid because value-initialization doesn't call a
  // trivial default constructor, so we never notice that defining the
  // constructor would be ill-formed.
  //
  // In C++11+DR543, this is ill-formed, because the default constructor is
  // deleted, and value-initialization *does* call a deleted default
  // constructor, even if it is trivial.
  struct A {
    const int n;
  };
  A a = A();
#if __cplusplus >= 201103L
  // expected-error@-2 {{deleted}}
  // expected-note@-5 {{would not be initialized}}
#endif
}

namespace dr544 { // dr544: yes
  int *n;

  template<class T> struct A { int n; };
  template<class T> struct B : A<T> { int get(); };
  template<> int B<int>::get() { return n; }
  int k = B<int>().get();
}

namespace dr546 { // dr546: yes
  template<typename T> struct A { void f(); };
  template struct A<int>;
  template<typename T> void A<T>::f() { T::error; }
}

namespace dr547 { // dr547: yes
  template<typename T> struct X;
  template<typename T> struct X<T() const> {};
  template<typename T, typename C> X<T> f(T C::*) { return X<T>(); }

  struct S { void f() const; };
  X<void() const> x = f(&S::f);
}

namespace dr548 { // dr548: dup 482
  template<typename T> struct S {};
  template<typename T> void f() {}
  template struct dr548::S<int>;
  template void dr548::f<int>();
}

namespace dr551 { // dr551: yes c++11
  // FIXME: This obviously should apply in C++98 mode too.
  template<typename T> void f() {}
  template inline void f<int>();
#if __cplusplus >= 201103L
  // expected-error@-2 {{cannot be 'inline'}}
#endif

  template<typename T> inline void g() {}
  template inline void g<int>();
#if __cplusplus >= 201103L
  // expected-error@-2 {{cannot be 'inline'}}
#endif

  template<typename T> struct X {
    void f() {}
  };
  template inline void X<int>::f();
#if __cplusplus >= 201103L
  // expected-error@-2 {{cannot be 'inline'}}
#endif
}

namespace dr552 { // dr552: yes
  template<typename T, typename T::U> struct X {};
  struct Y { typedef int U; };
  X<Y, 0> x;
}

struct dr553_class {
  friend void *operator new(size_t, dr553_class);
};
namespace dr553 {
  dr553_class c;
  // Contrary to the apparent intention of the DR, operator new is not actually
  // looked up with a lookup mechanism that performs ADL; the standard says it
  // "is looked up in global scope", where it is not visible.
  void *p = new (c) int; // expected-error {{no matching function}}

  struct namespace_scope {
    friend void *operator new(size_t, namespace_scope); // expected-error {{cannot be declared inside a namespace}}
  };
}

// dr556: na

namespace dr557 { // dr557: yes
  template<typename T> struct S {
    friend void f(S<T> *);
    friend void g(S<S<T> > *);
  };
  void x(S<int> *p, S<S<int> > *q) {
    f(p);
    g(q);
  }
}

namespace dr558 { // dr558: yes
  wchar_t a = L'\uD7FF';
  wchar_t b = L'\xD7FF';
  wchar_t c = L'\uD800'; // expected-error {{invalid universal character}}
  wchar_t d = L'\xD800';
  wchar_t e = L'\uDFFF'; // expected-error {{invalid universal character}}
  wchar_t f = L'\xDFFF';
  wchar_t g = L'\uE000';
  wchar_t h = L'\xE000';
}

template<typename> struct dr559 { typedef int T; dr559::T u; }; // dr559: yes

namespace dr561 { // dr561: yes
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

namespace dr564 { // dr564: yes
  extern "C++" void f(int);
  void f(int); // ok
  extern "C++" { extern int n; }
  int n; // ok
}

namespace dr565 { // dr565: yes
  namespace N {
    template<typename T> int f(T); // expected-note {{target}}
  }
  using N::f; // expected-note {{using}}
  template<typename T> int f(T*);
  template<typename T> void f(T);
  template<typename T, int = 0> int f(T); // expected-error 0-1{{extension}}
  template<typename T> int f(T, int = 0);
  template<typename T> int f(T); // expected-error {{conflicts with}}
}

namespace dr566 { // dr566: yes
#if __cplusplus >= 201103L
  int check[int(-3.99) == -3 ? 1 : -1];
#endif
}

// dr567: na

namespace dr568 { // dr568: yes c++11
  // FIXME: This is a DR issue against C++98, so should probably apply there
  // too.
  struct x { int y; };
  class trivial : x {
    x y;
  public:
    int n;
  };
  int check_trivial[__is_trivial(trivial) ? 1 : -1];

  struct std_layout {
    std_layout();
    std_layout(const std_layout &);
    ~std_layout();
  private:
    int n;
  };
  int check_std_layout[__is_standard_layout(std_layout) ? 1 : -1];

  struct aggregate {
    int x;
    int y;
    trivial t;
    std_layout sl;
  };
  aggregate aggr = {};

  void f(...);
  void g(trivial t) { f(t); }
#if __cplusplus < 201103L
  // expected-error@-2 {{non-POD}}
#endif

  void jump() {
    goto x;
#if __cplusplus < 201103L
    // expected-error@-2 {{cannot jump}}
    // expected-note@+2 {{non-POD}}
#endif
    trivial t;
  x: ;
  }
}

namespace dr569 { // dr569: yes c++11
  // FIXME: This is a DR issue against C++98, so should probably apply there
  // too.
  ;;;;;
#if __cplusplus < 201103L
  // expected-error@-2 {{C++11 extension}}
#endif
}

namespace dr570 { // dr570: dup 633
  int n;
  int &r = n; // expected-note {{previous}}
  int &r = n; // expected-error {{redefinition}}
}

namespace dr571 { // dr571 unknown
  // FIXME: Add a codegen test.
  typedef int &ir;
  int n;
  const ir r = n; // expected-warning {{has no effect}} FIXME: Test if this has internal linkage.
}

namespace dr572 { // dr572: yes
  enum E { a = 1, b = 2 };
  int check[a + b == 3 ? 1 : -1];
}

namespace dr573 { // dr573: no
  void *a;
  int *b = reinterpret_cast<int*>(a);
  void (*c)() = reinterpret_cast<void(*)()>(a);
  void *d = reinterpret_cast<void*>(c);
#if __cplusplus < 201103L
  // expected-error@-3 {{extension}}
  // expected-error@-3 {{extension}}
#endif
  void f() { delete a; } // expected-error {{cannot delete}}
  int n = d - a; // expected-error {{arithmetic on pointers to void}}
  // FIXME: This is ill-formed.
  template<void*> struct S;
  template<int*> struct T;
}

namespace dr574 { // dr574: yes
  struct A {
    A &operator=(const A&) const; // expected-note {{different qualifiers}}
  };
  struct B {
    B &operator=(const B&) volatile; // expected-note {{different qualifiers}}
  };
#if __cplusplus >= 201103L
  struct C {
    C &operator=(const C&) &; // expected-note {{not viable}} expected-note {{candidate}} expected-note {{here}}
  };
  struct D {
    D &operator=(const D&) &&; // expected-note {{not viable}} expected-note {{candidate}} expected-note {{here}}
  };
  void test(C c, D d) {
    c = c;
    C() = c; // expected-error {{no viable}}
    d = d; // expected-error {{no viable}}
    D() = d;
  }
#endif
  struct Test {
    friend A &A::operator=(const A&); // expected-error {{does not match}}
    friend B &B::operator=(const B&); // expected-error {{does not match}}
#if __cplusplus >= 201103L
    // FIXME: We shouldn't produce the 'cannot overload' diagnostics here.
    friend C &C::operator=(const C&); // expected-error {{does not match}} expected-error {{cannot overload}}
    friend D &D::operator=(const D&); // expected-error {{does not match}} expected-error {{cannot overload}}
#endif
  };
}

namespace dr575 { // dr575: yes
  template<typename T, typename U = typename T::type> void a(T); void a(...); // expected-error 0-1{{extension}}
  template<typename T, typename T::type U = 0> void b(T); void b(...); // expected-error 0-1{{extension}}
  template<typename T, int U = T::value> void c(T); void c(...); // expected-error 0-1{{extension}}
  template<typename T> void d(T, int = T::value); void d(...); // expected-error {{cannot be used prior to '::'}}
  void x() {
    a(0);
    b(0);
    c(0);
    d(0); // expected-note {{in instantiation of default function argument}}
  }

  template<typename T = int&> void f(T* = 0); // expected-error 0-1{{extension}}
  template<typename T = int> void f(T = 0); // expected-error 0-1{{extension}}
  void g() { f<>(); }

  template<typename T> T &h(T *);
  template<typename T> T *h(T *);
  void *p = h((void*)0);
}

namespace dr576 { // dr576: yes
  typedef void f() {} // expected-error {{function definition declared 'typedef'}}
  void f(typedef int n); // expected-error {{invalid storage class}}
  void f(char c) { typedef int n; }
}

namespace dr577 { // dr577: yes
  typedef void V;
  typedef const void CV;
  void a(void);
  void b(const void); // expected-error {{qualifiers}}
  void c(V);
  void d(CV); // expected-error {{qualifiers}}
  void (*e)(void) = c;
  void (*f)(const void); // expected-error {{qualifiers}}
  void (*g)(V) = a;
  void (*h)(CV); // expected-error {{qualifiers}}
  template<typename T> void i(T); // expected-note 2{{requires 1 arg}}
  template<typename T> void j(void (*)(T)); // expected-note 2{{argument may not have 'void' type}}
  void k() {
    a();
    c();
    i<void>(); // expected-error {{no match}}
    i<const void>(); // expected-error {{no match}}
    j<void>(0); // expected-error {{no match}}
    j<const void>(0); // expected-error {{no match}}
  }
}

namespace dr580 { // dr580: partial
  class C;
  struct A { static C c; };
  struct B { static C c; };
  class C {
    C(); // expected-note {{here}}
    ~C(); // expected-note {{here}}

    typedef int I; // expected-note 2{{here}}
    template<int> struct X;
    template<int> friend struct Y;
    template<int> void f();
    template<int> friend void g();
    friend struct A;
  };

  template<C::I> struct C::X {};
  template<C::I> struct Y {};
  template<C::I> struct Z {}; // expected-error {{private}}

  struct C2 {
    class X {
      struct A;
      typedef int I;
      friend struct A;
    };
    class Y {
      template<X::I> struct A {}; // FIXME: We incorrectly accept this
                                  // because we think C2::Y::A<...> might
                                  // instantiate to C2::X::A
    };
  };

  template<C::I> void C::f() {}
  template<C::I> void g() {}
  template<C::I> void h() {} // expected-error {{private}}

  C A::c;
  C B::c; // expected-error 2{{private}}
}

// dr582: na

namespace dr583 { // dr583: 4
  // see n3624
  int *p;
  bool b1 = p < 0; // expected-error {{ordered comparison between pointer and zero}}
  bool b2 = p > 0; // expected-error {{ordered comparison between pointer and zero}}
  bool b3 = p <= 0; // expected-error {{ordered comparison between pointer and zero}}
  bool b4 = p >= 0; // expected-error {{ordered comparison between pointer and zero}}
}

// dr584: na

namespace dr585 { // dr585: yes
  template<typename> struct T;
  struct A {
    friend T;
#if __cplusplus <= 201402L
    // expected-error@-2 {{a type specifier is required}} expected-error@-2 {{can only be classes or functions}}
#else
    // expected-error@-4 {{use of class template 'T' requires template arguments; argument deduction not allowed in friend declaration}}
    // expected-note@-7 {{here}}
#endif
    // FIXME: It's not clear whether the standard allows this or what it means,
    // but the DR585 writeup suggests it as an alternative.
    template<typename U> friend T<U>; // expected-error {{must use an elaborated type}}
  };
  template<template<typename> class T> struct B {
    friend T;
#if __cplusplus <= 201402L
    // expected-error@-2 {{a type specifier is required}} expected-error@-2 {{can only be classes or functions}}
#else
    // expected-error@-4 {{use of template template parameter 'T' requires template arguments; argument deduction not allowed in friend declaration}}
    // expected-note@-6 {{here}}
#endif
    template<typename U> friend T<U>; // expected-error {{must use an elaborated type}}
  };
}

// dr586: na

namespace dr587 { // dr587: yes
  template<typename T> void f(bool b, const T x, T y) {
    const T *p = &(b ? x : y);
  }
  struct S {};
  template void f(bool, const int, int);
  template void f(bool, const S, S);
}

namespace dr588 { // dr588: yes
  struct A { int n; }; // expected-note {{ambiguous}}
  template<typename T> int f() {
    struct S : A, T { int f() { return n; } } s;
    int a = s.f();
    int b = s.n; // expected-error {{found in multiple}}
  }
  struct B { int n; }; // expected-note {{ambiguous}}
  int k = f<B>(); // expected-note {{here}}
}

namespace dr589 { // dr589: yes
  struct B { };
  struct D : B { };
  D f();
  extern const B &b;
  bool a;
  const B *p = &(a ? f() : b); // expected-error {{temporary}}
  const B *q = &(a ? D() : b); // expected-error {{temporary}}
}

namespace dr590 { // dr590: yes
  template<typename T> struct A {
    struct B {
      struct C {
        A<T>::B::C f(A<T>::B::C); // ok, no 'typename' required.
      };
    };
  };
  template<typename T> typename A<T>::B::C A<T>::B::C::f(A<T>::B::C) {}
}

namespace dr591 { // dr591: no
  template<typename T> struct A {
    typedef int M;
    struct B {
      typedef void M;
      struct C;
    };
  };

  template<typename T> struct A<T>::B::C : A<T> {
    // FIXME: Should find member of non-dependent base class A<T>.
    M m; // expected-error {{incomplete type 'M' (aka 'void'}}
  };
}

// dr592: na
// dr593 needs an IRGen test.
// dr594: na

namespace dr595 { // dr595: dup 1330
  template<class T> struct X {
    void f() throw(T) {}
#if __cplusplus > 201402L
    // expected-error@-2 {{ISO C++17 does not allow}} expected-note@-2 {{use 'noexcept}}
#endif
  };
  struct S {
    X<S> xs;
  };
}

// dr597: na

namespace dr598 { // dr598: yes
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
  int &s = h(N::f); // expected-error {{undeclared}}
  int &t = h(N::i);
}

namespace dr599 { // dr599: partial
  typedef int Fn();
  struct S { operator void*(); };
  struct T { operator Fn*(); };
  struct U { operator int*(); operator void*(); }; // expected-note 2{{conversion}}
  struct V { operator int*(); operator Fn*(); };
  void f(void *p, void (*q)(), S s, T t, U u, V v) {
    delete p; // expected-error {{cannot delete}}
    delete q; // expected-error {{cannot delete}}
    delete s; // expected-error {{cannot delete}}
    delete t; // expected-error {{cannot delete}}
    // FIXME: This is valid, but is rejected due to a non-conforming GNU
    // extension allowing deletion of pointers to void.
    delete u; // expected-error {{ambiguous}}
    delete v;
  }
}
