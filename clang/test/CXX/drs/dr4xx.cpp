// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

// FIXME: __SIZE_TYPE__ expands to 'long long' on some targets.
__extension__ typedef __SIZE_TYPE__ size_t;

namespace std { struct type_info; }

namespace dr400 { // dr400: yes
  struct A { int a; struct a {}; }; // expected-note 2{{conflicting}} expected-note {{ambiguous}}
  struct B { int a; struct a {}; }; // expected-note 2{{target}} expected-note {{ambiguous}}
  struct C : A, B { using A::a; struct a b; };
  struct D : A, B { using A::a; using B::a; struct a b; }; // expected-error 2{{conflicts}}
  struct E : A, B { struct a b; }; // expected-error {{found in multiple base classes}}
}

namespace dr401 { // dr401: yes
  template<class T, class U = typename T::type> class A : public T {}; // expected-error {{protected}} expected-error 2{{private}}

  class B {
  protected:
    typedef int type; // expected-note {{protected}}
  };

  class C {
    typedef int type; // expected-note {{private}}
    friend class A<C>; // expected-note {{default argument}}
  };

  class D {
    typedef int type; // expected-note {{private}}
    friend class A<D, int>;
  };

  A<B> *b; // expected-note {{default argument}}
  // FIXME: We're missing the "in instantiation of" note for the default
  // argument here.
  A<D> *d;

  struct E {
    template<class T, class U = typename T::type> class A : public T {};
  };
  class F {
    typedef int type;
    friend class E;
  };
  E::A<F> eaf; // ok, default argument is in befriended context

  // FIXME: Why do we get different diagnostics in C++11 onwards here? We seem
  // to not treat the default template argument as a SFINAE context in C++98.
  template<class T, class U = typename T::type> void f(T) {}
  void g(B b) { f(b); }
#if __cplusplus < 201103L
  // expected-error@-3 0-1{{extension}} expected-error@-3 {{protected}} expected-note@-3 {{instantiation}}
  // expected-note@-3 {{substituting}}
#else
  // expected-error@-5 {{no matching}} expected-note@-6 {{protected}}
#endif
}

namespace dr403 { // dr403: yes
  namespace A {
    struct S {};
    int f(void*);
  }
  template<typename T> struct X {};
  typedef struct X<A::S>::X XS;
  XS *p;
  int k = f(p); // ok, finds A::f, even though type XS is a typedef-name
                // referring to an elaborated-type-specifier naming a
                // injected-class-name, which is about as far from a
                // template-id as we can make it.
}

// dr404: na
// (NB: also sup 594)

namespace dr406 { // dr406: yes
  typedef struct {
    static int n; // expected-error {{static data member 'n' not allowed in anonymous struct}}
  } A;
}

namespace dr407 { // dr407: 3.8
  struct S;
  typedef struct S S;
  void f() {
    struct S *p;
    {
      typedef struct S S; // expected-note {{here}}
      struct S *p; // expected-error {{typedef 'S' cannot be referenced with a struct specifier}}
    }
  }
  struct S {};

  namespace UsingDir {
    namespace A {
      struct S {}; // expected-note {{found}}
    }
    namespace B {
      typedef int S; // expected-note {{found}}
    }
    namespace C {
      using namespace A;
      using namespace B;
      struct S s; // expected-error {{ambiguous}}
    }
    namespace D {
      using A::S;
      typedef struct S S;
      struct S s;
    }
    namespace E {
      // The standard doesn't say whether this is valid. We interpret
      // DR407 as meaning "if lookup finds both a tag and a typedef with the
      // same type, then it's OK in an elaborated-type-specifier".
      typedef A::S S;
      using A::S;
      struct S s;
    }
    namespace F {
      typedef A::S S;
    }
    // The standard doesn't say what to do in these cases either.
    namespace G {
      using namespace A;
      using namespace F;
      struct S s;
    }
    namespace H {
      using namespace F;
      using namespace A;
      struct S s;
    }
  }
}

namespace dr408 { // dr408: 3.4
  template<int N> void g() { int arr[N != 1 ? 1 : -1]; }
  template<> void g<2>() { }

  template<typename T> struct S {
    static int i[];
    void f();
  };
  template<typename T> int S<T>::i[] = { 1 };

  template<typename T> void S<T>::f() {
    g<sizeof (i) / sizeof (int)>();
  }
  template<> int S<int>::i[] = { 1, 2 };
  template void S<int>::f(); // uses g<2>(), not g<1>().


  template<typename T> struct R {
    static int arr[];
    void f();
  };
  template<typename T> int R<T>::arr[1];
  template<typename T> void R<T>::f() {
    int arr[sizeof(arr) != sizeof(int) ? 1 : -1];
  }
  template<> int R<int>::arr[2];
  template void R<int>::f();
}

namespace dr409 { // dr409: yes
  template<typename T> struct A {
    typedef int B;
    B b1;
    A::B b2;
    A<T>::B b3;
    A<T*>::B b4; // expected-error {{missing 'typename'}}
  };
}

namespace dr410 { // dr410: no
  template<class T> void f(T);
  void g(int);
  namespace M {
    template<class T> void h(T);
    template<class T> void i(T);
    struct A {
      friend void f<>(int);
      friend void h<>(int);
      friend void g(int);
      template<class T> void i(T);
      friend void i<>(int);
    private:
      static void z(); // expected-note {{private}}
    };

    template<> void h(int) { A::z(); }
    // FIXME: This should be ill-formed. The member A::i<> is befriended,
    // not this function.
    template<> void i(int) { A::z(); }
  }
  template<> void f(int) { M::A::z(); }
  void g(int) { M::A::z(); } // expected-error {{private}}
}

// dr412 is in its own file.

namespace dr413 { // dr413: yes
  struct S {
    int a;
    int : 17;
    int b;
  };
  S s = { 1, 2, 3 }; // expected-error {{excess elements}}

  struct E {};
  struct T { // expected-note {{here}}
    int a;
    E e;
    int b;
  };
  T t1 = { 1, {}, 2 };
  T t2 = { 1, 2 }; // expected-error {{aggregate with no elements requires explicit braces}}
}

namespace dr414 { // dr414: dup 305
  struct X {};
  void f() {
    X x;
    struct X {};
    x.~X();
  }
}

namespace dr415 { // dr415: yes
  template<typename T> void f(T, ...) { T::error; }
  void f(int, int);
  void g() { f(0, 0); } // ok
}

namespace dr416 { // dr416: yes
  extern struct A a;
  int &operator+(const A&, const A&);
  int &k = a + a;
  struct A { float &operator+(A&); };
  float &f = a + a;
}

namespace dr417 { // dr417: no
  struct A;
  struct dr417::A {}; // expected-warning {{extra qualification}}
  struct B { struct X; };
  struct C : B {};
  struct C::X {}; // expected-error {{no struct named 'X' in 'dr417::C'}}
  struct B::X { struct Y; };
  struct C::X::Y {}; // ok!
  namespace N {
    struct D;
    struct E;
    struct F;
    struct H;
  }
  // FIXME: This is ill-formed.
  using N::D;
  struct dr417::D {}; // expected-warning {{extra qualification}}
  using namespace N;
  struct dr417::E {}; // expected-warning {{extra qualification}} expected-error {{no struct named 'E'}}
  struct N::F {};
  struct G;
  using N::H;
  namespace M {
    struct dr417::G {}; // expected-error {{namespace 'M' does not enclose}}
    struct dr417::H {}; // expected-error {{namespace 'M' does not enclose}}
  }
}

namespace dr420 { // dr420: yes
  template<typename T> struct ptr {
    T *operator->() const;
    T &operator*() const;
  };
  template<typename T, typename P> void test(P p) {
    p->~T();
    p->T::~T();
    (*p).~T();
    (*p).T::~T();
  }
  struct X {};
  template void test<int>(int*);
  template void test<int>(ptr<int>);
  template void test<X>(X*);
  template void test<X>(ptr<X>);

  template<typename T>
  void test2(T p) {
    p->template Y<int>::~Y<int>();
    p->~Y<int>();
    // FIXME: This is ill-formed, but this diagnostic is terrible. We should
    // reject this in the parser.
    p->template ~Y<int>(); // expected-error 2{{no member named '~typename Y<int>'}}
  }
  template<typename T> struct Y {};
  template void test2(Y<int>*); // expected-note {{instantiation}}
  template void test2(ptr<Y<int> >); // expected-note {{instantiation}}

  void test3(int *p, ptr<int> q) {
    typedef int Int;
    p->~Int();
    q->~Int();
    p->Int::~Int();
    q->Int::~Int();
  }

#if __cplusplus >= 201103L
  template<typename T> using id = T;
  struct A { template<typename T> using id = T; };
  void test4(int *p, ptr<int> q) {
    p->~id<int>();
    q->~id<int>();
    p->id<int>::~id<int>();
    q->id<int>::~id<int>();
    p->template id<int>::~id<int>(); // expected-error {{expected unqualified-id}}
    q->template id<int>::~id<int>(); // expected-error {{expected unqualified-id}}
    p->A::template id<int>::~id<int>();
    q->A::template id<int>::~id<int>();
  }
#endif
}

namespace dr421 { // dr421: yes
  struct X { X(); int n; int &r; };
  int *p = &X().n; // expected-error-re {{{{taking the address of a temporary|cannot take the address of an rvalue}}}}
  int *q = &X().r;
}

namespace dr422 { // dr422: yes
  template<typename T, typename U> void f() {
    typedef T type; // expected-note {{prev}}
    typedef U type; // expected-error {{redef}}
  }
  template void f<int, int>();
  template void f<int, char>(); // expected-note {{instantiation}}
}

namespace dr423 { // dr423: yes
  template<typename T> struct X { operator T&(); };
  void f(X<int> x) { x += 1; }
}

namespace dr424 { // dr424: yes
  struct A {
    typedef int N; // expected-note {{previous}}
    typedef int N; // expected-error {{redefinition}}

    struct X;
    typedef X X; // expected-note {{previous}}
    struct X {};

    struct X *p;
    struct A::X *q;
    X *r;

    typedef X X; // expected-error {{redefinition}}
  };
  struct B {
    typedef int N;
  };
  struct C : B {
    typedef int N; // expected-note {{previous}}
    typedef int N; // expected-error {{redefinition}}
  };
}

namespace dr425 { // dr425: yes
  struct A { template<typename T> operator T() const; } a;
  float f = 1.0f * a; // expected-error {{ambiguous}} expected-note 5+{{built-in candidate}}

  template<typename T> struct is_float;
  template<> struct is_float<float> { typedef void type; };

  struct B {
    template<typename T, typename U = typename is_float<T>::type> operator T() const; // expected-error 0-1{{extension}}
  } b;
  float g = 1.0f * b; // ok
}

namespace dr427 { // dr427: yes
  struct B {};
  struct D : public B {
    D(B &) = delete; // expected-error 0-1{{extension}} expected-note {{deleted}}
  };

  extern D d1;
  B &b = d1;
  const D &d2 = static_cast<const D&>(b);
  const D &d3 = (const D&)b;
  const D &d4(b); // expected-error {{deleted}}
}

namespace dr428 { // dr428: yes
  template<typename T> T make();
  extern struct X x; // expected-note 5{{forward declaration}}
  void f() {
    throw void(); // expected-error {{cannot throw}}
    throw make<void*>();
    throw make<const volatile void*>();
    throw x; // expected-error {{cannot throw}}
    throw make<X&>(); // expected-error {{cannot throw}}
    throw make<X*>(); // expected-error {{cannot throw}}
    throw make<const volatile X&>(); // expected-error {{cannot throw}}
    throw make<const volatile X*>(); // expected-error {{cannot throw}}
  }
}

namespace dr429 { // dr429: yes c++11
  // FIXME: This rule is obviously intended to apply to C++98 as well.
  struct A {
    static void *operator new(size_t, size_t);
    static void operator delete(void*, size_t);
  } *a = new (0) A;
#if __cplusplus >= 201103L
  // expected-error@-2 {{'new' expression with placement arguments refers to non-placement 'operator delete'}}
  // expected-note@-4 {{here}}
#endif
  struct B {
    static void *operator new(size_t, size_t);
    static void operator delete(void*);
    static void operator delete(void*, size_t);
  } *b = new (0) B; // ok, second delete is not a non-placement deallocation function
}

namespace dr430 { // dr430: yes c++11
  // resolved by n2239
  // FIXME: This should apply in C++98 too.
  void f(int n) {
    int a[] = { n++, n++, n++ };
#if __cplusplus < 201103L
    // expected-warning@-2 {{multiple unsequenced modifications to 'n'}}
#endif
  }
}

namespace dr431 { // dr431: yes
  struct A {
    template<typename T> T *get();
    template<typename T> struct B {
      template<typename U> U *get();
    };
  };

  template<typename T> void f(A a) {
    a.get<A>()->get<T>();
    a.get<T>()
        ->get<T>(); // expected-error {{use 'template'}}
    a.get<T>()->template get<T>();
    a.A::get<T>();
    A::B<int> *b = a.get<A::B<int> >();
    b->get<int>();
    b->A::B<int>::get<int>();
    b->A::B<int>::get<T>();
    b->A::B<T>::get<int>(); // expected-error {{use 'template'}}
    b->A::B<T>::template get<int>();
    b->A::B<T>::get<T>(); // expected-error {{use 'template'}}
    b->A::B<T>::template get<T>();
    A::B<T> *c = a.get<A::B<T> >();
    c->get<int>(); // expected-error {{use 'template'}}
    c->template get<int>();
  }
}

namespace dr432 { // dr432: yes
  template<typename T> struct A {};
  template<typename T> struct B : A<B> {}; // expected-error {{requires template arguments}} expected-note {{declared}}
  template<typename T> struct C : A<C<T> > {};
#if __cplusplus >= 201103L
  template<typename T> struct D : decltype(A<D>()) {}; // expected-error {{requires template arguments}} expected-note {{declared}}
#endif
}

namespace dr433 { // dr433: yes
  template<class T> struct S {
    void f(union U*);
  };
  U *p;
  template<class T> void S<T>::f(union U*) {}

  S<int> s;
}

namespace dr434 { // dr434: yes
  void f() {
    const int ci = 0;
    int *pi = 0;
    const int *&rpci = pi; // expected-error {{cannot bind}}
    rpci = &ci;
    *pi = 1;
  }
}

// dr435: na

namespace dr436 { // dr436: yes
  enum E { f }; // expected-note {{previous}}
  void f(); // expected-error {{redefinition}}
}

namespace dr437 { // dr437: sup 1308
  // This is superseded by 1308, which is in turn superseded by 1330,
  // which restores this rule.
  template<typename U> struct T : U {};
  struct S {
    void f() throw(S);
#if __cplusplus > 201402L
    // expected-error@-2 {{ISO C++1z does not allow}} expected-note@-2 {{use 'noexcept}}
#endif
    void g() throw(T<S>);
#if __cplusplus > 201402L
    // expected-error@-2 {{ISO C++1z does not allow}} expected-note@-2 {{use 'noexcept}}
#endif
    struct U;
    void h() throw(U);
#if __cplusplus > 201402L
    // expected-error@-2 {{ISO C++1z does not allow}} expected-note@-2 {{use 'noexcept}}
#endif
    struct U {};
  };
}

// dr438 FIXME write a codegen test
// dr439 FIXME write a codegen test
// dr441 FIXME write a codegen test
// dr442: sup 348
// dr443: na

namespace dr444 { // dr444: yes
  struct D;
  struct B { // expected-note {{candidate is the implicit copy}} expected-note 0-1 {{implicit move}}
    D &operator=(D &) = delete; // expected-error 0-1{{extension}} expected-note {{deleted}}
  };
  struct D : B { // expected-note {{candidate is the implicit}} expected-note 0-1 {{implicit move}}
    using B::operator=;
  } extern d;
  void f() {
    d = d; // expected-error {{deleted}}
  }
}

namespace dr445 { // dr445: yes
  class A { void f(); }; // expected-note {{private}}
  struct B {
    friend void A::f(); // expected-error {{private}}
  };
}

namespace dr446 { // dr446: yes
  struct C;
  struct A {
    A();
    A(const A&) = delete; // expected-error 0-1{{extension}} expected-note +{{deleted}}
    A(const C&);
  };
  struct C : A {};
  void f(A a, bool b, C c) {
    void(b ? a : a);
    b ? A() : a; // expected-error {{deleted}}
    b ? a : A(); // expected-error {{deleted}}
    b ? A() : A();
#if __cplusplus <= 201402L
    // expected-error@-2 {{deleted}}
#endif

    void(b ? a : c);
    b ? a : C(); // expected-error {{deleted}}
    b ? c : A();
#if __cplusplus <= 201402L
    // expected-error@-2 {{deleted}}
#endif
    b ? A() : C();
#if __cplusplus <= 201402L
    // expected-error@-2 {{deleted}}
#endif
  }
}

namespace dr447 { // dr447: yes
  struct A { int n; int a[4]; };
  template<int> struct U {
    typedef int type;
    template<typename V> static void h();
  };
  template<typename T> U<sizeof(T)> g(T);
  template<typename T, int N> void f(int n) {
    // ok, not type dependent
    g(__builtin_offsetof(A, n)).h<int>();
    g(__builtin_offsetof(T, n)).h<int>();
    // value dependent if first argument is a dependent type
    U<__builtin_offsetof(A, n)>::type a;
    U<__builtin_offsetof(T, n)>::type b; // expected-error +{{}} expected-warning 0+{{}}
    // as an extension, we allow the member-designator to include array indices
    g(__builtin_offsetof(A, a[0])).h<int>(); // expected-error {{extension}}
    g(__builtin_offsetof(A, a[N])).h<int>(); // expected-error {{extension}}
    U<__builtin_offsetof(A, a[0])>::type c; // expected-error {{extension}}
    U<__builtin_offsetof(A, a[N])>::type d; // expected-error {{extension}} expected-error +{{}} expected-warning 0+{{}}
  }
}

namespace dr448 { // dr448: yes
  template<typename T = int> void f(int); // expected-error 0-1{{extension}} expected-note {{no known conversion}}
  template<typename T> void g(T t) {
    f<T>(t); // expected-error {{neither visible in the template definition nor found by argument-dependent lookup}}
    dr448::f(t); // expected-error {{no matching function}}
  }
  template<typename T> void f(T); // expected-note {{should be declared prior to the call site}}
  namespace HideFromADL { struct X {}; }
  template void g(int); // ok
  template void g(HideFromADL::X); // expected-note {{instantiation of}}
}

// dr449: na

namespace dr450 { // dr450: yes
  typedef int A[3];
  void f1(const A &);
  void f2(A &); // expected-note +{{not viable}}
  struct S { A n; };
  void g() {
    f1(S().n);
    f2(S().n); // expected-error {{no match}}}
  }
#if __cplusplus >= 201103L
  void h() {
    f1(A{});
    f2(A{}); // expected-error {{no match}}
  }
#endif
}

namespace dr451 { // dr451: yes
  const int a = 1 / 0; // expected-warning {{undefined}}
  const int b = 1 / 0; // expected-warning {{undefined}}
  int arr[b]; // expected-error +{{variable length arr}}
}

namespace dr452 { // dr452: yes
  struct A {
    int a, b, c;
    A *p;
    int f();
    A() : a(f()), b(this->f() + a), c(this->a), p(this) {}
  };
}

// dr454 FIXME write a codegen test

namespace dr456 { // dr456: yes
  // sup 903 c++11
  const int null = 0;
  void *p = null;
#if __cplusplus >= 201103L
  // expected-error@-2 {{cannot initialize}}
#else
  // expected-warning@-4 {{null}}
#endif

  const bool f = false;
  void *q = f;
#if __cplusplus >= 201103L
  // expected-error@-2 {{cannot initialize}}
#else
  // expected-warning@-4 {{null}}
#endif
}

namespace dr457 { // dr457: yes
  const int a = 1;
  const volatile int b = 1;
  int ax[a];
  int bx[b]; // expected-error +{{variable length array}}

  enum E {
    ea = a,
    eb = b // expected-error {{constant}} expected-note {{read of volatile-qualified}}
  };
}

namespace dr458 { // dr458: no
  struct A {
    int T;
    int f();
    template<typename> int g();
  };

  template<typename> struct B : A {
    int f();
    template<typename> int g();
    template<typename> int h();
  };

  int A::f() {
    return T;
  }
  template<typename T>
  int A::g() {
    return T; // FIXME: this is invalid, it finds the template parameter
  }

  template<typename T>
  int B<T>::f() {
    return T;
  }
  template<typename T> template<typename U>
  int B<T>::g() {
    return T;
  }
  template<typename U> template<typename T>
  int B<U>::h() {
    return T; // FIXME: this is invalid, it finds the template parameter
  }
}

namespace dr460 { // dr460: yes
  namespace X { namespace Q { int n; } }
  namespace Y {
    using X; // expected-error {{requires a qualified name}}
    using dr460::X; // expected-error {{cannot refer to a namespace}}
    using X::Q; // expected-error {{cannot refer to a namespace}}
  }
}

// dr461: na
// dr462 FIXME write a codegen test
// dr463: na
// dr464: na
// dr465: na

namespace dr466 { // dr466: no
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

    a->CI::~VI(); // FIXME: This is invalid; CI and VI are not the same scalar type.

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
}

namespace dr467 { // dr467: yes
  int stuff();

  int f() {
    static bool done;
    if (done)
      goto later;
    static int k = stuff();
    done = true;
  later:
    return k;
  }
  int g() {
    goto later; // expected-error {{cannot jump}}
    int k = stuff(); // expected-note {{bypasses variable initialization}}
  later:
    return k;
  }
}

namespace dr468 { // dr468: yes c++11
  // FIXME: Should we allow this in C++98 too?
  template<typename> struct A {
    template<typename> struct B {
      static int C;
    };
  };
  int k = dr468::template A<int>::template B<char>::C;
#if __cplusplus < 201103L
  // expected-error@-2 2{{'template' keyword outside of a template}}
#endif
}

namespace dr469 { // dr469: no
  // FIXME: The core issue here didn't really answer the question. We don't
  // deduce 'const T' from a function or reference type in a class template...
  template<typename T> struct X; // expected-note 2{{here}}
  template<typename T> struct X<const T> {};
  X<int&> x; // expected-error {{undefined}}
  X<int()> y; // expected-error {{undefined}}

  // ... but we do in a function template. GCC and EDG fail deduction of 'f'
  // and the second 'h'.
  template<typename T> void f(const T *);
  template<typename T> void g(T *, const T * = 0);
  template<typename T> void h(T *) { T::error; }
  template<typename T> void h(const T *);
  void i() {
    f(&i);
    g(&i);
    h(&i);
  }
}

namespace dr470 { // dr470: yes
  template<typename T> struct A {
    struct B {};
  };
  template<typename T> struct C {
  };

  template struct A<int>; // expected-note {{previous}}
  template struct A<int>::B; // expected-error {{duplicate explicit instantiation}}

  // ok, instantiating C<char> doesn't instantiate base class members.
  template struct A<char>;
  template struct C<char>;
}

namespace dr471 { // dr471: yes
  struct A { int n; };
  struct B : private virtual A {};
  struct C : protected virtual A {};
  struct D : B, C { int f() { return n; } };
  struct E : private virtual A {
    using A::n;
  };
  struct F : E, B { int f() { return n; } };
  struct G : virtual A {
  private:
    using A::n; // expected-note {{here}}
  };
  struct H : B, G { int f() { return n; } }; // expected-error {{private}}
}

namespace dr474 { // dr474: yes
  namespace N {
    struct S {
      void f();
    };
  }
  void N::S::f() {
    void g(); // expected-note {{previous}}
  }
  int g();
  namespace N {
    int g(); // expected-error {{cannot be overloaded}}
  }
}

// dr475 FIXME write a codegen test

namespace dr477 { // dr477: 3.5
  struct A {
    explicit A();
    virtual void f();
  };
  struct B {
    friend explicit A::A(); // expected-error {{'explicit' is invalid in friend declarations}}
    friend virtual void A::f(); // expected-error {{'virtual' is invalid in friend declarations}}
  };
  explicit A::A() {} // expected-error {{can only be specified inside the class definition}}
  virtual void A::f() {} // expected-error {{can only be specified inside the class definition}}
}

namespace dr478 { // dr478: yes
  struct A { virtual void f() = 0; }; // expected-note {{unimplemented}}
  void f(A *a);
  void f(A a[10]); // expected-error {{array of abstract class type}}
}

namespace dr479 { // dr479: yes
  struct S {
    S();
  private:
    S(const S&); // expected-note +{{here}}
    ~S(); // expected-note +{{here}}
  };
  void f() {
    throw S();
    // expected-error@-1 {{temporary of type 'dr479::S' has private destructor}}
    // expected-error@-2 {{exception object of type 'dr479::S' has private destructor}}
#if __cplusplus < 201103L
    // expected-error@-4 {{C++98 requires an accessible copy constructor}}
#endif
#if __cplusplus <= 201402L
    // expected-error@-7 {{calling a private constructor}} (copy ctor)
#endif
  }
  void g() {
    S s; // expected-error {{private destructor}}}
    throw s;
    // expected-error@-1 {{calling a private constructor}}
    // expected-error@-2 {{exception object of type 'dr479::S' has private destructor}}
  }
  void h() {
    try {
      f();
      g();
    } catch (S s) {
      // expected-error@-1 {{calling a private constructor}}
      // expected-error@-2 {{variable of type 'dr479::S' has private destructor}}
    }
  }
}

namespace dr480 { // dr480: yes
  struct A { int n; };
  struct B : A {};
  struct C : virtual B {};
  struct D : C {};

  int A::*a = &A::n;
  int D::*b = a; // expected-error {{virtual base}}

  extern int D::*c;
  int A::*d = static_cast<int A::*>(c); // expected-error {{virtual base}}

  D *e;
  A *f = e;
  D *g = static_cast<D*>(f); // expected-error {{virtual base}}

  extern D &i;
  A &j = i;
  D &k = static_cast<D&>(j); // expected-error {{virtual base}}
}

namespace dr481 { // dr481: yes
  template<class T, T U> class A { T *x; };
  T *x; // expected-error {{unknown type}}

  template<class T *U> class B { T *x; };
  T *y; // ok

  struct C {
    template<class T> void f(class D *p);
  };
  D *z; // ok

  template<typename A = C, typename C = A> struct E {
    void f() {
      typedef ::dr481::C c; // expected-note {{previous}}
      typedef C c; // expected-error {{different type}}
    }
  };
  template struct E<>; // ok
  template struct E<int>; // expected-note {{instantiation of}}

  template<template<typename U_no_typo_correction> class A,
           A<int> *B,
           U_no_typo_correction *C> // expected-error {{unknown type}}
  struct F {
    U_no_typo_correction *x; // expected-error {{unknown type}}
  };

  template<template<class H *> class> struct G {
    H *x;
  };
  H *q;

  typedef int N;
  template<N X, typename N, template<N Y> class T> struct I;
  template<char*> struct J;
  I<123, char*, J> *j;
}

namespace dr482 { // dr482: 3.5
  extern int a;
  void f();

  int dr482::a = 0; // expected-warning {{extra qualification}}
  void dr482::f() {} // expected-warning {{extra qualification}}

  inline namespace X { // expected-error 0-1{{C++11 feature}}
    extern int b;
    void g();
    struct S;
  }
  int dr482::b = 0; // expected-warning {{extra qualification}}
  void dr482::g() {} // expected-warning {{extra qualification}}
  struct dr482::S {}; // expected-warning {{extra qualification}}

  void dr482::f(); // expected-warning {{extra qualification}}
  void dr482::g(); // expected-warning {{extra qualification}}

  // FIXME: The following are valid in DR482's wording, but these are bugs in
  // the wording which we deliberately don't implement.
  namespace N { typedef int type; }
  typedef int N::type; // expected-error {{typedef declarator cannot be qualified}}
  struct A {
    struct B;
    struct A::B {}; // expected-error {{extra qualification}}

#if __cplusplus >= 201103L
    enum class C;
    enum class A::C {}; // expected-error {{extra qualification}}
#endif
  };
}

namespace dr483 { // dr483: yes
  namespace climits {
    int check1[__SCHAR_MAX__ >= 127 ? 1 : -1];
    int check2[__SHRT_MAX__ >= 32767 ? 1 : -1];
    int check3[__INT_MAX__ >= 32767 ? 1 : -1];
    int check4[__LONG_MAX__ >= 2147483647 ? 1 : -1];
    int check5[__LONG_LONG_MAX__ >= 9223372036854775807 ? 1 : -1];
#if __cplusplus < 201103L
    // expected-error@-2 {{extension}}
#endif
  }
  namespace cstdint {
    int check1[__PTRDIFF_WIDTH__ >= 16 ? 1 : -1];
    int check2[__SIG_ATOMIC_WIDTH__ >= 8 ? 1 : -1];
    int check3[__SIZE_WIDTH__ >= 16 ? 1 : -1];
    int check4[__WCHAR_WIDTH__ >= 8 ? 1 : -1];
    int check5[__WINT_WIDTH__ >= 16 ? 1 : -1];
  }
}

namespace dr484 { // dr484: yes
  struct A {
    A();
    void f();
  };
  typedef const A CA;
  void CA::f() {
    this->~CA();
    this->CA::~A();
    this->CA::A::~A();
  }
  CA::A() {}

  struct B : CA {
    B() : CA() {}
    void f() { return CA::f(); }
  };

  struct C;
  typedef C CT; // expected-note {{here}}
  struct CT {}; // expected-error {{conflicts with typedef}}

  namespace N {
    struct D;
    typedef D DT; // expected-note {{here}}
  }
  struct N::DT {}; // expected-error {{conflicts with typedef}}

  typedef struct {
    S(); // expected-error {{requires a type}}
  } S;
}

namespace dr485 { // dr485: yes
  namespace N {
    struct S {};
    int operator+(S, S);
    template<typename T> int f(S);
  }
  template<typename T> int f();

  N::S s;
  int a = operator+(s, s);
  int b = f<int>(s);
}

namespace dr486 { // dr486: yes
  template<typename T> T f(T *); // expected-note 2{{substitution failure}}
  int &f(...);

  void g();
  int n[10];

  void h() {
    int &a = f(&g);
    int &b = f(&n);
    f<void()>(&g); // expected-error {{no match}}
    f<int[10]>(&n); // expected-error {{no match}}
  }
}

namespace dr487 { // dr487: yes
  enum E { e };
  int operator+(int, E);
  int i[4 + e]; // expected-error 2{{variable length array}}
}

namespace dr488 { // dr488: yes c++11
  template <typename T> void f(T);
  void f(int);
  void g() {
    // FIXME: It seems CWG thought this should be a SFINAE failure prior to
    // allowing local types as template arguments. In C++98, we should either
    // allow local types as template arguments or treat this as a SFINAE
    // failure.
    enum E { e };
    f(e);
#if __cplusplus < 201103L
    // expected-error@-2 {{local type}}
#endif
  }
}

// dr489: na

namespace dr490 { // dr490: yes
  template<typename T> struct X {};

  struct A {
    typedef int T;
    struct K {}; // expected-note {{declared}}

    int f(T);
    int g(T);
    int h(X<T>);
    int X<T>::*i(); // expected-note {{previous}}
    int K::*j();

    template<typename T> T k();

    operator X<T>();
  };

  struct B {
    typedef char T;
    typedef int U;
    friend int A::f(T);
    friend int A::g(U);
    friend int A::h(X<T>);

    // FIXME: Per this DR, these two are valid! That is another defect
    // (no number yet...) which will eventually supersede this one.
    friend int X<T>::*A::i(); // expected-error {{return type}}
    friend int K::*A::j(); // expected-error {{undeclared identifier 'K'; did you mean 'A::K'?}}

    // ok, lookup finds B::T, not A::T, so return type matches
    friend char A::k<T>();
    friend int A::k<U>();

    // A conversion-type-id in a conversion-function-id is always looked up in
    // the class of the conversion function first.
    friend A::operator X<T>();
  };
}

namespace dr491 { // dr491: dup 413
  struct A {} a, b[3] = { a, {} };
  A c[2] = { a, {}, b[1] }; // expected-error {{excess elements}}
}

// dr492 FIXME write a codegen test

namespace dr493 { // dr493: dup 976
  struct X {
    template <class T> operator const T &() const;
  };
  void f() {
    if (X()) {
    }
  }
}

namespace dr494 { // dr494: dup 372
  class A {
    class B {};
    friend class C;
  };
  class C : A::B {
    A::B x;
    class D : A::B {
      A::B y;
    };
  };
}

namespace dr495 { // dr495: 3.5
  template<typename T>
  struct S {
    operator int() { return T::error; }
    template<typename U> operator U();
  };
  S<int> s;
  long n = s;

  template<typename T>
  struct S2 {
    template<typename U> operator U();
    operator int() { return T::error; }
  };
  S2<int> s2;
  long n2 = s2;
}

namespace dr496 { // dr496: no
  struct A { int n; };
  struct B { volatile int n; };
  int check1[ __is_trivially_copyable(const int) ? 1 : -1];
  int check2[!__is_trivially_copyable(volatile int) ? 1 : -1];
  int check3[ __is_trivially_constructible(A, const A&) ? 1 : -1];
  // FIXME: This is wrong.
  int check4[ __is_trivially_constructible(B, const B&) ? 1 : -1];
  int check5[ __is_trivially_assignable(A, const A&) ? 1 : -1];
  // FIXME: This is wrong.
  int check6[ __is_trivially_assignable(B, const B&) ? 1 : -1];
}

namespace dr497 { // dr497: sup 253
  void before() {
    struct S {
      mutable int i;
    };
    const S cs;
    int S::*pm = &S::i;
    cs.*pm = 88; // expected-error {{not assignable}}
  }

  void after() {
    struct S {
      S() : i(0) {}
      mutable int i;
    };
    const S cs;
    int S::*pm = &S::i;
    cs.*pm = 88; // expected-error {{not assignable}}
  }
}

namespace dr499 { // dr499: yes
  extern char str[];
  void f() { throw str; }
}
