// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++98 %s -verify=expected,cxx98-14,cxx98-17,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++11 %s -verify=expected,cxx98-14,cxx98-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++14 %s -verify=expected,cxx98-14,cxx98-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++17 %s -verify=expected,since-cxx17,cxx98-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++20 %s -verify=expected,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: env ASAN_OPTIONS=detect_stack_use_after_return=0 %clang_cc1 -std=c++23 %s -verify=expected,since-cxx17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

// FIXME: __SIZE_TYPE__ expands to 'long long' on some targets.
__extension__ typedef __SIZE_TYPE__ size_t;

namespace std { struct type_info; }

namespace dr400 { // dr400: yes
  struct A { int a; struct a {}; }; // #dr400-A
  struct B { int a; struct a {}; }; // #dr400-B
  struct C : A, B { using A::a; struct a b; };
  struct D : A, B { 
    using A::a;
    // FIXME: we should issue a single diagnostic
    using B::a; // #dr400-using-B-a
    // expected-error@#dr400-using-B-a {{target of using declaration conflicts with declaration already in scope}}
    //   expected-note@#dr400-B {{target of using declaration}}
    //   expected-note@#dr400-A {{conflicting declaration}}
    // expected-error@#dr400-using-B-a {{target of using declaration conflicts with declaration already in scope}}
    //   expected-note@#dr400-B {{target of using declaration}}
    //   expected-note@#dr400-A {{conflicting declaration}}
    struct a b;
  };
  struct E : A, B { struct a b; };
  // expected-error@-1 {{member 'a' found in multiple base classes of different types}}
  // expected-note@#dr400-A {{member type 'dr400::A::a' found by ambiguous name lookup}}
  // expected-note@#dr400-B {{member type 'dr400::B::a' found by ambiguous name lookup}}
}

namespace dr401 { // dr401: 2.8
  template<class T, class U = typename T::type> class A : public T {}; // #dr401-A
  // expected-error@#dr401-A {{'type' is a private member of 'dr401::C'}}
  //   expected-note@#dr402-friend-A-C {{in instantiation of default argument for 'A<C>' required here}}
  //   expected-note@#dr402-C-type {{implicitly declared private here}}
  // expected-error@#dr401-A {{'type' is a protected member of 'dr401::B'}}
  //   expected-note@#dr402-b {{in instantiation of default argument for 'A<B>' required here}}
  //   expected-note@#dr402-B-type {{declared protected here}}
  // expected-error@#dr401-A {{'type' is a private member of 'dr401::D'}}
  //   expected-note@#dr402-d {{in instantiation of default argument for 'A<D>' required here}}
  //   expected-note@#dr402-D-type {{implicitly declared private here}}
  class B {
  protected:
    typedef int type; // #dr402-B-type
  };

  class C {
    typedef int type; // #dr402-C-type
    friend class A<C>; // #dr402-friend-A-C
  };

  class D {
    typedef int type; // #dr402-D-type
    friend class A<D, int>;
  };

  A<B> *b; // #dr402-b
  A<D> *d; // #dr402-d

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
  template<class T, class U = typename T::type> void f(T) {} // #dr402-f
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  // cxx98-error@-2 {{'type' is a protected member of 'dr401::B'}}
  //   cxx98-note@-3 {{in instantiation of default argument for 'f<B>' required here}}
  //   cxx98-note@#dr402-f-b {{while substituting deduced template arguments into function template 'f' [with T = B, U = (no value)]}}
  //   cxx98-note@#dr402-B-type {{declared protected here}}
  void g(B b) { f(b); } // #dr402-f-b
  // since-cxx11-error@-1 {{no matching function for call to 'f'}}
  // since-cxx11-note@#dr402-f {{candidate template ignored: substitution failure [with T = B, U = typename B::type]: 'type' is a protected member of 'dr401::B'}}
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

namespace dr405 { // dr405: yes
                  // NB: also dup 218
  namespace A {
    struct S {};
    void f(S);
  }
  namespace B {
    struct S {};
    void f(S);
  }

  struct C {
    int f;
    void test1(A::S as) { f(as); }
    // expected-error@-1 {{called object type 'int' is not a function or function pointer}}
    void test2(A::S as) { void f(); f(as); }
    // expected-error@-1 {{too many arguments to function call, expected 0, have 1}}
    // expected-note@-2 {{'f' declared here}}
    void test3(A::S as) { using A::f; f(as); } // ok
    void test4(A::S as) { using B::f; f(as); } // ok
    void test5(A::S as) { int f; f(as); }
    // expected-error@-1 {{called object type 'int' is not a function or function pointer}}
    void test6(A::S as) { struct f {}; (void) f(as); }
    // expected-error@-1 {{no matching conversion for functional-style cast from 'A::S' to 'f'}}
    // expected-note@-2 {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'A::S' to 'const f' for 1st argument}}
    // since-cxx11-note@-3 {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'A::S' to 'f' for 1st argument}}
    // expected-note@-4 {{candidate constructor (the implicit default constructor) not viable: requires 0 arguments, but 1 was provided}}
  };

  namespace D {
    struct S {};
    struct X { void operator()(S); } f;
  }
  void testD(D::S ds) { f(ds); }
  // expected-error@-1 {{use of undeclared identifier 'f'}}

  namespace E {
    struct S {};
    struct f { f(S); };
  }
  void testE(E::S es) { f(es); }
  // expected-error@-1 {{use of undeclared identifier 'f'}}
}

namespace dr406 { // dr406: 2.9
  typedef struct {
    static int n;
    // expected-error@-1 {{static data member 'n' not allowed in anonymous struct}}
  } A;
  typedef union {
    static int n;
    // expected-error@-1 {{static data member 'n' not allowed in anonymous union}}
  } B;
}

namespace dr407 { // dr407: 3.8
                  // NB: reused by dr1894 and dr2199
  struct S;
  typedef struct S S;
  void f() {
    struct S *p;
    {
      typedef struct S S; // #dr407-typedef-S
      struct S *p;
      // expected-error@-1 {{typedef 'S' cannot be referenced with a struct specifier}}
      // expected-note@#dr407-typedef-S {{declared here}}
    }
  }
  struct S {};

  namespace UsingDir {
    namespace A {
      struct S {}; // #dr407-A-S
    }
    namespace B {
      typedef int S; // #dr407-B-S
    }
    namespace C {
      using namespace A;
      using namespace B;
      struct S s;
      // expected-error@-1 {{ambiguous}}
      // expected-note@#dr407-A-S {{candidate found by name lookup is 'dr407::UsingDir::A::S'}}
      // expected-note@#dr407-B-S {{candidate found by name lookup is 'dr407::UsingDir::B::S'}}
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
    A<T*>::B b4;
    // cxx98-17-error@-1 {{missing 'typename' prior to dependent type name A<T *>::B; implicit 'typename' is a C++20 extension}}
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
      static void z(); // #dr410-z
    };

    template<> void h(int) { A::z(); }
    // FIXME: This should be ill-formed. The member A::i<> is befriended,
    // not this function.
    template<> void i(int) { A::z(); }
  }
  template<> void f(int) { M::A::z(); }
  void g(int) { M::A::z(); }
  // expected-error@-1 {{'z' is a private member of 'dr410::M::A'}}
  // expected-note@#dr410-z {{declared private here}}
}

// dr412 is in its own file.

namespace dr413 { // dr413: yes
  struct S {
    int a;
    int : 17;
    int b;
  };
  S s = { 1, 2, 3 };
  // expected-error@-1 {{excess elements in struct initializer}}

  struct E {};
  struct T { // #dr413-T
    int a;
    E e;
    int b;
  };
  T t1 = { 1, {}, 2 };
  T t2 = { 1, 2 };
  // expected-error@-1 {{initializer for aggregate with no elements requires explicit braces}}
  // expected-note@#dr413-T {{'dr413::T' declared here}}
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
  struct dr417::A {};
  // expected-warning@-1 {{extra qualification on member 'A'}}
  struct B { struct X; };
  struct C : B {};
  struct C::X {};
  // expected-error@-1 {{no struct named 'X' in 'dr417::C'}}
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
  struct dr417::D {};
  // expected-warning@-1 {{extra qualification on member 'D'}}
  using namespace N;
  struct dr417::E {};
  // expected-error@-1 {{no struct named 'E' in namespace 'dr417'}}
  // expected-warning@-2 {{extra qualification on member 'E'}}
  struct N::F {};
  struct G;
  using N::H;
  namespace M {
    struct dr417::G {};
    // expected-error@-1 {{cannot define or redeclare 'G' here because namespace 'M' does not enclose namespace 'dr417'}}
    struct dr417::H {};
    // expected-error@-1 {{cannot define or redeclare 'H' here because namespace 'M' does not enclose namespace 'dr417'}}
  }
}

namespace dr418 { // dr418: no
namespace example1 {
void f1(int, int = 0);
void f1(int = 0, int);

void g() { f1(); }
} // namespace example1

namespace example2 {
namespace A {
void f2(int); // #dr418-f2
}
namespace B {
using A::f2;
}
namespace A {
void f2(int = 3);
}
void g2() {
  using B::f2;
  f2();
  // expected-error@-1 {{no matching function for call to 'f2'}}
  // expected-note@#dr418-f2 {{candidate function not viable: requires 1 argument, but 0 were provided}}
}
} // namespace example2

// example from [over.match.best]/4
namespace example3 {
namespace A {
extern "C" void f(int = 5);
}
namespace B {
extern "C" void f(int = 5);
}

using A::f;
using B::f;

void use() {
  f(3);
  f(); // FIXME: this should fail
}
} // namespace example3
} // namespace dr418

namespace dr420 { // dr420: 9
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
    p->template ~Y<int>();
    // expected-error@-1 {{'template' keyword not permitted in destructor name}}
  }
  template<typename T> struct Y {};
  template void test2(Y<int>*);
  template void test2(ptr<Y<int> >);

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
    p->template id<int>::~id<int>(); // OK since dr2292
    q->template id<int>::~id<int>(); // OK since dr2292
    p->A::template id<int>::~id<int>();
    q->A::template id<int>::~id<int>();
  }
#endif
}

namespace dr421 { // dr421: yes
  struct X { X(); int n; int &r; };
  int *p = &X().n;
  // cxx98-error@-1 {{taking the address of a temporary object of type 'int'}}
  // since-cxx11-error@-2 {{cannot take the address of an rvalue of type 'int'}}
  int *q = &X().r;
}

namespace dr422 { // dr422: yes
  template<typename T, typename U> void f() {
    typedef T type; // #dr422-typedef-T
    typedef U type;
    // expected-error@-1 {{typedef redefinition with different types ('char' vs 'int')}}
    // expected-note@#dr422-f-int-char {{in instantiation of function template specialization 'dr422::f<int, char>' requested here}}
    // expected-note@#dr422-typedef-T {{previous definition is here}}
  }
  template void f<int, int>();
  template void f<int, char>(); // #dr422-f-int-char
}

namespace dr423 { // dr423: yes
  template<typename T> struct X { operator T&(); };
  void f(X<int> x) { x += 1; }
}

namespace dr424 { // dr424: yes
  struct A {
    typedef int N; // #dr424-N
    typedef int N;
    // expected-error@-1 {{redefinition of 'N'}}
    // expected-note@#dr424-N {{previous definition is here}}

    struct X;
    typedef X X; // #dr424-X
    struct X {};

    struct X *p;
    struct A::X *q;
    X *r;

    typedef X X;
    // expected-error@-1 {{redefinition of 'X'}}
    // expected-note@#dr424-X {{previous definition is here}}
  };
  struct B {
    typedef int M;
  };
  struct C : B {
    typedef int M; // #dr424-M
    typedef int M;
    // expected-error@-1 {{redefinition of 'M'}}
    // expected-note@#dr424-M {{previous definition is here}}
  };
}

namespace dr425 { // dr425: yes
  struct A { template<typename T> operator T() const; } a;
  float f = 1.0f * a;
  // expected-error@-1 {{use of overloaded operator '*' is ambiguous (with operand types 'float' and 'struct A')}}
  // expected-note@-2 +{{built-in candidate}}

  template<typename T> struct is_float;
  template<> struct is_float<float> { typedef void type; };

  struct B {
    template<typename T, typename U = typename is_float<T>::type> operator T() const;
    // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  } b;
  float g = 1.0f * b; // ok
}

namespace dr427 { // dr427: yes
  struct B {};
  struct D : public B {
    D(B &) = delete; // #dr427-D
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  };

  extern D d1;
  B &b = d1;
  const D &d2 = static_cast<const D&>(b);
  const D &d3 = (const D&)b;
  const D &d4(b);
  // expected-error@-1 {{conversion function from 'B' to 'const D' invokes a deleted function}}
  // expected-note@#dr427-D {{'D' has been explicitly marked deleted here}}
}

namespace dr428 { // dr428: yes
  template<typename T> T make();
  extern struct X x; // #dr428-X
  void f() {
    throw void();
    // expected-error@-1 {{cannot throw}}
    throw make<void*>();
    throw make<const volatile void*>();
    throw x;
    // expected-error@-1 {{cannot throw}}
    // expected-note@#dr428-X {{forward declaration of 'dr428::X'}}
    throw make<X&>();
    // expected-error@-1 {{cannot throw}}
    // expected-note@#dr428-X {{forward declaration of 'dr428::X'}}
    throw make<X*>();
    // expected-error@-1 {{cannot throw}}
    // expected-note@#dr428-X {{forward declaration of 'dr428::X'}}
    throw make<const volatile X&>();
    // expected-error@-1 {{cannot throw}}
    // expected-note@#dr428-X {{forward declaration of 'dr428::X'}}
    throw make<const volatile X*>();
    // expected-error@-1 {{cannot throw}}
    // expected-note@#dr428-X {{forward declaration of 'dr428::X'}}
  }
}

namespace dr429 { // dr429: 2.8 c++11
  // FIXME: This rule is obviously intended to apply to C++98 as well.
  struct A {
    static void *operator new(size_t, size_t);
    static void operator delete(void*, size_t); // #dr429-delete
  } *a = new (0) A;
  // since-cxx11-error@-1 {{'new' expression with placement arguments refers to non-placement 'operator delete'}}
  // since-cxx11-note@#dr429-delete {{here}}
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
    // cxx98-warning@-1 {{multiple unsequenced modifications to 'n'}}
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
        ->get<T>();
        // expected-error@-1 {{use 'template' keyword to treat 'get' as a dependent template name}}
    a.get<T>()->template get<T>();
    a.A::get<T>();
    A::B<int> *b = a.get<A::B<int> >();
    b->get<int>();
    b->A::B<int>::get<int>();
    b->A::B<int>::get<T>();
    b->A::B<T>::get<int>();
    // expected-error@-1 {{use 'template' keyword to treat 'get' as a dependent template name}}
    b->A::B<T>::template get<int>();
    b->A::B<T>::get<T>();
    // expected-error@-1 {{use 'template' keyword to treat 'get' as a dependent template name}}
    b->A::B<T>::template get<T>();
    A::B<T> *c = a.get<A::B<T> >();
    c->get<int>();
    // expected-error@-1 {{use 'template' keyword to treat 'get' as a dependent template name}}
    c->template get<int>();
  }
}

namespace dr432 { // dr432: 3.0
  template<typename T> struct A {};
  template<typename T> struct B : A<B> {};
  // expected-error@-1 {{use of class template 'B' requires template arguments}}
  // expected-note@-2 {{template is declared here}}
  template<typename T> struct C : A<C<T> > {};
#if __cplusplus >= 201103L
  template<typename T> struct D : decltype(A<D>()) {};
  // since-cxx11-error@-1 {{use of class template 'D' requires template arguments}}
  // since-cxx11-note@-2 {{template is declared here}}
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

namespace dr434 { // dr434: sup 2352
  void f() {
    const int ci = 0;
    int *pi = 0;
    const int *&rpci = pi;
    // expected-error@-1 {{binding reference of type 'const int *' to value of type 'int *' not permitted due to incompatible qualifiers}}
    const int * const &rcpci = pi; // OK
    rpci = &ci;
    *pi = 1;
  }

#if __cplusplus >= 201103L
  int *pi = 0;
  const int * const &rcpci = pi;
  static_assert(&rcpci == &pi, "");
#endif
}

// dr435: na

namespace dr436 { // dr436: yes
  enum E { f }; // #dr436-f
  void f();
  // expected-error@-1 {{redefinition of 'f' as different kind of symbol}}
  // expected-note@#dr436-f {{previous definition is here}}
}

namespace dr437 { // dr437: sup 1308
  // This is superseded by 1308, which is in turn superseded by 1330,
  // which restores this rule.
  template<typename U> struct T : U {};
  struct S {
    void f() throw(S);
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
    void g() throw(T<S>);
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
    struct U;
    void h() throw(U);
    // since-cxx17-error@-1 {{ISO C++17 does not allow dynamic exception specifications}}
    // since-cxx17-note@-2 {{use 'noexcept(false)' instead}}
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
  struct B { // #dr444-B
    D &operator=(D &) = delete; // #dr444-deleted
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
  };
  struct D : B { // #dr444-D
    using B::operator=;
  } extern d;
  void f() {
    d = d;
    // expected-error@-1 {{overload resolution selected deleted operator '='}}
    // expected-note@#dr444-deleted {{candidate function has been explicitly deleted}}
    // expected-note@#dr444-D {{candidate function (the implicit copy assignment operator)}}
    // expected-note@#dr444-B {{candidate function (the implicit copy assignment operator)}}
    // since-cxx11-note@#dr444-B {{candidate function (the implicit move assignment operator) not viable: expects an rvalue for 1st argument}}
    // since-cxx11-note@#dr444-D {{candidate function (the implicit move assignment operator) not viable: expects an rvalue for 1st argument}}
  }
}

namespace dr445 { // dr445: 3.2
  class A { void f(); }; // #dr445-f
  struct B {
    friend void A::f();
    // expected-error@-1 {{friend function 'f' is a private member of 'dr445::A'}}
    // expected-note@#dr445-f {{implicitly declared private here}}
  };
}

namespace dr446 { // dr446: 2.8
  struct C;
  struct A {
    A();
    A(const A&) = delete; // #dr446-deleted
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    A(const C&);
  };
  struct C : A {};
  void f(A a, bool b, C c) {
    void(b ? a : a);
    b ? A() : a;
    // expected-error@-1 {{call to deleted constructor of 'A'}}
    // expected-note@#dr446-deleted {{'A' has been explicitly marked deleted here}}
    b ? a : A();
    // expected-error@-1 {{call to deleted constructor of 'A'}}
    // expected-note@#dr446-deleted {{'A' has been explicitly marked deleted here}}
    b ? A() : A();
    // cxx98-14-error@-1 {{call to deleted constructor of 'A'}}
    // expected-note@#dr446-deleted {{'A' has been explicitly marked deleted here}}

    void(b ? a : c);
    b ? a : C();
    // expected-error@-1 {{call to deleted constructor of 'A'}}
    // cxx98-14-note@#dr446-deleted {{'A' has been explicitly marked deleted here}}
    b ? c : A();
    // cxx98-14-error@-1 {{call to deleted constructor of 'A'}}
    // cxx98-14-note@#dr446-deleted {{'A' has been explicitly marked deleted here}}
    b ? A() : C();
    // cxx98-14-error@-1 {{call to deleted constructor of 'A'}}
    // cxx98-14-note@#dr446-deleted {{'A' has been explicitly marked deleted here}}
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
    // FIXME: instead of ";", we should be suggesting "typename"
    U<__builtin_offsetof(T, n)>::type b;
    // expected-error@-1 {{expected ';' after expression}}
    // expected-error@-2 {{use of undeclared identifier 'b'}}
    // as an extension, we allow the member-designator to include array indices
    g(__builtin_offsetof(A, a[0])).h<int>();
    g(__builtin_offsetof(A, a[N])).h<int>();
    U<__builtin_offsetof(A, a[0])>::type c;
    // FIXME: instead of ";", we should be suggesting "typename"
    U<__builtin_offsetof(A, a[N])>::type d;
    // expected-error@-1 {{expected ';' after expression}}
    // expected-error@-2 {{use of undeclared identifier 'd'}}
  }
}

namespace dr448 { // dr448: 2.8
  template<typename T = int> void f(int); // #dr448-f-int
  // cxx98-error@-1 {{default template arguments for a function template are a C++11 extension}}
  template<typename T> void g(T t) {
    f<T>(t);
    // expected-error@-1 {{call to function 'f' that is neither visible in the template definition nor found by argument-dependent lookup}}
    // expected-note@#dr448-g {{in instantiation of function template specialization 'dr448::g<dr448::HideFromADL::X>' requested here}}
    // expected-note@#dr448-f-T {{'f' should be declared prior to the call site or in namespace 'dr448::HideFromADL'}}
    dr448::f(t);
    // expected-error@-1 {{no matching function for call to 'f'}}
    // expected-note@#dr448-f-int {{candidate function template not viable: no known conversion from 'dr448::HideFromADL::X' to 'int' for 1st argument}}
  }
  template<typename T> void f(T); // #dr448-f-T
  namespace HideFromADL { struct X {}; }
  template void g(int); // ok
  template void g(HideFromADL::X); // #dr448-g
}

// dr449: na

namespace dr450 { // dr450: yes
  typedef int A[3];
  void f1(const A &);
  void f2(A &); // #dr450-f2
  struct S { A n; };
  void g() {
    f1(S().n);
    f2(S().n);
    // expected-error@-1 {{no matching function for call to 'f2'}}}
    // expected-note@#dr450-f2 {{candidate function not viable: expects an lvalue for 1st argument}}
  }
#if __cplusplus >= 201103L
  void h() {
    f1(A{});
    f2(A{});
    // expected-error@-1 {{no matching function for call to 'f2'}}}
    // expected-note@#dr450-f2 {{candidate function not viable: expects an lvalue for 1st argument}}
  }
#endif
}

namespace dr451 { // dr451: yes
  const int a = 1 / 0;
  // expected-warning@-1 {{division by zero is undefined}}
  const int b = 1 / 0; // #dr451-b
  // expected-warning@-1 {{division by zero is undefined}}
  int arr[b]; // #dr451-arr
  // expected-error@-1 {{variable length arrays in C++ are a Clang extension}}
  //   expected-note@-2 {{initializer of 'b' is not a constant expression}}
  //   expected-note@#dr451-b {{declared here}}
  // expected-error@#dr451-arr {{variable length array declaration not allowed at file scope}}
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
  // cxx98-warning@-1 {{expression which evaluates to zero treated as a null pointer constant of type 'void *'}}
  // since-cxx11-error@-2 {{cannot initialize a variable of type 'void *' with an lvalue of type 'const int'}}

  const bool f = false;
  void *q = f;
  // cxx98-warning@-1 {{initialization of pointer of type 'void *' to null from a constant boolean}}
  // since-cxx11-error@-2 {{cannot initialize a variable of type 'void *' with an lvalue of type 'const bool'}}
}

namespace dr457 { // dr457: yes
  const int a = 1;
  const volatile int b = 1;
  int ax[a];
  int bx[b];
  // expected-error@-1 {{variable length arrays in C++ are a Clang extension}}
  //   expected-note@-2 {{read of volatile-qualified type 'const volatile int' is not allowed in a constant expression}}
  // expected-error@-3 {{variable length array declaration not allowed at file scope}}

  enum E {
    ea = a,
    eb = b
    // expected-error@-1 {{expression is not an integral constant expression}}
    // expected-note@-2 {{read of volatile-qualified type 'const volatile int' is not allowed in a constant expression}}
  };
}

namespace dr458 { // dr458: 11
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
  template<typename T> // #dr458-g-T
  int A::g() {
    return T;
    // expected-error@-1 {{'T' does not refer to a value}}
    // expected-note@#dr458-g-T {{declared here}}
  }

  template<typename T>
  int B<T>::f() {
    return T;
  }
  template<typename T>
  template<typename U>
  int B<T>::g() {
    return T;
  }
  template<typename U>
  template<typename T> // #dr458-h-T
  int B<U>::h() {
    return T;
    // expected-error@-1 {{'T' does not refer to a value}}
    // expected-note@#dr458-h-T {{declared here}}
  }
}

namespace dr460 { // dr460: yes
  namespace X { namespace Q { int n; } }
  namespace Y {
    using X;
    // expected-error@-1 {{using declaration requires a qualified name}}
    using dr460::X;
    // expected-error@-1 {{using declaration cannot refer to a namespace}}
    using X::Q;
    // expected-error@-1 {{using declaration cannot refer to a namespace}}
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
    goto later;
    // expected-error@-1 {{cannot jump from this goto statement to its label}}
    // expected-note@#dr467-k {{jump bypasses variable initialization}}
    int k = stuff(); // #dr467-k
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
  // cxx98-error@-1 {{'template' keyword outside of a template}}
  // cxx98-error@-2 {{'template' keyword outside of a template}}
}

namespace dr469 { // dr469: no
  template<typename T> struct X; // #dr469-X
  template<typename T> struct X<const T> {};
  X<int&> x;
  // expected-error@-1 {{implicit instantiation of undefined template 'dr469::X<int &>'}}
  // expected-note@#dr469-X {{template is declared here}}
}

namespace dr470 { // dr470: yes
  template<typename T> struct A {
    struct B {};
  };
  template<typename T> struct C {
  };

  template struct A<int>; // #dr470-A-int
  template struct A<int>::B;
  // expected-error@-1 {{duplicate explicit instantiation of 'B'}}
  // expected-note@#dr470-A-int {{previous explicit instantiation is here}}

  // ok, instantiating C<char> doesn't instantiate base class members.
  template struct A<char>;
  template struct C<char>;
}

namespace dr471 { // dr471: 2.8
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
    using A::n; // #dr471-G-using
  };
  struct H : B, G { int f() { return n; } };
  // expected-error@-1 {{'n' is a private member of 'dr471::G'}}
  // expected-note@#dr471-G-using {{declared private here}}
}

namespace dr474 { // dr474: 3.4
  namespace N {
    struct S {
      void f();
    };
  }
  void N::S::f() {
    void g(); // #dr474-g
  }
  int g();
  namespace N {
    int g();
    // expected-error@-1 {{functions that differ only in their return type cannot be overloaded}}
    // expected-note@#dr474-g {{previous declaration is here}}
  }
}

// dr475 FIXME write a codegen test

namespace dr477 { // dr477: 3.5
  struct A {
    explicit A();
    virtual void f();
  };
  struct B {
    friend explicit A::A();
    // expected-error@-1 {{'explicit' is invalid in friend declarations}}
    friend virtual void A::f();
    // expected-error@-1 {{'virtual' is invalid in friend declarations}}
  };
  explicit A::A() {}
  // expected-error@-1 {{can only be specified inside the class definition}}
  virtual void A::f() {}
  // expected-error@-1 {{can only be specified inside the class definition}}
}

namespace dr478 { // dr478: yes
  struct A { virtual void f() = 0; }; // #dr478-f
  void f(A *a);
  void f(A a[10]);
  // expected-error@-1 {{array of abstract class type 'A'}}
  // expected-note@#dr478-f {{unimplemented pure virtual method 'f' in 'A'}}
}

namespace dr479 { // dr479: 2.8
  struct S {
    S();
  private:
    S(const S&); // #dr479-S-copy-ctor
    ~S(); // #dr479-S-dtor
  };
  void f() {
    throw S();
    // expected-error@-1 {{temporary of type 'S' has private destructor}}
    //   expected-note@#dr479-S-dtor {{declared private here}}
    // expected-error@-3 {{exception object of type 'S' has private destructor}}
    //   expected-note@#dr479-S-dtor {{declared private here}}
    // cxx98-error@-5 {{C++98 requires an accessible copy constructor for class 'dr479::S' when binding a reference to a temporary; was private}}
    //   cxx98-note@#dr479-S-copy-ctor {{declared private here}}
    // cxx98-14-error@-7 {{calling a private constructor of class 'dr479::S'}}
    //   cxx98-14-note@#dr479-S-copy-ctor {{declared private here}}
  }
  void g() {
    S s;
    // expected-error@-1 {{variable of type 'S' has private destructor}}
    // expected-note@#dr479-S-dtor {{declared private here}}
    throw s;
    // expected-error@-1 {{exception object of type 'S' has private destructor}}
    //   expected-note@#dr479-S-dtor {{declared private here}}
    // expected-error@-3 {{calling a private constructor of class 'dr479::S'}}
    //   expected-note@#dr479-S-copy-ctor {{declared private here}}
  }
  void h() {
    try {
      f();
      g();
    } catch (S s) {
      // expected-error@-1 {{calling a private constructor of class 'dr479::S'}}
      //   expected-note@#dr479-S-copy-ctor {{declared private here}}
      // expected-error@-3 {{variable of type 'S' has private destructor}}
      //   expected-note@#dr479-S-dtor {{declared private here}}
    }
  }
}

namespace dr480 { // dr480: yes
  struct A { int n; };
  struct B : A {};
  struct C : virtual B {};
  struct D : C {};

  int A::*a = &A::n;
  int D::*b = a;
  // expected-error@-1 {{conversion from pointer to member of class 'dr480::A' to pointer to member of class 'dr480::D' via virtual base 'dr480::B' is not allowed}}

  extern int D::*c;
  int A::*d = static_cast<int A::*>(c);
  // expected-error@-1 {{conversion from pointer to member of class 'dr480::D' to pointer to member of class 'dr480::A' via virtual base 'dr480::B' is not allowed}}

  D *e;
  A *f = e;
  D *g = static_cast<D*>(f);
  // expected-error@-1 {{cannot cast 'dr480::A *' to 'D *' via virtual base 'dr480::B'}}

  extern D &i;
  A &j = i;
  D &k = static_cast<D&>(j);
  // expected-error@-1 {{cannot cast 'A' to 'D &' via virtual base 'dr480::B'}}
}

namespace dr481 { // dr481: 2.8
  template<class T, T U> class A { T *x; };
  T *x;
  // expected-error@-1 {{unknown type name 'T'}}

  template<class T *U> class B { T *x; };
  T *y; // ok

  struct C {
    template<class T> void f(class D *p);
  };
  D *z; // ok

  template<typename A = C, typename C = A> struct E {
    void f() {
      typedef ::dr481::C c; // #dr481-c
      typedef C c;
      // expected-error@-1 {{typedef redefinition with different types ('int' vs '::dr481::C')}}
      // expected-note@#dr481-E-int {{in instantiation of member function 'dr481::E<int>::f' requested here}}
      // expected-note@#dr481-c {{previous definition is here}}
    }
  };
  template struct E<>; // ok
  template struct E<int>; // #dr481-E-int

  template<template<typename U_no_typo_correction> class A,
           A<int> *B,
           U_no_typo_correction *C>
           // expected-error@-1 {{unknown type name 'U_no_typo_correction'}}
  struct F {
    U_no_typo_correction *x;
    // expected-error@-1 {{unknown type name 'U_no_typo_correction'}}
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

  int dr482::a = 0;
  // expected-warning@-1 {{extra qualification on member 'a'}}
  void dr482::f() {}
  // expected-warning@-1 {{extra qualification on member 'f'}}

  inline namespace X {
  // cxx98-error@-1 {{inline namespaces are a C++11 feature}}
    extern int b;
    void g();
    struct S;
  }
  int dr482::b = 0;
  // expected-warning@-1 {{extra qualification on member 'b'}}
  void dr482::g() {}
  // expected-warning@-1 {{extra qualification on member 'g'}}
  struct dr482::S {};
  // expected-warning@-1 {{extra qualification on member 'S'}}

  void dr482::f();
  // expected-warning@-1 {{extra qualification on member 'f'}}
  void dr482::g();
  // expected-warning@-1 {{extra qualification on member 'g'}}

  // FIXME: The following are valid in DR482's wording, but these are bugs in
  // the wording which we deliberately don't implement.
  namespace N { typedef int type; }
  typedef int N::type;
  // expected-error@-1 {{typedef declarator cannot be qualified}}
  struct A {
    struct B;
    struct A::B {};
    // expected-error@-1 {{extra qualification on member 'B'}}

#if __cplusplus >= 201103L
    enum class C;
    enum class A::C {};
    // expected-error@-1 {{extra qualification on member 'C'}}
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
    // cxx98-error@-1 {{'long long' is a C++11 extension}}
    // cxx98-error@-2 0-1{{'long long' is a C++11 extension}}
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
  typedef C CT; // #dr484-typedef-CT
  struct CT {};
  // expected-error@-1 {{definition of type 'CT' conflicts with typedef of the same name}}
  // expected-note@#dr484-typedef-CT {{'CT' declared here}}

  namespace N {
    struct D;
    typedef D DT; // #dr484-typedef-DT
  }
  struct N::DT {};
  // expected-error@-1 {{definition of type 'DT' conflicts with typedef of the same name}}
  // expected-note@#dr484-typedef-DT {{'DT' declared here}}

  typedef struct {
    S();
    // expected-error@-1 {{a type specifier is required for all declarations}}
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
  template<typename T> T f(T *); // #dr486-f
  int &f(...);

  void g();
  int n[10];

  void h() {
    int &a = f(&g);
    int &b = f(&n);
    f<void()>(&g);
    // expected-error@-1 {{no matching function for call to 'f'}}
    // expected-note@#dr486-f {{candidate template ignored: substitution failure [with T = void ()]: function cannot return function type 'void ()'}}
    f<int[10]>(&n);
    // expected-error@-1 {{no matching function for call to 'f'}}
    // expected-note@#dr486-f {{candidate template ignored: substitution failure [with T = int[10]]: function cannot return array type 'int[10]'}}
  }
}

namespace dr487 { // dr487: yes
  enum E { e };
  int operator+(int, E); // #dr487-operator-plus
  int i[4 + e]; // #dr487-i
  // expected-error@-1 {{variable length arrays in C++ are a Clang extension}}
  //   since-cxx11-note@-2 {{non-constexpr function 'operator+' cannot be used in a constant expression}}
  //   since-cxx11-note@#dr487-operator-plus {{declared here}}
  // expected-error@#dr487-i {{variable length array declaration not allowed at file scope}}
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
    // cxx98-error@-1 {{template argument uses local type 'E'}}
  }
}

// dr489: na

namespace dr490 { // dr490: 2.8
  template<typename T> struct X {};

  struct A {
    typedef int T;
    struct K {}; // #dr490-k

    int f(T);
    int g(T);
    int h(X<T>);
    int X<T>::*i(); // #dr490-i
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
    friend int X<T>::*A::i();
    // expected-error@-1 {{return type of out-of-line definition of 'dr490::A::i' differs from that in the declaration}}
    // expected-note@#dr490-i {{previous declaration is here}}
    friend int K::*A::j();
    // expected-error@-1 {{use of undeclared identifier 'K'; did you mean 'A::K'?}}
    // expected-note@#dr490-k {{'A::K' declared here}}

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
  A c[2] = { a, {}, b[1] };
  // expected-error@-1 {{excess elements in array initializer}}
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

namespace dr496 { // dr496: sup 2094
  struct A { int n; };
  struct B { volatile int n; };
  int check1[ __is_trivially_copyable(const int) ? 1 : -1];
  // This checks the dr2094 behavior, not dr496
  int check2[ __is_trivially_copyable(volatile int) ? 1 : -1];
  int check3[ __is_trivially_constructible(A, const A&) ? 1 : -1];
  int check4[ __is_trivially_constructible(B, const B&) ? 1 : -1];
  int check5[ __is_trivially_assignable(A, const A&) ? 1 : -1];
  int check6[ __is_trivially_assignable(B, const B&) ? 1 : -1];
}

namespace dr497 { // dr497: sup 253
  void before() {
    struct S {
      mutable int i;
    };
    const S cs;
    int S::*pm = &S::i;
    cs.*pm = 88;
    // expected-error@-1 {{read-only variable is not assignable}}
  }

  void after() {
    struct S {
      S() : i(0) {}
      mutable int i;
    };
    const S cs;
    int S::*pm = &S::i;
    cs.*pm = 88;
    // expected-error@-1 {{read-only variable is not assignable}}
  }
}

namespace dr499 { // dr499: yes
  extern char str[];
  void f() { throw str; }
}
