// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98-14,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,cxx98-14,cxx11-17,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx14,cxx98-14,cxx11-17,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx14,since-cxx17,cxx11-17,since-cxx11,since-cxx14,cxx17 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,cxx11-20,since-cxx14,since-cxx17,since-cxx20,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx14,since-cxx17,since-cxx20,since-cxx23,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx14,since-cxx17,since-cxx20,since-cxx23,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace cwg1800 { // cwg1800: 2.9
struct A { union { int n; }; };
static_assert(__is_same(__decltype(&A::n), int A::*), "");
} // namespace cwg1800

namespace cwg1801 { // cwg1801: 2.8
static union {
  int i;
};

template <int &> struct S {}; // #cwg1801-S
S<i> V; // #cwg1801-S-i
// cxx98-14-error@-1 {{non-type template argument does not refer to any declaration}}
//   cxx98-14-note@#cwg1801-S {{template parameter is declared here}}
// cxx17-error@#cwg1801-S-i {{non-type template argument refers to subobject '.i'}}
//   cxx17-note@#cwg1801-S {{template parameter is declared here}}
} // namespace cwg1801

namespace cwg1802 { // cwg1802: 3.1
#if __cplusplus >= 201103L
// Using a Wikipedia example of surrogate pair:
// https://en.wikipedia.org/wiki/UTF-16#Examples
constexpr char16_t a[3] = u"\U00010437";
static_assert(a[0] == 0xD801, "");
static_assert(a[1] == 0xDC37, "");
static_assert(a[2] == 0x0, "");
#endif
} // namespace cwg1802

namespace cwg1803 { // cwg1803: 2.9
#if __cplusplus >= 201103L
struct A {
  enum E : int;
  enum E : int {};
  enum class EC;
  enum class EC {};
  enum struct ES;
  enum struct ES {};
};
#endif
} // namespace cwg1803

namespace cwg1804 { // cwg1804: 2.7
template <typename, typename>
struct A {
  void f1();

  template <typename V>
  void f2(V);

  class B {
    void f3();
  };

  template <typename>
  class C {
    void f4();
  };
};

template <typename U>
struct A<int, U> {
  void f1();

  template <typename V>
  void f2(V);

  class B {
    void f3();
  };

  template <typename>
  class C {
    void f4();
  };
};

class D {
  int i;

  template <typename, typename>
  friend struct A;
};

template <typename U>
struct A<double, U> {
  void f1();

  template <typename V>
  void f2(V);

  class B {
    void f3();
  };

  template <typename>
  class C {
    void f4();
  };
};

template <typename U>
void A<int, U>::f1() {
  D d;
  d.i = 0;
}

template <typename U>
void A<double, U>::f1() {
  D d;
  d.i = 0;
}

template <typename U>
template <typename V>
void A<int, U>::f2(V) {
  D d;
  d.i = 0;
}

template <typename U>
template <typename V>
void A<double, U>::f2(V) {
  D d;
  d.i = 0;
}

template <typename U>
void A<int, U>::B::f3() {
  D d;
  d.i = 0;
}

template <typename U>
void A<double, U>::B::f3() {
  D d;
  d.i = 0;
}

template <typename U>
template <typename V>
void A<int, U>::C<V>::f4() {
  D d;
  d.i = 0;
}

template <typename U>
template <typename V>
void A<double, U>::C<V>::f4() {
  D d;
  d.i = 0;
}
} // namespace cwg1804

// cwg1807 is in cwg1807.cpp

namespace cwg1812 { // cwg1812: no
                   // NB: dup 1710
#if __cplusplus >= 201103L
template <typename T> struct A {
  using B = typename T::C<int>;
  // since-cxx11-error@-1 {{use 'template' keyword to treat 'C' as a dependent template name}}
};
#endif
} // namespace cwg1812

namespace cwg1813 { // cwg1813: 7
  struct B { int i; };
  struct C : B {};
  struct D : C {};
  struct E : D { char : 4; };

  static_assert(__is_standard_layout(B), "");
  static_assert(__is_standard_layout(C), "");
  static_assert(__is_standard_layout(D), "");
  static_assert(!__is_standard_layout(E), "");

  struct Q {};
  struct S : Q {};
  struct T : Q {};
  struct U : S, T {};

  static_assert(__is_standard_layout(Q), "");
  static_assert(__is_standard_layout(S), "");
  static_assert(__is_standard_layout(T), "");
  static_assert(!__is_standard_layout(U), "");
}

namespace cwg1814 { // cwg1814: 3.1
#if __cplusplus >= 201103L
  void test() {
    auto lam = [](int x = 42) { return x; };
  }
#endif
} // namespace cwg1814

namespace cwg1815 { // cwg1815: 20
#if __cplusplus >= 201402L
  struct A { int &&r = 0; };
  A a = {};

  struct B { int &&r = 0; }; // #cwg1815-B
  // since-cxx14-error@-1 {{reference member 'r' binds to a temporary object whose lifetime would be shorter than the lifetime of the constructed object}}
  //   since-cxx14-note@#cwg1815-B {{initializing field 'r' with default member initializer}}
  //   since-cxx14-note@#cwg1815-b {{in implicit default constructor for 'cwg1815::B' first required here}}
  B b; // #cwg1815-b

#if __cplusplus >= 201703L
  struct C { const int &r = 0; };
  constexpr C c = {}; // OK, since cwg1815
  static_assert(c.r == 0);

  constexpr int f() {
    A a = {}; // OK, since cwg1815
    return a.r;
  }
  static_assert(f() == 0);
#endif
#endif
} // namespace cwg1815

// cwg1818 is in cwg1818.cpp

namespace cwg1820 { // cwg1820: 3.5
typedef int A;
typedef int cwg1820::A;
// expected-warning@-1 {{extra qualification on member 'A'}}
// expected-error@-2 {{typedef declarator cannot be qualified}}

namespace B {
typedef int cwg1820::A;
// expected-error@-1 {{cannot define or redeclare 'A' here because namespace 'B' does not enclose namespace 'cwg1820'}}
// expected-error@-2 {{typedef declarator cannot be qualified}}
}

class C1 {
  typedef int cwg1820::A;
  // expected-error@-1 {{non-friend class member 'A' cannot have a qualified name}}
  // expected-error@-2 {{typedef declarator cannot be qualified}}
};

template <typename>
class C2 {
  typedef int cwg1820::A;
  // expected-error@-1 {{non-friend class member 'A' cannot have a qualified name}}
  // expected-error@-2 {{typedef declarator cannot be qualified}}
};

void d1() {
  typedef int cwg1820::A;
  // expected-error@-1 {{definition or redeclaration of 'A' not allowed inside a function}}
  // expected-error@-2 {{typedef declarator cannot be qualified}}
}

template<typename>
void d2() {
  typedef int cwg1820::A;
  // expected-error@-1 {{definition or redeclaration of 'A' not allowed inside a function}}
  // expected-error@-2 {{typedef declarator cannot be qualified}}
}

#if __cplusplus >= 201103L
auto e = [] {
  typedef int cwg1820::A;
  // since-cxx11-error@-1 {{definition or redeclaration of 'A' not allowed inside a function}}
  // since-cxx11-error@-2 {{typedef declarator cannot be qualified}}
};
#endif
} // namespace cwg1820

namespace cwg1821 { // cwg1821: 2.9
struct A {
  template <typename> struct B {
    void f();
  };
  template <typename T> void B<T>::f(){};
  // expected-error@-1 {{non-friend class member 'f' cannot have a qualified name}}

  struct C {
    void f();
  };
  void C::f() {}
  // expected-error@-1 {{non-friend class member 'f' cannot have a qualified name}}
};
} // namespace cwg1821

namespace cwg1822 { // cwg1822: 3.1
#if __cplusplus >= 201103L
  double a;
  auto x = [] (int a) {
    static_assert(__is_same(decltype(a), int), "should be resolved to lambda parameter");
  };
#endif
} // namespace cwg1822

namespace cwg1824 { // cwg1824: 2.7
template<typename T>
struct A {
  T t;
};

struct S {
  A<S> f() { return A<S>(); }
};
} // namespace cwg1824

namespace cwg1832 { // cwg1832: 3.0
enum E { // #cwg1832-E
  a = static_cast<int>(static_cast<E>(0))
  // expected-error@-1 {{'E' is an incomplete type}}
  //   expected-note@#cwg1832-E {{definition of 'cwg1832::E' is not complete until the closing '}'}}
};

#if __cplusplus >= 201103L
enum E2: decltype(static_cast<E2>(0), 0) {};
// since-cxx11-error@-1 {{unknown type name 'E2'}}
enum class E3: decltype(static_cast<E3>(0), 0) {};
// since-cxx11-error@-1 {{unknown type name 'E3'}}
#endif
} // namespace cwg1832

namespace cwg1837 { // cwg1837: 3.3
#if __cplusplus >= 201103L
  template <typename T>
  struct Fish { static const bool value = true; };

  struct Other {
    int p();
    auto q() -> decltype(p()) *;
  };

  class Outer {
    friend auto Other::q() -> decltype(this->p()) *;
    // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
    int g();
    int f() {
      extern void f(decltype(this->g()) *);
      struct Inner {
        static_assert(Fish<decltype(this->g())>::value, "");
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
        enum { X = Fish<decltype(this->f())>::value };
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
        struct Inner2 : Fish<decltype(this->g())> { };
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
        friend void f(decltype(this->g()) *);
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
        friend auto Other::q() -> decltype(this->p()) *;
        // since-cxx11-error@-1 {{invalid use of 'this' outside of a non-static member function}}
      };
      return 0;
    }
  };

  struct A {
    int f();
    bool b = [] {
      // since-cxx11-warning@-1 {{address of lambda function pointer conversion operator will always evaluate to 'true'}}
      struct Local {
        static_assert(sizeof(this->f()) == sizeof(int), "");
      };
    };
  };
#endif
} // namespace cwg1837

namespace cwg1862 { // cwg1862: no
template<class T>
struct A {
  struct B {
    void e();
  };

  void f();

  struct D {
    void g();
  };

  T h();

  template<T U>
  T i();
};

template<>
struct A<int> {
  struct B {
    void e();
  };

  int f();

  struct D {
    void g();
  };

  template<int U>
  int i();
};

template<>
struct A<float*> {
  int* h();
};

class C {
  int private_int;

  template<class T>
  friend struct A<T>::B;
  // expected-warning@-1 {{dependent nested name specifier 'A<T>' for friend class declaration is not supported; turning off access control for 'C'}}

  template<class T>
  friend void A<T>::f();
  // expected-warning@-1 {{dependent nested name specifier 'A<T>' for friend class declaration is not supported; turning off access control for 'C'}}

  // FIXME: this is ill-formed, because A<T>​::​D does not end with a simple-template-id
  template<class T>
  friend void A<T>::D::g();
  // expected-warning@-1 {{dependent nested name specifier 'A<T>::D' for friend class declaration is not supported; turning off access control for 'C'}}

  template<class T>
  friend int *A<T*>::h();
  // expected-warning@-1 {{dependent nested name specifier 'A<T *>' for friend class declaration is not supported; turning off access control for 'C'}}

  template<class T>
  template<T U>
  friend T A<T>::i();
  // expected-warning@-1 {{dependent nested name specifier 'A<T>' for friend class declaration is not supported; turning off access control for 'C'}}
};

C c;

template<class T>
void A<T>::B::e() { (void)c.private_int; }
void A<int>::B::e() { (void)c.private_int; }

template<class T>
void A<T>::f() { (void)c.private_int; }
int A<int>::f() { (void)c.private_int; return 0; }

// FIXME: both definition of 'D::g' are not friends, so they don't have access to 'private_int'
template<class T>
void A<T>::D::g() { (void)c.private_int; }
void A<int>::D::g() { (void)c.private_int; }

template<class T>
T A<T>::h() { (void)c.private_int; }
int* A<float*>::h() { (void)c.private_int; return 0; }

template<class T>
template<T U>
T A<T>::i() { (void)c.private_int; }
template<int U>
int A<int>::i() { (void)c.private_int; }
} // namespace cwg1862

namespace cwg1872 { // cwg1872: 9
#if __cplusplus >= 201103L
  template<typename T> struct A : T {
    constexpr int f() const { return 0; }
  };
  struct X {};
  struct Y { virtual int f() const; };
  struct Z : virtual X {};

  constexpr int x = A<X>().f();
  constexpr int y = A<Y>().f();
  // cxx11-17-error@-1 {{constexpr variable 'y' must be initialized by a constant expression}}
  //   cxx11-17-note@-2 {{cannot evaluate call to virtual function in a constant expression in C++ standards before C++20}}
#if __cplusplus >= 202002L
  static_assert(y == 0);
#endif
  // Note, this is invalid even though it would not use virtual dispatch.
  constexpr int y2 = A<Y>().A<Y>::f();
  // cxx11-17-error@-1 {{constexpr variable 'y2' must be initialized by a constant expression}}
  //   cxx11-17-note@-2 {{cannot evaluate call to virtual function in a constant expression in C++ standards before C++20}}
#if __cplusplus >= 202002L
  static_assert(y2 == 0);
#endif
  constexpr int z = A<Z>().f();
  // since-cxx11-error@-1 {{constexpr variable 'z' must be initialized by a constant expression}}
  //   cxx11-20-note@-2 {{non-literal type 'A<Z>' cannot be used in a constant expression}}
  //   since-cxx23-note@-3 {{cannot construct object of type 'A<cwg1872::Z>' with virtual base class in a constant expression}}
#endif
} // namespace cwg1872

namespace cwg1878 { // cwg1878: 18
#if __cplusplus >= 201402L
#if __cplusplus >= 202002L
template <typename T>
concept C = true;
#endif

struct S {
  template <typename T>
  operator auto() const { return short(); }
  // since-cxx14-error@-1 {{'auto' not allowed in declaration of conversion function template}}
  template <typename T>
  operator const auto() const { return int(); }
  // since-cxx14-error@-1 {{'auto' not allowed in declaration of conversion function template}}
  template <typename T>
  operator const auto&() const { return char(); }
  // since-cxx14-error@-1 {{'auto' not allowed in declaration of conversion function template}}
  template <typename T>
  operator const auto*() const { return long(); }
  // since-cxx14-error@-1 {{'auto' not allowed in declaration of conversion function template}}
  template <typename T>
  operator decltype(auto)() const { return unsigned(); }
  // since-cxx14-error@-1 {{'decltype(auto)' not allowed in declaration of conversion function template}}
#if __cplusplus >= 202002L
  template <typename T>
  operator C auto() const { return float(); }
  // since-cxx20-error@-1 {{'auto' not allowed in declaration of conversion function template}}
  template <typename T>
  operator C decltype(auto)() const { return double(); }
  // since-cxx20-error@-1 {{'decltype(auto)' not allowed in declaration of conversion function template}}
#endif
};
#endif
} // namespace cwg1878

namespace cwg1881 { // cwg1881: 7
  struct A { int a : 4; };
  struct B : A { int b : 3; };
  static_assert(__is_standard_layout(A), "");
  static_assert(!__is_standard_layout(B), "");

  struct C { int : 0; };
  struct D : C { int : 0; };
  static_assert(__is_standard_layout(C), "");
  static_assert(!__is_standard_layout(D), "");
} // namespace cwg1881

// cwg1884 is in cwg1884.cpp

namespace cwg1890 { // cwg1890: no drafting 2018-06-04
// FIXME: current consensus for CWG2335 is that the examples are well-formed.
namespace ex1 {
#if __cplusplus >= 201402L
struct A {
  struct B {
    auto foo() { return 0; } // #cwg1890-foo
  };
  decltype(B().foo()) x;
  // since-cxx14-error@-1 {{function 'foo' with deduced return type cannot be used before it is defined}}
  //   since-cxx14-note@#cwg1890-foo {{'foo' declared here}}
};
#endif
} // namespace ex1

namespace ex2 {
#if __cplusplus >= 201103L
struct Bar {
  struct Baz {
    int a = 0;
  };
  static_assert(__is_constructible(Baz), "");
  // since-cxx11-error@-1 {{static assertion failed due to requirement '__is_constructible(cwg1890::ex2::Bar::Baz)'}}
};
#endif
} // namespace ex2
} // namespace cwg1890

void cwg1891() { // cwg1891: 4
#if __cplusplus >= 201103L
  int n;
  auto a = []{}; // #cwg1891-a
  auto b = [=]{ return n; }; // #cwg1891-b
  typedef decltype(a) A;
  typedef decltype(b) B;

  static_assert(!__is_trivially_constructible(A), "");
  // since-cxx20-error@-1 {{failed}}
  static_assert(!__is_trivially_constructible(B), "");

  // C++20 allows default construction for non-capturing lambdas (P0624R2).
  A x;
  // cxx11-17-error@-1 {{no matching constructor for initialization of 'A' (aka '(lambda at}}
  //   cxx11-17-note@#cwg1891-a {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
  //   cxx11-17-note@#cwg1891-a {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}
  B y;
  // since-cxx11-error@-1 {{no matching constructor for initialization of 'B' (aka '(lambda at}}
  //   since-cxx11-note@#cwg1891-b {{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 0 were provided}}
  //   since-cxx11-note@#cwg1891-b {{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 0 were provided}}

  // C++20 allows assignment for non-capturing lambdas (P0624R2).
  a = a;
  // cxx11-17-error-re@-1 {{{{object of type '\(lambda at .+\)' cannot be assigned because its copy assignment operator is implicitly deleted}}}}
  //   cxx11-17-note@#cwg1891-a {{lambda expression begins here}}
  a = static_cast<A&&>(a);
  // cxx11-17-error-re@-1 {{{{object of type '\(lambda at .+\)' cannot be assigned because its copy assignment operator is implicitly deleted}}}}
  //   cxx11-17-note@#cwg1891-a {{lambda expression begins here}}
  b = b;
  // since-cxx11-error-re@-1 {{{{object of type '\(lambda at .+\)' cannot be assigned because its copy assignment operator is implicitly deleted}}}}
  //   since-cxx11-note@#cwg1891-b {{lambda expression begins here}}
  b = static_cast<B&&>(b);
  // since-cxx11-error-re@-1 {{{{object of type '\(lambda at .+\)' cannot be assigned because its copy assignment operator is implicitly deleted}}}}
  //   since-cxx11-note@#cwg1891-b {{lambda expression begins here}}
#endif
} // void cwg1891()

namespace cwg1894 { // cwg1894: 3.8
                   // NB: reusing part of cwg407 test
namespace A {
  struct S {};
}
namespace B {
  typedef int S;
}
namespace E {
  typedef A::S S;
  using A::S;
  struct S s;
}
namespace F {
  typedef A::S S;
}
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
} // namespace cwg1894

namespace cwg1898 { // cwg1898: 2.7
void e(int) {} // #cwg1898-e
void e(int) {}
// expected-error@-1 {{redefinition of 'e'}}
//   expected-note@#cwg1898-e {{previous definition is here}}

void e2(int) {}
void e2(long) {} // OK, different type

void f(int) {} // #cwg1898-f
void f(const int) {}
// expected-error@-1 {{redefinition of 'f'}}
//   expected-note@#cwg1898-f {{previous definition is here}}

void g(int) {} // #cwg1898-g
void g(volatile int) {}
// since-cxx20-warning@-1 {{volatile-qualified parameter type 'volatile int' is deprecated}}
// expected-error@-2 {{redefinition of 'g'}}
//   expected-note@#cwg1898-g {{previous definition is here}}

void h(int *) {} // #cwg1898-h
void h(int[]) {}
// expected-error@-1 {{redefinition of 'h'}}
//   expected-note@#cwg1898-h {{previous definition is here}}

void h2(int *) {} // #cwg1898-h2
void h2(int[2]) {}
// expected-error@-1 {{redefinition of 'h2'}}
//   expected-note@#cwg1898-h2 {{previous definition is here}}

void h3(int (*)[2]) {} // #cwg1898-h3
void h3(int [3][2]) {}
// expected-error@-1 {{redefinition of 'h3'}}
//   expected-note@#cwg1898-h3 {{previous definition is here}}

void h4(int (*)[2]) {}
void h4(int [3][3]) {} // OK, differ in non-top-level extent of array

void i(int *) {}
void i(const int *) {} // OK, pointee cv-qualification is not discarded

void i2(int *) {} // #cwg1898-i2
void i2(int * const) {}
// expected-error@-1 {{redefinition of 'i2'}}
//   expected-note@#cwg1898-i2 {{previous definition is here}}

void j(void(*)()) {} // #cwg1898-j
void j(void()) {}
// expected-error@-1 {{redefinition of 'j'}}
//   expected-note@#cwg1898-j {{previous definition is here}}

void j2(void(int)) {} // #cwg1898-j2
void j2(void(const int)) {}
// expected-error@-1 {{redefinition of 'j2'}}
//   expected-note@#cwg1898-j2 {{previous definition is here}}

struct A {
  void k(int) {} // #cwg1898-k
  void k(int) {}
  // expected-error@-1 {{class member cannot be redeclared}}
  //   expected-note@#cwg1898-k {{previous definition is here}}
};

struct B : A {
  void k(int) {} // OK, shadows A::k
};

void l() {}
void l(...) {}

#if __cplusplus >= 201103L
template <typename T>
void m(T) {}
template <typename... Ts>
void m(Ts...) {}

template <typename T, typename U>
void m2(T, U) {}
template <typename... Ts, typename U>
void m2(Ts..., U) {}
#endif
} // namespace cwg1898
