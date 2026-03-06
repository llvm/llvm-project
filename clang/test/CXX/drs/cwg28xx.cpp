// RUN: %clang_cc1 -std=c++98 -pedantic-errors -verify=expected,cxx98 %s
// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=expected,since-cxx11,cxx11-23 %s
// RUN: %clang_cc1 -std=c++14 -pedantic-errors -verify=expected,since-cxx11,cxx11-23 %s
// RUN: %clang_cc1 -std=c++17 -pedantic-errors -verify=expected,since-cxx11,cxx11-23 %s
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify=expected,since-cxx11,cxx11-23,since-cxx20 %s
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify=expected,since-cxx11,cxx11-23,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify=expected,since-cxx11,since-cxx20,since-cxx23,since-cxx26 %s


int main() {} // required for cwg2811

namespace cwg2811 { // cwg2811: 3.5
#if __cplusplus >= 201103L
void f() {
  (void)[&] {
    using T = decltype(main);
    // since-cxx11-error@-1 {{referring to 'main' within an expression is a Clang extension}}
  };
  using T2 = decltype(main);
  // since-cxx11-error@-1 {{referring to 'main' within an expression is a Clang extension}}
}

using T = decltype(main);
// since-cxx11-error@-1 {{referring to 'main' within an expression is a Clang extension}}

int main();

using U = decltype(main);
using U2 = decltype(&main);
#endif
} // namespace cwg2811

namespace cwg2813 { // cwg2813: 20
#if __cplusplus >= 202302L
struct X {
  X() = default;

  X(const X&) = delete;
  X& operator=(const X&) = delete;

  void f(this X self) { }
};

void f() {
  X{}.f();
}
#endif
} // namespace cwg2813

namespace cwg2819 { // cwg2819: 19 c++26
#if __cplusplus >= 201103L
  // CWG 2024-04-19: This issue is not a DR.
  constexpr void* p = nullptr;
  constexpr int* q = static_cast<int*>(p); // #cwg2819-q
  // cxx11-23-error@-1 {{constexpr variable 'q' must be initialized by a constant expression}}
  //   cxx11-23-note@-2 {{cast from 'void *' is not allowed in a constant expression}}
  static_assert(q == nullptr, "");
  // cxx11-23-error@-1 {{static assertion expression is not an integral constant expression}}
  //   cxx11-23-note@-2 {{initializer of 'q' is not a constant expression}}
  //   cxx11-23-note@#cwg2819-q {{declared here}}
#endif
} // namespace cwg2819

namespace cwg2823 { // cwg2823: no
#if __cplusplus >= 201103L
  constexpr int *p = 0;
  constexpr int *q1 = &*p;
  // expected-error@-1 {{constexpr variable 'q1' must be initialized by a constant expression}}
  //   expected-note@-2 {{dereferencing a null pointer is not allowed in a constant expression}}
  // FIXME: invalid: dereferencing a null pointer.
  constexpr int *q2 = &p[0];

  int arr[32];
  constexpr int *r = arr;
  // FIXME: invalid: dereferencing a past-the-end pointer.
  constexpr int *s1 = &*(r + 32);
  // FIXME: invalid: dereferencing a past-the-end pointer.
  constexpr int *s2 = &r[32];
#endif
}

namespace cwg2847 { // cwg2847: 19 review 2024-03-01

#if __cplusplus >= 202002L

template<typename>
void i();

struct A {
  template<typename>
  void f() requires true;

  template<>
  void f<int>() requires true;
  // since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

  friend void i<int>() requires true;
  // since-cxx20-error@-1 {{friend specialization cannot have a trailing requires clause unless it declares a function template}}
};

template<typename>
struct B {
  void f() requires true;

  template<typename>
  void g() requires true;

  template<typename>
  void h() requires true;

  template<>
  void h<int>() requires true;
  // since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

  friend void i<int>() requires true;
  // since-cxx20-error@-1 {{friend specialization cannot have a trailing requires clause unless it declares a function template}}
};

template<>
void B<int>::f() requires true;
// since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

template<>
template<typename T>
void B<int>::g() requires true;

#endif

} // namespace cwg2847

namespace cwg2857 { // cwg2857: no
struct A {};
template <typename>
struct D;
namespace N {
  struct B {};
  void adl_only(A*, D<int>*); // #cwg2857-adl_only
}

void f(A* a, D<int>* d) {
  adl_only(a, d);
  // expected-error@-1 {{use of undeclared identifier 'adl_only'; did you mean 'N::adl_only'?}}
  //   expected-note@#cwg2857-adl_only {{'N::adl_only' declared here}}
}

#if __cplusplus >= 201103L
template <typename>
struct D : N::B {
  // FIXME: ADL shouldn't associate it's base B and N since D is not complete here
  decltype(adl_only((A*) nullptr, (D*) nullptr)) f;
};
#endif
} // namespace cwg2857

namespace cwg2858 { // cwg2858: 19

#if __cplusplus > 202302L

template<typename... Ts>
struct A {
  // FIXME: The nested-name-specifier in the following friend declarations are declarative,
  // but we don't treat them as such (yet).
  friend void Ts...[0]::f();
  template<typename U>
  friend void Ts...[0]::g();

  friend struct Ts...[0]::B;
  // FIXME: The index of the pack-index-specifier is printed as a memory address in the diagnostic.
  template<typename U>
  friend struct Ts...[0]::C;
  // since-cxx26-warning@-1 {{dependent nested name specifier 'Ts...[0]' for friend template declaration is not supported; ignoring this friend declaration}}
};

#endif

} // namespace cwg2858

namespace cwg2877 { // cwg2877: 19
#if __cplusplus >= 202002L
enum E { x };
void f() {
  int E;
  using enum E;   // OK
}
using F = E;
using enum F;     // OK
template<class T> using EE = T;
void g() {
  using enum EE<E>;  // OK
}
#endif
} // namespace cwg2877

// cwg2881 is in cwg2881.cpp

namespace cwg2882 { // cwg2882: 2.7
struct C {
  operator void() = delete;
  // expected-warning@-1 {{conversion function converting 'cwg2882::C' to 'void' will never be used}}
  // cxx98-error@-2 {{deleted function definitions are a C++11 extension}}
};

void f(C c) {
  (void)c;
}
} // namespace cwg2882

namespace cwg2883 { // cwg2883: no
#if __cplusplus >= 201103L
void f() {
  int x;
  (void)[&] {
    return x;
  };
}
#endif
#if __cplusplus >= 202002L
struct A {
  A() = default;
  A(const A &) = delete; // #cwg2883-A-copy-ctor
  constexpr operator int() { return 42; }
};
void g() {
  constexpr A a;
  // FIXME: OK, not odr-usable from a default template argument, and not odr-used
  (void)[=]<typename T, int = a> {};
  // since-cxx20-error@-1 {{call to deleted constructor of 'const A'}}
  //   since-cxx20-note@#cwg2883-A-copy-ctor {{'A' has been explicitly marked deleted here}}
}
#endif
} // namespace cwg2883

namespace cwg2885 { // cwg2885: 16 review 2024-05-31
#if __cplusplus >= 202002L
template <class T>
struct A {
  A() requires (false) = default;
  A() : t(42) {}
  T t;
};

struct B : A<int> {};
static_assert(!__is_trivially_constructible(B));
#endif
} // namespace cwg2885

namespace cwg2886 { // cwg2886: 9
#if __cplusplus >= 201103L
struct C {
  C() = default;
  ~C() noexcept(false) = default;
};

static_assert(noexcept(C()), "");
#endif
} // namespace cwg2886
