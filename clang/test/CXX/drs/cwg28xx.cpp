// RUN: %clang_cc1 -std=c++98 -pedantic-errors -verify=expected,cxx98 %s
// RUN: %clang_cc1 -std=c++11 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -pedantic-errors -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -pedantic-errors -verify=expected,since-cxx20 %s
// RUN: %clang_cc1 -std=c++23 -pedantic-errors -verify=expected,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -verify=expected,since-cxx20,since-cxx23,since-cxx26 %s


int main() {} // required for cwg2811

namespace cwg2811 { // cwg2811: 3.5
#if __cplusplus >= 201103L
void f() {
  (void)[&] {
    using T = decltype(main);
    // expected-error@-1 {{referring to 'main' within an expression is a Clang extension}}
  };
  using T2 = decltype(main);
  // expected-error@-1 {{referring to 'main' within an expression is a Clang extension}}
}

using T = decltype(main);
// expected-error@-1 {{referring to 'main' within an expression is a Clang extension}}

int main();

using U = decltype(main);
using U2 = decltype(&main);
#endif
} // namespace cwg2811

namespace cwg2819 { // cwg2819: 19
#if __cpp_constexpr >= 202306L
  constexpr void* p = nullptr;
  constexpr int* q = static_cast<int*>(p);
  static_assert(q == nullptr);
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
  // expected-warning-re@-1 {{dependent nested name specifier 'Ts...[{{.*}}]::' for friend template declaration is not supported; ignoring this friend declaration}}
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

namespace cwg2881 { // cwg2881: 19

#if __cplusplus >= 202302L

template <typename T> struct A : T {};
template <typename T> struct B : T {};
template <typename T> struct C : virtual T { C(T t) : T(t) {} };
template <typename T> struct D : virtual T { D(T t) : T(t) {} };

template <typename Ts>
struct O1 : A<Ts>, B<Ts> {
  using A<Ts>::operator();
  using B<Ts>::operator();
};

template <typename Ts> struct O2 : protected Ts { // expected-note {{declared protected here}}
  using Ts::operator();
  O2(Ts ts) : Ts(ts) {}
};

template <typename Ts> struct O3 : private Ts { // expected-note {{declared private here}}
  using Ts::operator();
  O3(Ts ts) : Ts(ts) {}
};

// Not ambiguous because of virtual inheritance.
template <typename Ts>
struct O4 : C<Ts>, D<Ts> {
  using C<Ts>::operator();
  using D<Ts>::operator();
  O4(Ts t) : Ts(t), C<Ts>(t), D<Ts>(t) {}
};

// This still has a public path to the lambda, and it's also not
// ambiguous because of virtual inheritance.
template <typename Ts>
struct O5 : private C<Ts>, D<Ts> {
  using C<Ts>::operator();
  using D<Ts>::operator();
  O5(Ts t) : Ts(t), C<Ts>(t), D<Ts>(t) {}
};

// This is only invalid if we call T's call operator.
template <typename T, typename U>
struct O6 : private T, U { // expected-note {{declared private here}}
  using T::operator();
  using U::operator();
  O6(T t, U u) : T(t), U(u) {}
};

void f() {
  int x;
  auto L1 = [=](this auto&& self) { (void) &x; };
  auto L2 = [&](this auto&& self) { (void) &x; };
  O1<decltype(L1)>{L1, L1}(); // expected-error {{inaccessible due to ambiguity}}
  O1<decltype(L2)>{L2, L2}(); // expected-error {{inaccessible due to ambiguity}}
  O2{L1}(); // expected-error {{must derive publicly from the lambda}}
  O3{L1}(); // expected-error {{must derive publicly from the lambda}}
  O4{L1}();
  O5{L1}();
  O6 o{L1, L2};
  o.decltype(L1)::operator()(); // expected-error {{must derive publicly from the lambda}}
  o.decltype(L1)::operator()(); // No error here because we've already diagnosed this method.
  o.decltype(L2)::operator()();
}

void f2() {
  int x = 0;
  auto lambda = [x] (this auto self) { return x; };
  using Lambda = decltype(lambda);
  struct D : private Lambda { // expected-note {{declared private here}}
    D(Lambda l) : Lambda(l) {}
    using Lambda::operator();
    friend Lambda;
  } d(lambda);
  d(); // expected-error {{must derive publicly from the lambda}}
}

template <typename L>
struct Private : private L {
  using L::operator();
  Private(L l) : L(l) {}
};

template<typename T>
struct Indirect : T {
  using T::operator();
};

template<typename T>
struct Ambiguous : Indirect<T>, T { // expected-warning {{is inaccessible due to ambiguity}}
  using Indirect<T>::operator();
};

template <typename L>
constexpr auto f3(L l) -> decltype(Private<L>{l}()) { return l(); }
// expected-note@-1 {{must derive publicly from the lambda}}

template <typename L>
constexpr auto f4(L l) -> decltype(Ambiguous<L>{{l}, l}()) { return l(); }
// expected-note@-1 {{is inaccessible due to ambiguity}}
// expected-note@-2 {{in instantiation of template class}}

template<typename T>
concept is_callable = requires(T t) { { t() }; };

void g() {
  int x = 0;
  auto lambda = [x](this auto self) {};
  f3(lambda); // expected-error {{no matching function for call to 'f3'}}
  f4(lambda); // expected-error {{no matching function for call to 'f4'}}
  // expected-note@-1 {{while substituting deduced template arguments into function template 'f4'}}
  static_assert(!is_callable<Private<decltype(lambda)>>);
  static_assert(!is_callable<Ambiguous<decltype(lambda)>>);
}

#endif

} // namespace cwg2881

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
