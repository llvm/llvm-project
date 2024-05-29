// RUN: %clang_cc1 -std=c++98 -verify=expected %s
// RUN: %clang_cc1 -std=c++11 -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,since-cxx20 %s
// RUN: %clang_cc1 -std=c++23 -verify=expected,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -std=c++2c -verify=expected,since-cxx20,since-cxx23,since-cxx26 %s

namespace cwg2819 { // cwg2819: 19 tentatively ready 2023-12-01
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

namespace cwg2858 { // cwg2858: 19 tentatively ready 2024-04-05

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

namespace cwg2881 { // cwg2881: 19 tentatively ready 2024-04-19

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

#endif

} // namespace cwg2881

