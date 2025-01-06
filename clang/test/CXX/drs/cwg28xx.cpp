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
  // since-cxx26-warning@-1 {{dependent nested name specifier 'Ts...[0]::' for friend template declaration is not supported; ignoring this friend declaration}}
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

template <typename Ts> struct O2 : protected Ts { // #cwg2881-O2
  using Ts::operator();
  O2(Ts ts) : Ts(ts) {}
};

template <typename Ts> struct O3 : private Ts { // #cwg2881-O3
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
struct O6 : private T, U { // #cwg2881-O6
  using T::operator();
  using U::operator();
  O6(T t, U u) : T(t), U(u) {}
};

void f() {
  int x;
  auto L1 = [=](this auto&& self) { (void) &x; };
  auto L2 = [&](this auto&& self) { (void) &x; };
  O1<decltype(L1)>{L1, L1}();
  /* since-cxx23-error-re@-1 {{inaccessible due to ambiguity:
    struct cwg2881::O1<class (lambda at {{.+}})> -> A<(lambda at {{.+}})> -> class (lambda at {{.+}})
    struct cwg2881::O1<class (lambda at {{.+}})> -> B<(lambda at {{.+}})> -> class (lambda at {{.+}})}}*/
  O1<decltype(L2)>{L2, L2}();
  /* since-cxx23-error-re@-1 {{inaccessible due to ambiguity:
    struct cwg2881::O1<class (lambda at {{.+}})> -> A<(lambda at {{.+}})> -> class (lambda at {{.+}})
    struct cwg2881::O1<class (lambda at {{.+}})> -> B<(lambda at {{.+}})> -> class (lambda at {{.+}})}}*/
  O2{L1}();
  // since-cxx23-error-re@-1 {{invalid explicit object parameter type 'cwg2881::O2<(lambda at {{.+}})>' in lambda with capture; the type must derive publicly from the lambda}}
  //   since-cxx23-note@#cwg2881-O2 {{declared protected here}}
  O3{L1}();
  // since-cxx23-error-re@-1 {{invalid explicit object parameter type 'cwg2881::O3<(lambda at {{.+}})>' in lambda with capture; the type must derive publicly from the lambda}}
  //   since-cxx23-note@#cwg2881-O3 {{declared private here}}
  O4{L1}();
  O5{L1}();
  O6 o{L1, L2};
  o.decltype(L1)::operator()();
  // since-cxx23-error-re@-1 {{invalid explicit object parameter type 'cwg2881::O6<(lambda at {{.+}}), (lambda at {{.+}})>' in lambda with capture; the type must derive publicly from the lambda}}
  //   since-cxx23-note@#cwg2881-O6 {{declared private here}}
  o.decltype(L1)::operator()(); // No error here because we've already diagnosed this method.
  o.decltype(L2)::operator()();
}

void f2() {
  int x = 0;
  auto lambda = [x] (this auto self) { return x; };
  using Lambda = decltype(lambda);
  struct D : private Lambda { // #cwg2881-D
    D(Lambda l) : Lambda(l) {}
    using Lambda::operator();
    friend Lambda;
  } d(lambda);
  d();
  // since-cxx23-error@-1 {{invalid explicit object parameter type 'D' in lambda with capture; the type must derive publicly from the lambda}}
  //   since-cxx23-note@#cwg2881-D {{declared private here}}
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
struct Ambiguous : Indirect<T>, T {
/* since-cxx23-warning-re@-1 {{direct base '(lambda at {{.+}})' is inaccessible due to ambiguity:
    struct cwg2881::Ambiguous<class (lambda at {{.+}})> -> Indirect<(lambda at {{.+}})> -> class (lambda at {{.+}})
    struct cwg2881::Ambiguous<class (lambda at {{.+}})> -> class (lambda at {{.+}})}}*/
//   since-cxx23-note-re@#cwg2881-f4 {{in instantiation of template class 'cwg2881::Ambiguous<(lambda at {{.+}})>' requested here}}
//   since-cxx34-note-re@#cwg2881-f4-call {{while substituting deduced template arguments into function template 'f4' [with L = (lambda at {{.+}})]}}
  using Indirect<T>::operator();
};

template <typename L>
constexpr auto f3(L l) -> decltype(Private<L>{l}()) { return l(); } // #cwg2881-f3

template <typename L>
constexpr auto f4(L l) -> decltype(Ambiguous<L>{{l}, l}()) { return l(); } // #cwg2881-f4

template<typename T>
concept is_callable = requires(T t) { { t() }; };

void g() {
  int x = 0;
  auto lambda = [x](this auto self) {};
  f3(lambda);
  // since-cxx23-error@-1 {{no matching function for call to 'f3'}}
  //   since-cxx23-note-re@#cwg2881-f3 {{candidate template ignored: substitution failure [with L = (lambda at {{.+}})]: invalid explicit object parameter type 'cwg2881::Private<(lambda at {{.+}})>' in lambda with capture; the type must derive publicly from the lambda}}
  f4(lambda); // #cwg2881-f4-call
  // expected-error@-1 {{no matching function for call to 'f4'}}
  //   expected-note-re@-2 {{while substituting deduced template arguments into function template 'f4' [with L = (lambda at {{.+}})]}}
  /*   expected-note-re@#cwg2881-f4 {{candidate template ignored: substitution failure [with L = (lambda at {{.+}})]: lambda '(lambda at {{.+}})' is inaccessible due to ambiguity:
    struct cwg2881::Ambiguous<class (lambda at {{.+}})> -> Indirect<(lambda at {{.+}})> -> class (lambda at {{.+}})
    struct cwg2881::Ambiguous<class (lambda at {{.+}})> -> class (lambda at {{.+}})}}*/
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
