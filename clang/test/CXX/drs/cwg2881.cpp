// RUN: %clang_cc1 -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98-20
// RUN: %clang_cc1 -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98-20
// RUN: %clang_cc1 -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98-20
// RUN: %clang_cc1 -std=c++17 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98-20
// RUN: %clang_cc1 -std=c++20 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,cxx98-20
// RUN: %clang_cc1 -std=c++23 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx23
// RUN: %clang_cc1 -std=c++2c %s -fexceptions -fcxx-exceptions -pedantic-errors -verify=expected,since-cxx23

// cxx98-20-no-diagnostics

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
  /* since-cxx23-error-re@-1 {{lambda '(lambda at {{.+}})' is inaccessible due to ambiguity:
    struct cwg2881::O1<class (lambda at {{.+}})> -> A<class (lambda at {{.+}})> -> class (lambda at {{.+}})
    struct cwg2881::O1<class (lambda at {{.+}})> -> B<class (lambda at {{.+}})> -> class (lambda at {{.+}})}}*/
  O1<decltype(L2)>{L2, L2}();
  /* since-cxx23-error-re@-1 {{lambda '(lambda at {{.+}})' is inaccessible due to ambiguity:
    struct cwg2881::O1<class (lambda at {{.+}})> -> A<class (lambda at {{.+}})> -> class (lambda at {{.+}})
    struct cwg2881::O1<class (lambda at {{.+}})> -> B<class (lambda at {{.+}})> -> class (lambda at {{.+}})}}*/
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
    struct cwg2881::Ambiguous<class (lambda at {{.+}})> -> Indirect<class (lambda at {{.+}})> -> class (lambda at {{.+}})
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
    struct cwg2881::Ambiguous<class (lambda at {{.+}})> -> Indirect<class (lambda at {{.+}})> -> class (lambda at {{.+}})
    struct cwg2881::Ambiguous<class (lambda at {{.+}})> -> class (lambda at {{.+}})}}*/
  static_assert(!is_callable<Private<decltype(lambda)>>);
  static_assert(!is_callable<Ambiguous<decltype(lambda)>>);
}
#endif
} // namespace cwg2881
