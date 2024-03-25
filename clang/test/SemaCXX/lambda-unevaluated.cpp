// RUN: %clang_cc1 -std=c++20 %s -Wno-c++23-extensions -verify
// RUN: %clang_cc1 -std=c++23 %s -verify


template <auto> struct Nothing {};
Nothing<[]() { return 0; }()> nothing;

template <typename> struct NothingT {};
Nothing<[]() { return 0; }> nothingT;

template <typename T>
concept True = [] { return true; }();
static_assert(True<int>);

static_assert(sizeof([] { return 0; }));
static_assert(sizeof([] { return 0; }()));

void f()  noexcept(noexcept([] { return 0; }()));

using a = decltype([] { return 0; });
using b = decltype([] { return 0; }());
using c = decltype([]() noexcept(noexcept([] { return 0; }())) { return 0; });
using d = decltype(sizeof([] { return 0; }));

template <auto T>
int unique_test1();
static_assert(&unique_test1<[](){}> != &unique_test1<[](){}>);

template <class T>
auto g(T) -> decltype([]() { T::invalid; } ());
auto e = g(0); // expected-error@-1{{type 'int' cannot be used prior to '::'}}
               // expected-note@-1{{while substituting deduced template}}
               // expected-note@-3{{while substituting into a lambda}}
               // expected-error@-3 {{no matching function for call to 'g'}}
               // expected-note@-5 {{substitution failure}}

template <typename T>
auto foo(decltype([] {
  return [] { return T(); }();
})) {}

void test() {
  foo<int>({});
}

template <typename T>
struct C {
  template <typename U>
  auto foo(decltype([] {
    return [] { return T(); }();
  })) {}
};

void test2() {
  C<int>{}.foo<long>({});
}

namespace PR52073 {
// OK, these are distinct functions not redefinitions.
template<typename> void f(decltype([]{})) {} // expected-note {{candidate}}
template<typename> void f(decltype([]{})) {} // expected-note {{candidate}}
void use_f() { f<int>({}); } // expected-error {{ambiguous}}

// Same.
template<int N> void g(const char (*)[([]{ return N; })()]) {} // expected-note {{candidate}}
template<int N> void g(const char (*)[([]{ return N; })()]) {} // expected-note {{candidate}}
void use_g() { g<6>(&"hello"); } // expected-error {{ambiguous}}
}

namespace GH51416 {

template <class T>
struct A {
  void spam(decltype([] {}));
};

template <class T>
void A<T>::spam(decltype([] {})) // expected-error{{out-of-line definition of 'spam' does not match}}
{}

struct B {
  template <class T>
  void spam(decltype([] {}));
};

template <class T>
void B::spam(decltype([] {})) {} // expected-error{{out-of-line definition of 'spam' does not match}}

} // namespace GH51416

namespace GH50376 {

template <typename T, typename Fn>
struct foo_t {    // expected-note 2{{candidate constructor}}
  foo_t(T ptr) {} // expected-note{{candidate constructor}}
};

template <typename T>
using alias = foo_t<T, decltype([](int) { return 0; })>;

template <typename T>
auto fun(T const &t) -> alias<T> {
  return alias<T>{t}; // expected-error{{no viable conversion from returned value of type 'alias<...>'}}
}

void f() {
  int i;
  auto const error = fun(i); // expected-note{{in instantiation}}
}

} // namespace GH50376

namespace GH51414 {
template <class T> void spam(decltype([] {}) (*s)[sizeof(T)] = nullptr) {}
void foo() {
  spam<int>();
}
} // namespace GH51414

namespace GH51641 {
template <class T>
void foo(decltype(+[](T) {}) lambda, T param);
static_assert(!__is_same(decltype(foo<int>), void));
} // namespace GH51641

namespace StaticLambdas {
template <auto> struct Nothing {};
Nothing<[]() static { return 0; }()> nothing;

template <typename> struct NothingT {};
Nothing<[]() static { return 0; }> nothingT;

template <typename T>
concept True = [] static { return true; }();
static_assert(True<int>);

static_assert(sizeof([] static { return 0; }));
static_assert(sizeof([] static { return 0; }()));

void f()  noexcept(noexcept([] static { return 0; }()));

using a = decltype([] static { return 0; });
using b = decltype([] static { return 0; }());
using c = decltype([]() static noexcept(noexcept([] { return 0; }())) { return 0; });
using d = decltype(sizeof([] static { return 0; }));

}

namespace lambda_in_trailing_decltype {
auto x = ([](auto) -> decltype([] {}()) {}(0), 2);
}

namespace lambda_in_constraints {
struct WithFoo { static void foo(); };

template <class T>
concept lambda_works = requires {
    []() { T::foo(); }; // expected-error{{type 'int' cannot be used prior to '::'}}
                        // expected-note@-1{{while substituting into a lambda expression here}}
                        // expected-note@-2{{in instantiation of requirement here}}
                        // expected-note@-4{{while substituting template arguments into constraint expression here}}
};

static_assert(!lambda_works<int>); // expected-note {{while checking the satisfaction of concept 'lambda_works<int>' requested here}}
static_assert(lambda_works<WithFoo>);

template <class T>
int* func(T) requires requires { []() { T::foo(); }; }; // expected-error{{type 'int' cannot be used prior to '::'}}
                                                        // expected-note@-1{{while substituting into a lambda expression here}}
                                                        // expected-note@-2{{in instantiation of requirement here}}
                                                        // expected-note@-3{{while substituting template arguments into constraint expression here}}
double* func(...);

static_assert(__is_same(decltype(func(0)), double*)); // expected-note {{while checking constraint satisfaction for template 'func<int>' required here}}
                                                      // expected-note@-1 {{in instantiation of function template specialization 'lambda_in_constraints::func<int>'}}
static_assert(__is_same(decltype(func(WithFoo())), int*));

template <class T>
auto direct_lambda(T) -> decltype([] { T::foo(); }) {}
void direct_lambda(...) {}

void recursive() {
    direct_lambda(0); // expected-error@-4 {{type 'int' cannot be used prior to '::'}}
                      // expected-note@-1 {{while substituting deduced template arguments}}
                      // expected-note@-6 {{while substituting into a lambda}}
    bool x = requires { direct_lambda(0); }; // expected-error@-7 {{type 'int' cannot be used prior to '::'}}
                                             // expected-note@-1 {{while substituting deduced template arguments}}
                                             // expected-note@-9 {{while substituting into a lambda}}

}
}
