// RUN: %clang_cc1 -std=c++20 %s -verify


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
auto e = g(0); // expected-error{{no matching function for call}}
// expected-note@-2 {{substitution failure}}

namespace PR52073 {
// OK, these are distinct functions not redefinitions.
template<typename> void f(decltype([]{})) {} // expected-note {{candidate}}
template<typename> void f(decltype([]{})) {} // expected-note {{candidate}}
void use_f() { f<int>({}); } // expected-error {{ambiguous}}

// Same.
template<int N> void g(const char (*)[([]{ return N; })()]) {} // expected-note {{candidate}}
template<int N> void g(const char (*)[([]{ return N; })()]) {} // expected-note {{candidate}}
// FIXME: We instantiate the lambdas into the context of the function template,
// so we think they're dependent and can't evaluate a call to them.
void use_g() { g<6>(&"hello"); } // expected-error {{no matching function}}
}
