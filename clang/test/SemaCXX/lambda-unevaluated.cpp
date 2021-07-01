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
