// RUN: %clang_cc1 -fsyntax-only -std=c++14 -Wpre-c++14-compat-pedantic -verify %s

decltype(auto) x = 0; // expected-warning {{'decltype(auto)' type specifier is incompatible}}

auto init_capture = [a(0)] {}; // expected-warning {{initialized lambda captures are incompatible with C++ standards before C++14}}

auto generic_lambda =
  [](
       auto // expected-warning {{generic lambdas are incompatible}}
    *a) {};

auto deduced_return_type(); // expected-warning {{incompatible with C++ standards before C++14}}
auto *another_deduced_return_type(); // expected-warning {{incompatible with C++ standards before C++14}}
decltype(auto) also_deduced_return_type(); // expected-warning {{return type deduction}} expected-warning {{'decltype(auto)' type specifier is incompatible}}
int f();
auto (*not_deduced_return_type)() = f;

auto deduced_lambda_return_type = []() ->
  auto // expected-warning {{return type deduction is incompatible}}
{};

auto trailing_non_deduced_return_type() -> int;
auto trailing_deduced_return_type() -> auto; // expected-warning {{incompatible with C++ standards before C++14}}

struct A {
  operator auto(); // expected-warning {{return type deduction is incompatible}}
};

template<typename T> T var = T(10); // expected-warning {{variable templates are incompatible with C++ standards before C++14}}

// No diagnostic for specializations of variable templates; we will have
// diagnosed the primary template.
template<typename T> T* var<T*> = new T();
template<> int var<int> = 10;
template char var<char>;
float fvar = var<float>;

class B {
  template<typename T> static T var = T(10); // expected-warning {{variable templates are incompatible with C++ standards before C++14}}

  template<typename T> static T* var<T*> = new T();
};

struct C {  template<typename T> static T v; }; // expected-warning {{variable templates are incompatible with C++ standards before C++14}}

template<typename T> T C::v = T(); // expected-warning {{variable templates are incompatible with C++ standards before C++14}}

template<typename T> T* C::v<T*> = new T();
template<> int C::v<int> = 10;
template char C::v<char>;
float fsvar = C::v<float>;

int digit_seps = 123'456; // expected-warning {{digit separators are incompatible with C++ standards before C++14}}

int k = 0b1001; // expected-warning {{binary integer literals are incompatible with C++ standards before C++14}}
