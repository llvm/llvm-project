// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++20-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -pedantic -verify %s

struct A { // expected-note 0+{{candidate}}
  A() = default; // expected-note 0+{{candidate}}
  int x, y;
};
A a1 = {1, 2};
#if __cplusplus <= 201703L
  // expected-warning@-2 {{aggregate initialization of type 'A' with user-declared constructors is incompatible with C++20}}
#else
  // expected-error@-4 {{no matching constructor}}
#endif
A a2 = {};

struct B : A { A a; };
B b1 = {{}, {}}; // ok
B b2 = {1, 2, 3, 4};
#if __cplusplus <= 201703L
  // expected-warning@-2 2{{aggregate initialization of type 'A' with user-declared constructors is incompatible with C++20}}
#else
  // expected-error@-4 2{{no viable conversion from 'int' to 'A'}}
#endif

// Essentially any use of a u8 string literal in C++<=17 is broken by C++20.
// Just warn on all such string literals.
struct string { string(const char*); }; // expected-note 0+{{candidate}}
char u8arr[] = u8"hello";
const char *u8ptr = "wo" u8"rld";
string u8str = u8"test" u8"test";
#if __cplusplus <= 201703L
// expected-warning@-4 {{type of UTF-8 string literal will change}} expected-note@-4 {{remove 'u8' prefix}}
// expected-warning@-4 {{type of UTF-8 string literal will change}} expected-note@-4 {{remove 'u8' prefix}}
// expected-warning@-4 {{type of UTF-8 string literal will change}} expected-note@-4 {{remove 'u8' prefix}}
#else
// expected-error@-8 {{ISO C++20 does not permit initialization of char array with UTF-8 string literal}}
// expected-error@-8 {{cannot initialize a variable of type 'const char *' with an lvalue of type 'const char8_t [6]'}}
// expected-error@-8 {{no viable conversion from 'const char8_t [9]' to 'string'}}
#endif

template<bool b>
struct C {
  explicit(C)(int);
};
#if __cplusplus <= 201703L
// expected-warning@-3 {{this expression will be parsed as explicit(bool) in C++20}}
#if defined(__cpp_conditional_explicit)
#error "the feature test macro __cpp_conditional_explicit isn't correct"
#endif
#else
// expected-error@-8 {{does not refer to a value}}
// expected-error@-9 {{expected member name or ';'}}
// expected-error@-10 {{expected ')'}}
// expected-note@-12 {{declared here}}
// expected-note@-12 {{to match this '('}}
#if !defined(__cpp_conditional_explicit) || __cpp_conditional_explicit != 201806L
#error "the feature test macro __cpp_conditional_explicit isn't correct"
#endif
#endif

auto l = []() consteval {};
int consteval();
#if __cplusplus <= 201703L
// expected-warning@-3 {{'consteval' is a keyword in C++20}}
// expected-error@-4 {{expected body of lambda expression}}
#else
// expected-error@-5 {{expected unqualified-id}}
#endif
