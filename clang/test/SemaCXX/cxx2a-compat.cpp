// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++2a-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++2a -pedantic -verify %s

struct A { // expected-note 0+{{candidate}}
  A() = default; // expected-note 0+{{candidate}}
  int x, y;
};
A a1 = {1, 2};
#if __cplusplus <= 201703L
  // expected-warning@-2 {{aggregate initialization of type 'A' with user-declared constructors is incompatible with C++2a}}
#else
  // expected-error@-4 {{no matching constructor}}
#endif
A a2 = {};

struct B : A { A a; };
B b1 = {{}, {}}; // ok
B b2 = {1, 2, 3, 4};
#if __cplusplus <= 201703L
  // expected-warning@-2 2{{aggregate initialization of type 'A' with user-declared constructors is incompatible with C++2a}}
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
