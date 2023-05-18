// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify -pedantic %s
double e = 0x.p0; // expected-error-re {{hexadecimal floating {{constant|literal}} requires a significand}}

float f = 0x1p+1;
double d = 0x.2p2;
float g = 0x1.2p2;
double h = 0x1.p2;
#if __cplusplus <= 201402L
// expected-warning@-5 {{hexadecimal floating literals are a C++17 feature}}
// expected-warning@-5 {{hexadecimal floating literals are a C++17 feature}}
// expected-warning@-5 {{hexadecimal floating literals are a C++17 feature}}
// expected-warning@-5 {{hexadecimal floating literals are a C++17 feature}}
#endif

// PR12717: In order to minimally diverge from the C++ standard, we do not lex
// 'p[+-]' as part of a pp-number unless the token starts 0x and doesn't contain
// an underscore.
double i = 0p+3; // expected-error {{invalid suffix 'p' on integer constant}}
#define PREFIX(x) foo ## x
double foo0p = 1, j = PREFIX(0p+3); // ok
double k = 0x42_amp+3;
#if __cplusplus > 201402L
// expected-error@-2 {{no matching literal operator for call to 'operator""_amp+3'}}
#elif __cplusplus >= 201103L
// expected-error@-4 {{no matching literal operator for call to 'operator""_amp'}}
#else
// expected-error@-6 {{invalid suffix '_amp' on integer constant}}
#endif
