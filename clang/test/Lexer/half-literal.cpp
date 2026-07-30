// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify -pedantic -triple aarch64-linux-gnu %s
float a = 1.0h; // expected-error{{no matching literal operator for call to 'operator""h' with argument of type 'long double' or 'const char *', and no matching literal operator template}}
float b = 1.0H; // expected-error{{invalid suffix 'H' on floating constant}}

_Float16 c = 1.f166; // expected-error{{invalid suffix 'f166' on floating constant}}
_Float16 d = 1.f1;   // expected-error{{invalid suffix 'f1' on floating constant}}
