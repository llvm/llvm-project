// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/odr %s -verify -std=c++11

// expected-error@a.h:8 {{'X::n' from module 'a' is not present in definition of 'X' provided earlier}}
struct X { // expected-note {{definition has no member 'n'}}
};

@import a;

bool b = F<int>{0} == F<int>{1};

@import b;

// Trigger the declarations from a and b to be imported.
int x = f() + g();

// expected-note@a.h:5 {{definition has no member 'e2'}}
// expected-note@a.h:3 {{declaration of 'f' does not match}}
// expected-note@a.h:1 {{definition has no member 'm'}}

// expected-error@b.h:5 {{'e2' from module 'b' is not present in definition of 'E' in module 'a'}}
// expected-error@b.h:3 {{'Y::f' from module 'b' is not present in definition of 'Y' in module 'a'}}
// expected-error@b.h:2 {{'Y::m' from module 'b' is not present in definition of 'Y' in module 'a'}}
