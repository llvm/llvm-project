// RUN: %clang_cc1 %s -verify -fsyntax-only -std=c++11 -x c++
void foo [[clang::xray_always_instrument]] ();

struct [[clang::xray_always_instrument]] a { int x; }; // expected-warning {{'xray_always_instrument' attribute only applies to functions and methods}}

class b {
 void c [[clang::xray_always_instrument]] ();
};

void baz [[clang::xray_always_instrument("not-supported")]] (); // expected-error {{'xray_always_instrument' attribute takes no arguments}}
