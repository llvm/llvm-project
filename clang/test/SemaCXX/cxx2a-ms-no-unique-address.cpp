// RUN: %clang_cc1 -std=c++2a %s -verify=unsupported -triple x86_64-linux-gnu
// RUN: %clang_cc1 -std=c++2a %s -verify -triple x86_64-windows -fms-compatibility

[[msvc::no_unique_address]] int a; // expected-error {{only applies to non-bit-field non-static data members}} unsupported-warning {{unknown}}
[[msvc::no_unique_address]] void f(); // expected-error {{only applies to non-bit-field non-static data members}} unsupported-warning {{unknown}}
struct [[msvc::no_unique_address]] S { // expected-error {{only applies to non-bit-field non-static data members}} unsupported-warning {{unknown}}
  [[msvc::no_unique_address]] int a; // unsupported-warning {{unknown}}
  [[msvc::no_unique_address]] void f(); // expected-error {{only applies to non-bit-field non-static data members}} unsupported-warning {{unknown}}
  [[msvc::no_unique_address]] static int sa;// expected-error {{only applies to non-bit-field non-static data members}} unsupported-warning {{unknown}}
  [[msvc::no_unique_address]] static void sf(); // expected-error {{only applies to non-bit-field non-static data members}} unsupported-warning {{unknown}}
  [[msvc::no_unique_address]] int b : 3; // expected-error {{only applies to non-bit-field non-static data members}} unsupported-warning {{unknown}}

  [[msvc::no_unique_address, msvc::no_unique_address]] int duplicated; // ok
  // unsupported-warning@-1 2{{unknown}}
  [[msvc::no_unique_address]] [[msvc::no_unique_address]] int duplicated2; // unsupported-warning 2{{unknown}}
  [[msvc::no_unique_address()]] int arglist; // expected-error {{cannot have an argument list}} unsupported-warning {{unknown}}

  int [[msvc::no_unique_address]] c; // expected-error {{cannot be applied to types}} unsupported-error {{cannot be applied to types}}
};
