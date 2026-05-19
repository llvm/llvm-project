// RUN: %clang_cc1 -std=c++11 %s -verify

struct A {};
struct B : [[]] A {};
struct C : [[]] virtual A {};
struct D : [[]] public virtual A {};
struct E : public [[]] virtual A {}; // expected-error {{an attribute list cannot appear here}}
struct F : virtual [[]] public A {}; // expected-error {{an attribute list cannot appear here}}
struct G : [[noreturn]] A {}; // expected-error {{'noreturn' attribute cannot be applied to a base specifier}}
struct H : [[unknown::foobar]] A {}; // expected-warning {{unknown attribute 'unknown::foobar' ignored}}
