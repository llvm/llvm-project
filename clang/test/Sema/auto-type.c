// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic -std=c11

__auto_type a = 5; // expected-warning {{'__auto_type' is a GNU extension}}
__extension__ __auto_type a1 = 5;
#pragma clang diagnostic ignored "-Wgnu-auto-type"
__auto_type b = 5.0;
__auto_type c = &b;
__auto_type d = (struct {int a;}) {5};
_Static_assert(__builtin_types_compatible_p(__typeof(a), int), "");
__auto_type e = e; // expected-error {{variable 'e' declared with '__auto_type' type cannot appear in its own initializer}}

struct s { __auto_type a; }; // expected-error {{'__auto_type' not allowed in struct member}}

__auto_type f = 1, g = 1.0; // expected-error {{'__auto_type' deduced as 'int' in declaration of 'f' and deduced as 'double' in declaration of 'g'}}

__auto_type h() {} // expected-error {{'__auto_type' not allowed in function return type}}

int i() {
  struct bitfield { int field:2; };
  __auto_type j = (struct bitfield){1}.field; // expected-error {{cannot pass bit-field as __auto_type initializer in C}}

}

int k(l)
__auto_type l; // expected-error {{'__auto_type' not allowed in K&R-style function parameter}}
{}
