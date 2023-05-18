// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

using fp0_t = void (*)();
using fp1_t = int (*)();

extern fp0_t a, b;
extern fp1_t c;

bool eq0 = a == b;
bool ne0 = a != b;
bool lt0 = a < b;   // expected-warning {{ordered comparison of function pointers ('fp0_t' (aka 'void (*)()') and 'fp0_t')}}
bool le0 = a <= b;  // expected-warning {{ordered comparison of function pointers}}
bool gt0 = a > b;   // expected-warning {{ordered comparison of function pointers}}
bool ge0 = a >= b;  // expected-warning {{ordered comparison of function pointers}}
auto tw0 = a <=> b; // expected-error {{ordered comparison of function pointers}}

bool eq1 = a == c;  // expected-error {{comparison of distinct pointer types}}
bool ne1 = a != c;  // expected-error {{comparison of distinct pointer types}}
bool lt1 = a < c;   // expected-warning {{ordered comparison of function pointers ('fp0_t' (aka 'void (*)()') and 'fp1_t' (aka 'int (*)()'))}}
                    // expected-error@-1 {{comparison of distinct pointer types}}
bool le1 = a <= c;  // expected-warning {{ordered comparison of function pointers}}
                    // expected-error@-1 {{comparison of distinct pointer types}}
bool gt1 = a > c;   // expected-warning {{ordered comparison of function pointers}}
                    // expected-error@-1 {{comparison of distinct pointer types}}
bool ge1 = a >= c;  // expected-warning {{ordered comparison of function pointers}}
                    // expected-error@-1 {{comparison of distinct pointer types}}
auto tw1 = a <=> c; // expected-error {{ordered comparison of function pointers}}
