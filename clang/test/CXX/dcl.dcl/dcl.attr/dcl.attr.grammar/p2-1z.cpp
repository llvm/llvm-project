// RUN: %clang_cc1 -std=c++1z -verify %s

[[disable_tail_calls, noduplicate]] void f() {} // expected-warning {{unknown attribute 'disable_tail_calls'}} expected-warning {{unknown attribute 'noduplicate'}}

[[using clang: disable_tail_calls, noduplicate]] void g() {} // ok

[[using]] extern int n; // expected-error {{expected identifier}}
[[using foo
] // expected-error {{expected ':'}}
] extern int n;
[[using 42:]] extern int n; // expected-error {{expected identifier}}
[[using clang:]] extern int n; // ok
[[using blah: clang::optnone]] extern int n; // expected-error {{attribute with scope specifier cannot follow}} expected-warning {{only applies to functions}}

[[using clang: unknown_attr]] extern int n; // expected-warning {{unknown attribute}}
[[using unknown_ns: something]] extern int n; // expected-warning {{unknown attribute}}
