// RUN: %clang_cc1 -fsyntax-only -verify %s
extern const int PR6495a = 42;
extern int PR6495b = 42; // expected-warning{{'extern' variable has an initializer}}
extern const int PR6495c[] = {42,43,44};

extern struct Test1 {}; // expected-warning {{'extern' ignored on this declaration}}
extern "C" struct Test0 {}; // no warning
