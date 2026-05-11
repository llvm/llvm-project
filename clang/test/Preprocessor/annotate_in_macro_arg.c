// RUN: %clang_cc1 -verify %s
#define M1() // expected-note{{macro 'M1' defined here}}

M1( // expected-error{{unterminated function-like macro invocation}}

// expected-warning@+1 {{embedding a #if directive within macro arguments has undefined behavior}}
#if M1() // expected-error{{expected value in expression}}
#endif
#pragma pack()
