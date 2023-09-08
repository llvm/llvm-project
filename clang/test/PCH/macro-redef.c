// RUN: %clang_cc1 %s -emit-pch -o %t1.pch -verify
// RUN: %clang_cc1 %s -emit-pch -o %t2.pch -include-pch %t1.pch -verify
// RUN: %clang_cc1 -fsyntax-only %s -include-pch %t2.pch -verify

// Test that a redefinition inside the PCH won't manifest as an ambiguous macro.

#ifndef HEADER1
#define HEADER1

#define M1 0 // expected-note {{previous}}
#define M1 1 // expected-warning {{redefined}}

#define M2 3

#elif !defined(HEADER2)
#define HEADER2

#define M2 4 // expected-warning {{redefined}}
 // expected-note@-6 {{previous}}

#else

// Use the error to verify it was parsed.
int x = M1; // expected-note {{previous}}
int x = M2; // expected-error {{redefinition}}

#endif
