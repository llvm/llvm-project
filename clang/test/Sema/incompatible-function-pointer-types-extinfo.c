// RUN: %clang_cc1 -fsyntax-only %s -std=c99 -verify=expected
// RUN: %clang_cc1 -fsyntax-only %s -std=c11 -verify=expected
// RUN: %clang_cc1 -fsyntax-only %s -std=c23 -verify=expected,proto

// This verifies that -Wincompatible-function-pointer-type diagnostics for
// extended function type information are consistent, also in case of other
// allowed funcion type difference in C.
//
// Test case adapted from issue #160474.

enum E { A = -1, B };

// Case 1: assignment adding noreturn

int      f1   (int);
int    (*fp1a)(int) __attribute__((noreturn)) = &f1; // expected-error {{incompatible function pointer types}}
enum E (*fp1b)(int) __attribute__((noreturn)) = &f1; // expected-error {{incompatible function pointer types}}
int    (*fp1c)()    __attribute__((noreturn)) = &f1; // expected-error {{incompatible function pointer types}}

// Case 2: assignment adding noescape on arg

int   f2   (int*                          ) __attribute__((noreturn));
int (*fp2a)(int* __attribute__((noescape))) __attribute__((noreturn)) = &f2; // expected-error {{incompatible function pointer types}}
int (*fp2b)(int* __attribute__((noescape)))                           = &f2; // expected-error {{incompatible function pointer types}}

// Case 3: assignment adding cfi_unchecked_callee

int   f3   (int*                          );
int (*fp3a)(int*                          ) __attribute__((noreturn                     )) = &f3; // expected-error {{incompatible function pointer types}}
int (*fp3b)(int* __attribute__((noescape)))                                                = &f3; // expected-error {{incompatible function pointer types}}
int (*fp3c)(int*                          ) __attribute__((noreturn,cfi_unchecked_callee)) = &f3; // expected-error {{incompatible function pointer types}}
int (*fp3d)(int* __attribute__((noescape))) __attribute__((         cfi_unchecked_callee)) = &f3; // expected-error {{incompatible function pointer types}}
int (*fp3e)(int* __attribute__((noescape))) __attribute__((noreturn,cfi_unchecked_callee)) = &f3; // expected-error {{incompatible function pointer types}}

// Case 4: assignment to function with no prototype

int   f4   (int);
int (*fp4a)(int) = &f4;
int (*fp4b)()    = &f4; // proto-error {{incompatible function pointer types}}
