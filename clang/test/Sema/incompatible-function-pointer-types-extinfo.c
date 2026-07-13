// RUN: %clang_cc1 -fsyntax-only %s -std=c99 -verify=expected
// RUN: %clang_cc1 -fsyntax-only %s -std=c11 -verify=expected
// RUN: %clang_cc1 -fsyntax-only %s -std=c23 -verify=expected,proto

// This verifies that -Wincompatible-function-pointer-type diagnostics for
// extended function type information are consistent, also in case of other
// allowed funcion type difference in C.
//
// Test case adapted from issue #41465, with suggestions from PR #160477.

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

// Case 4: assignment converting between prototype/no-prototype

void p1(void);
void i1(int );
void n1(    );
void p2(void) __attribute__((noreturn));
void i2(int ) __attribute__((noreturn));
void n2(    ) __attribute__((noreturn));

void (*t1)() = p1;
void (*t2)() = i1; // proto-error {{incompatible function pointer types}}
void (*t3)() = n1;
void (*t4)() __attribute__((noreturn)) = p1; // expected-error {{incompatible function pointer types}}
void (*t5)() __attribute__((noreturn)) = i1; // expected-error {{incompatible function pointer types}}
void (*t6)() __attribute__((noreturn)) = n1; // expected-error {{incompatible function pointer types}}

void (*t7)() = p2;
void (*t8)() = i2; // proto-error {{incompatible function pointer types}}
void (*t9)() = n2;
void (*tA)() __attribute__((noreturn)) = p2;
void (*tB)() __attribute__((noreturn)) = i2; // proto-error {{incompatible function pointer types}}
void (*tC)() __attribute__((noreturn)) = n2;
