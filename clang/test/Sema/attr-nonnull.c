// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -fexperimental-new-constant-interpreter

void f1(int *a1, int *a2, int *a3, int *a4, int *a5, int *a6, int *a7,
        int *a8, int *a9, int *a10, int *a11, int *a12, int *a13, int *a14,
        int *a15, int *a16) __attribute__((nonnull(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)));

void f2(void) __attribute__((nonnull())); // expected-warning {{'nonnull' attribute applied to function with no pointer arguments}}

// Regression test for GH199407: verify no crash when structural equivalence
// checks nonnull attrs with VariadicParamIdx (always-valid) elements.
#define GH199407(Ty, Name) _Alignas(Ty) char Name[sizeof(Ty)]
GH199407(struct GH199407T { void *g(void *p) __attribute__((nonnull(1))); }, gbuf); // expected-error 2{{field 'g' declared as a function}} expected-error{{redefinition of 'GH199407T'}} expected-note{{previous definition is here}}
