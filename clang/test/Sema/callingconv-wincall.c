// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64apx-pc-windows-msvc %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64apx-pc-windows-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64apx-pc-windows-cygnus %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64apx-pc-windows-msys %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64apx-unknown-uefi %s

// wincall is the default calling convention on x86_64apx PE/COFF targets
// (Windows, Cygwin, MSYS and UEFI).
void __attribute__((wincall)) foo(void);
void __attribute__((cdecl)) cdeclfoo(void);

void (*pw)(void) = foo; // no error: plain function pointers are wincall by default
void (*pc)(void) = cdeclfoo; // expected-error{{incompatible function pointer types}}

void (__attribute__((wincall)) *pw2)(void) = foo; // no error: same calling convention
