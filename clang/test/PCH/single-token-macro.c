// rdar://10588825

// Test this without pch.
// RUN: %clang_cc1 %s -include %s -verify -fsyntax-only

// Test with pch.
// RUN: %clang_cc1 %s -emit-pch -o %t
// RUN: %clang_cc1 %s -include-pch %t -verify -fsyntax-only

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#ifdef __stdcall
// __stdcall is defined as __attribute__((__stdcall__)) for targeting mingw32.
#undef __stdcall
#endif

#define __stdcall
#define STDCALL __stdcall

void STDCALL Foo(void);

#else

void STDCALL Foo(void)
{
}

#endif
