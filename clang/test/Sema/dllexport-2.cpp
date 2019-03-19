// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -fms-extensions -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsyntax-only -fms-extensions -verify %s -DMSVC

// Export const variable.

#ifdef MSVC
// expected-error@+4 {{'j' must have external linkage when declared 'dllexport'}}
#else
// expected-warning@+2 {{__declspec attribute 'dllexport' is not supported}}
#endif
__declspec(dllexport) int const j; // expected-error {{default initialization of an object of const type 'const int'}}

// With typedef
typedef const int CInt;

#ifdef MSVC
// expected-error@+4 {{'j2' must have external linkage when declared 'dllexport'}}
#else
// expected-warning@+2 {{__declspec attribute 'dllexport' is not supported}}
#endif
__declspec(dllexport) CInt j2; //expected-error {{default initialization of an object of const type 'CInt'}}

#ifndef MSVC
// expected-warning@+2 {{__declspec attribute 'dllexport' is not supported}}
#endif
__declspec(dllexport) CInt j3 = 3;
