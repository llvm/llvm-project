// RUN: %clang_cc1 -fsyntax-only %s -include %s -verify
// RUN: %clang_cc1 -Wpedantic -fsyntax-only %s -include %s -verify -DWARN_PEDANTIC

#ifndef enable_if_ext_included
#define enable_if_ext_included

#if !defined(WARN_PEDANTIC)
// expected-no-diagnostics
#endif

__attribute__ (( enable_if(1, "") ))
#if defined(WARN_PEDANTIC)
// expected-warning@-2 {{'enable_if' is a clang extension}}
#endif
void f() { }

__attribute__ (( __enable_if__(1, "") ))
#if defined(WARN_PEDANTIC)
// expected-warning@-2 {{'enable_if' is a clang extension}}
#endif
void g() { }

__attribute__ (( enable_if(0, "") ))
#if defined(WARN_PEDANTIC)
// expected-warning@-2 {{'enable_if' is a clang extension}}
#endif
void h() { }

__attribute__ (( __enable_if__(0, "") ))
#if defined(WARN_PEDANTIC)
// expected-warning@-2 {{'enable_if' is a clang extension}}
#endif
void i() { }

#pragma clang system_header

__attribute__ (( enable_if(1, "") ))
void j() { }

__attribute__ (( __enable_if__(1, "") ))
void k() { }

__attribute__ (( enable_if(0, "") ))
void l() { }

__attribute__ (( __enable_if__(0, "") ))
void m() { }

#endif

