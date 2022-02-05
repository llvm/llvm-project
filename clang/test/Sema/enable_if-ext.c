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
void f(void) { }

__attribute__ (( __enable_if__(1, "") ))
#if defined(WARN_PEDANTIC)
// expected-warning@-2 {{'enable_if' is a clang extension}}
#endif
void g(void) { }

__attribute__ (( enable_if(0, "") ))
#if defined(WARN_PEDANTIC)
// expected-warning@-2 {{'enable_if' is a clang extension}}
#endif
void h(void) { }

__attribute__ (( __enable_if__(0, "") ))
#if defined(WARN_PEDANTIC)
// expected-warning@-2 {{'enable_if' is a clang extension}}
#endif
void i(void) { }

#pragma clang system_header

__attribute__ (( enable_if(1, "") ))
void j(void) { }

__attribute__ (( __enable_if__(1, "") ))
void k(void) { }

__attribute__ (( enable_if(0, "") ))
void l(void) { }

__attribute__ (( __enable_if__(0, "") ))
void m(void) { }

#endif

