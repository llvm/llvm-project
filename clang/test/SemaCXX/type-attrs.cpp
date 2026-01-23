// RUN: %clang_cc1 -triple x86_64-pc-win32 -Wgcc-compat -std=c++11 -verify %s

void f() [[gnu::cdecl]] {}  // expected-warning {{GCC does not allow the 'gnu::cdecl' attribute to be written on a type}}
void g() [[gnu::stdcall]] {}  // expected-warning {{GCC does not allow the 'gnu::stdcall' attribute to be written on a type}}
void i() [[gnu::fastcall]] {}  // expected-warning {{GCC does not allow the 'gnu::fastcall' attribute to be written on a type}}
