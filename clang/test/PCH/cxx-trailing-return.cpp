// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch %s -o %t-cxx11
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t-cxx11 -verify %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED
typedef auto f() -> int;
typedef int g();

#else

typedef void f; // expected-error {{typedef redefinition with different types ('void' vs 'auto () -> int')}}
                // expected-note@7 {{here}}
typedef void g; // expected-error {{typedef redefinition with different types ('void' vs 'int ()')}}
                // expected-note@8 {{here}}

#endif
