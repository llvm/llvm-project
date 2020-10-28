// RUN: %clang_cc1 -pedantic-errors -std=c++98 -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic-errors -std=c++98 -include-pch %t -verify %s

// RUN: %clang_cc1 -pedantic-errors -std=c++11 -emit-pch %s -o %t-cxx11
// RUN: %clang_cc1 -pedantic-errors -std=c++11 -include-pch %t-cxx11 -verify %s

// RUN: %clang_cc1 -pedantic-errors -std=c++98 -emit-pch %s -o %t -fmodules
// RUN: %clang_cc1 -pedantic-errors -std=c++98 -include-pch %t -verify %s -fmodules

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED
extern const int a;
const int b = a;

#else

const int a = 5;
typedef int T[b]; // expected-error {{variable length array}} expected-error {{must be an integer constant expression}} expected-note {{initializer of 'b'}}
// expected-note@14 {{here}}
typedef int T[5];

#endif
