// Test this without pch.
// RUN: %clang_cc1 -fsyntax-only %s -DBODY

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only %s -DBODY

// Test with pch with template instantiation in the pch.
// RUN: %clang_cc1 -emit-pch -fpch-instantiate-templates -o %t %s -verify

#ifndef HEADER_H
#define HEADER_H

template <typename T>
void f();
struct X;            // @16
void g() { f<X>(); } // @17 instantiation not performed yet

template <typename T>
void f() { T t; }; // @20

#endif

#ifdef BODY
struct X {};
#endif

// expected-error@20 {{variable has incomplete type}}
// expected-note@17 {{in instantiation of function template specialization}}
// expected-note@16 {{forward declaration}}
