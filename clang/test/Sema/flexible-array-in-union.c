// RUN: %clang_cc1 %s -verify=c -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -x c++
// RUN: %clang_cc1 %s -verify -fsyntax-only -fms-compatibility
// RUN: %clang_cc1 %s -verify -fsyntax-only -fms-compatibility -x c++

// The test checks that an attempt to initialize union with flexible array
// member with an initializer list doesn't crash clang.


union { char x[]; } r = {0}; // c-error {{flexible array member 'x' in a union is not allowed}}

// expected-no-diagnostics

