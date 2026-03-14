// Based on C++20 10.2 example 3.

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -o %t

export module M;
struct S;
export using T = S; // OK, exports name T denoting type S

// expected-no-diagnostics
