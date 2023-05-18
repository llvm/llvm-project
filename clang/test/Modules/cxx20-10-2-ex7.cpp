// Based on C++20 10.2 example 6.

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -verify -o %t

// expected-no-diagnostics

export module M;
export namespace N {
int x;                 // OK
static_assert(1 == 1); // No diagnostic after P2615R1 DR
} // namespace N
