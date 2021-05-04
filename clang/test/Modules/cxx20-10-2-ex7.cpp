// Based on C++20 10.2 example 6.

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -verify -o M.pcm

export module M;
export namespace N {
int x;                 // OK
static_assert(1 == 1); // expected-error {{static_assert declaration cannot be exported}}
} // namespace N
