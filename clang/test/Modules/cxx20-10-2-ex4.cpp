// Based on C++20 10.2 example 4.

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -verify -o M.pcm

export module M;

struct S { // expected-note {{previous declaration is here}}
  int n;
};
typedef S S;
export typedef S S; // OK, does not redeclare an entity
export struct S;    // expected-error {{cannot export redeclaration 'S' here since the previous declaration has module linkage}}
