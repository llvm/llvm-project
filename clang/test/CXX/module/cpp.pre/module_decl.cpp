// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %s -verify -o %t/M.pcm

// This is a comment
#define I32 int // expected-note {{add 'module;' to the start of the file to introduce a global module fragment}}
export module M; // expected-error {{module declaration must occur at the start of the translation unit}}
export I32 i32;
