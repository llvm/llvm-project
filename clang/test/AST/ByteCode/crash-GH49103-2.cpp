// RUN: %clang_cc1 -verify -std=c++98 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify -std=c++11 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify -std=c++14 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify -std=c++17 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify -std=c++20 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify -std=c++23 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify -std=c++2c %s -fexperimental-new-constant-interpreter

// https://github.com/llvm/llvm-project/issues/49103

template<class> struct A; // expected-note 0+ {{}}
struct S : __make_integer_seq<A, int, 42> { }; // expected-error 0+ {{}}
S s;
