// RUN: %clang_cc1 -triple arm64-apple-ios -x c   -fsyntax-only -verify -fptrauth-intrinsics %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple arm64-apple-ios -x c++ -fsyntax-only -verify -fptrauth-intrinsics %s -fexperimental-new-constant-interpreter

struct Foo {
  void (*f)(int) __ptrauth(1,1,1);
  // expected-error@-1 {{expected ';' at end of declaration list}}
};
