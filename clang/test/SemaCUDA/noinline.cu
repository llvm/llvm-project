// RUN: %clang_cc1 -fsyntax-only -verify=cuda %s
// RUN: %clang_cc1 -fsyntax-only -verify=cuda -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify=cpp -x c++ %s

// cuda-no-diagnostics

__noinline__ void fun1() { } // cpp-error {{unknown type name '__noinline__'}}

__attribute__((noinline)) void fun2() { }
__attribute__((__noinline__)) void fun3() { }
[[gnu::__noinline__]] void fun4() { }

#define __noinline__ __attribute__((__noinline__))
__noinline__ void fun5() {}

#undef __noinline__
#10 "cuda.h" 3
#define __noinline__ __attribute__((__noinline__))
__noinline__ void fun6() {}
