// RUN: %clang_cc1 -fsyntax-only -verify=cuda %s
// RUN: %clang_cc1 -fsyntax-only -verify=pedantic -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify=cpp -x c++ %s

__noinline__ void fun1() { } // cpp-error {{unknown type name '__noinline__'}}

__attribute__((noinline)) void fun2() { }
__attribute__((__noinline__)) void fun3() { }
[[gnu::__noinline__]] void fun4() { }

#define __noinline__ __attribute__((__noinline__))
__noinline__ void fun5() {}

#undef __noinline__ // cuda-warning {{keyword or identifier with special meaning is used as a macro name}} \
                    // pedantic-warning {{keyword or identifier with special meaning is used as a macro name}}
#10 "cuda.h" 3 // pedantic-warning {{this style of line directive is a GNU extension}}
#define __noinline__ __attribute__((__noinline__))
__noinline__ void fun6() {}
