// Based on C++20 10.2 example 1.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std-10-2-ex1-tu1.cpp \
// RUN: -pedantic-errors -verify -o %t/m1.pcm

//--- std-10-2-ex1.h
export int x;

//--- std-10-2-ex1-tu1.cpp
module;

#include "std-10-2-ex1.h"
// expected-error@std-10-2-ex1.h:* {{export declaration can only be used within a module interface unit after the module declaration}}

export module M1;
export namespace {} // expected-error {{declaration does not introduce any names to be exported}}
export namespace {
int a1; // expected-error {{declaration of 'a1' with internal linkage cannot be exported}}
}
namespace {    // expected-note {{anonymous namespace begins here}}
export int a2; // expected-error {{export declaration appears within anonymous namespace}}
}
export static int b; // expected-error {{declaration of 'b' with internal linkage cannot be exported}}
export int f();      // OK

export namespace N {}     // namespace N
export using namespace N; // expected-error {{ISO C++20 does not permit using directive to be exported}}
