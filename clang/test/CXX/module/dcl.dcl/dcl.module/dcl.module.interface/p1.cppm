// RUN: %clang_cc1 -std=c++20 %s -verify -emit-module-interface -o /dev/null
// RUN: %clang_cc1 -std=c++20 %s -DINTERFACE -verify -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++20 %s -DIMPLEMENTATION -verify -fmodule-file=A=%t -o /dev/null
//
// RUN: %clang_cc1 -std=c++20 %s -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null
// RUN: %clang_cc1 -std=c++20 %s -DINTERFACE -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null
// RUN: %clang_cc1 -std=c++20 %s -DIMPLEMENTATION -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null

#if INTERFACE
// expected-no-diagnostics
export module A;
#elif IMPLEMENTATION
module A; // #module-decl
 #ifdef BUILT_AS_INTERFACE
  // expected-error@-2 {{missing 'export' specifier in module declaration while building module interface}}
  #define INTERFACE
 #endif
#else // Not in a module
// expected-error@* {{missing 'export module' declaration in module interface unit}}
#endif

#ifndef INTERFACE
export int b; // expected-error {{export declaration can only be used within a module purview}}
#ifdef IMPLEMENTATION
// expected-note@#module-decl {{add 'export' here}}
#endif
#else
export int a;
#endif
