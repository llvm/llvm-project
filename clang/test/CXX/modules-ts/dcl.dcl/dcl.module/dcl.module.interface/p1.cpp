// RUN: %clang_cc1 -fmodules-ts %s -verify -o /dev/null
// RUN: %clang_cc1 -fmodules-ts %s -DINTERFACE -verify -emit-module-interface -o %t
// RUN: %clang_cc1 -fmodules-ts %s -DIMPLEMENTATION -verify -fmodule-file=%t -o /dev/null
//
// RUN: %clang_cc1 -fmodules-ts %s -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null
// RUN: %clang_cc1 -fmodules-ts %s -DINTERFACE -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null
// RUN: %clang_cc1 -fmodules-ts %s -DIMPLEMENTATION -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null

#if INTERFACE
// expected-no-diagnostics
export module A;
#elif IMPLEMENTATION
module A;
 #ifdef BUILT_AS_INTERFACE
  // expected-error@-2 {{missing 'export' specifier in module declaration while building module interface}}
  #define INTERFACE
 #endif
#else
 #ifdef BUILT_AS_INTERFACE
  // expected-error@1 {{missing 'export module' declaration in module interface unit}}
 #endif
#endif

#ifndef INTERFACE
export int b; // expected-error {{export declaration can only be used within a module interface unit}}
#else
export int a;
#endif
