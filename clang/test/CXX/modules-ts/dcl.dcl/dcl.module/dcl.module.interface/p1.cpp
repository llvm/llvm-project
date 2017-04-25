// RUN: %clang_cc1 -fmodules-ts %s -verify -o /dev/null
// RUN: %clang_cc1 -fmodules-ts %s -DINTERFACE -verify -o /dev/null
// RUN: %clang_cc1 -fmodules-ts %s -DIMPLEMENTATION -verify -o /dev/null
//
// RUN: %clang_cc1 -fmodules-ts %s -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null
// RUN: %clang_cc1 -fmodules-ts %s -DINTERFACE -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null
// RUN: %clang_cc1 -fmodules-ts %s -DIMPLEMENTATION -DBUILT_AS_INTERFACE -emit-module-interface -verify -o /dev/null

#if INTERFACE
export module A;
#elif IMPLEMENTATION
module A;
 #ifdef BUILT_AS_INTERFACE
  // expected-error@-2 {{missing 'export' specifier in module declaration while building module interface}}
 #endif
#else
 #ifdef BUILT_AS_INTERFACE
  // FIXME: Diagnose missing module declaration (at end of TU)
 #endif
#endif

export int a;
#ifndef INTERFACE
// expected-error@-2 {{export declaration can only be used within a module interface unit}}
#else
// expected-no-diagnostics
#endif
