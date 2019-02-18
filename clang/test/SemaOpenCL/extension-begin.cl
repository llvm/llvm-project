// Test this without pch.
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only

// Test with pch.
// RUN: %clang_cc1 -x cl %S/extension-begin.h -triple spir-unknown-unknown -emit-pch -o %t.pch -pedantic
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -include-pch %t.pch -DIMPLICIT_INCLUDE -DUSE_PCH -fsyntax-only -verify -pedantic

// Test with modules
// RUN: rm -rf %t.modules
// RUN: mkdir -p %t.modules
//
// RUN: %clang_cc1 -cl-std=CL1.2 -DIMPLICIT_INCLUDE -include %S/extension-begin.h -triple spir-unknown-unknown -O0 -emit-llvm -o - -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.modules %s -verify -pedantic
//
// RUN: rm -rf %t.modules
// RUN: mkdir -p %t.modules
//
// RUN: %clang_cc1 -cl-std=CL2.0 -DIMPLICIT_INCLUDE -include %S/extension-begin.h -triple spir-unknown-unknown -O0 -emit-llvm -o - -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.modules %s -verify -pedantic

#ifndef IMPLICIT_INCLUDE
#include "extension-begin.h"
#endif // IMPLICIT_INCLUDE
#ifndef USE_PCH
// expected-warning@extension-begin.h:4 {{expected 'disable' - ignoring}}
// expected-warning@extension-begin.h:5 {{expected 'disable' - ignoring}}
// expected-warning@extension-begin.h:21 {{OpenCL extension end directive mismatches begin directive - ignoring}}
#endif // USE_PCH

#pragma OPENCL EXTENSION my_ext : enable
void test_f1(void) {
  struct A test_A1;
  f();
  g(0);
}

#pragma OPENCL EXTENSION my_ext : disable 
void test_f2(void) {
  struct A test_A2; // expected-error {{use of type 'struct A' requires my_ext extension to be enabled}}
  const struct A test_A_local; // expected-error {{use of type 'struct A' requires my_ext extension to be enabled}}
  TypedefOfA test_typedef_A; // expected-error {{use of type 'TypedefOfA' (aka 'struct A') requires my_ext extension to be enabled}}
  PointerOfA test_A_pointer; // expected-error {{use of type 'PointerOfA' (aka 'const struct A *') requires my_ext extension to be enabled}}
  f(); // expected-error {{use of declaration 'f' requires my_ext extension to be enabled}}
  g(0); // expected-error {{no matching function for call to 'g'}}
        // expected-note@extension-begin.h:18 {{candidate unavailable as it requires OpenCL extension 'my_ext' to be enabled}}
        // expected-note@extension-begin.h:23 {{candidate function not viable: requires 0 arguments, but 1 was provided}}
}

