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

#pragma OPENCL EXTENSION my_ext : enable
#ifndef IMPLICIT_INCLUDE
// expected-warning@-2 {{OpenCL extension 'my_ext' unknown or does not require pragma - ignoring}}
// expected-warning@+2 {{OpenCL extension 'my_ext' unknown or does not require pragma - ignoring}}
#endif // IMPLICIT_INCLUDE
#pragma OPENCL EXTENSION my_ext : disable

#ifndef IMPLICIT_INCLUDE
#include "extension-begin.h"
#endif // IMPLICIT_INCLUDE
#ifndef USE_PCH
// expected-warning@extension-begin.h:4 {{expected 'disable' - ignoring}}
// expected-warning@extension-begin.h:5 {{expected 'disable' - ignoring}}
#endif // USE_PCH

#if defined(IMPLICIT_INCLUDE) && defined(USE_PCH)
//expected-no-diagnostics
#endif

// Tests that the pragmas are accepted for backward compatibility.
#pragma OPENCL EXTENSION my_ext : enable
#pragma OPENCL EXTENSION my_ext : disable 

#ifndef my_ext
#error "Missing my_ext macro"
#endif
 
// When extension is supported its functionality can be used freely.
void test(void) {
  struct A test_A1;
  f();
  g(0);
}
