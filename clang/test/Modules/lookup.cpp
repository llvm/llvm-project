
#define import @__experimental_modules_import
import lookup_left_cxx;
#undef import
#define IMPORT(X) @__experimental_modules_import X
IMPORT(lookup_right_cxx);

// in lookup_left.hpp: expected-warning@3 {{weak identifier 'weak_identifier' never declared}}

void test(int i, float f) {
  // unqualified lookup
  f0(&i);
  f0(&f);

  // qualified lookup into the translation unit
  ::f0(&i);
  ::f0(&f);
}

int import;

void f() {
 int import;
}

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c++ -emit-module -fmodule-cache-path %t -fmodule-name=lookup_left_cxx %S/Inputs/module.map -verify
// RUN: %clang_cc1 -fmodules -x objective-c++ -emit-module -fmodule-cache-path %t -fmodule-name=lookup_right_cxx %S/Inputs/module.map -verify
// RUN: %clang_cc1 -fmodules -x objective-c++ -fmodule-cache-path %t %s -verify
// RUN: %clang_cc1 -fmodules -ast-print -x objective-c++ -fmodule-cache-path %t %s | FileCheck -check-prefix=CHECK-PRINT %s
// FIXME: When we have a syntax for modules in C++, use that.

// CHECK-PRINT: int *f0(int *);
// CHECK-PRINT: float *f0(float *);
// CHECK-PRINT: void test(int i, float f)

