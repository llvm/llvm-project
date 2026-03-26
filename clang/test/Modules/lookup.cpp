
#define import @import
import lookup_left_cxx;
#undef import
#define IMPORT(X) @import X
IMPORT(lookup_right_cxx);

// expected-warning@Inputs/lookup_left.hpp:3 {{weak identifier 'weak_identifier' never declared}}

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
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c++ -emit-module -fmodules-cache-path=%t -fmodule-name=lookup_left_cxx %S/Inputs/module.modulemap -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c++ -emit-module -fmodules-cache-path=%t -fmodule-name=lookup_right_cxx %S/Inputs/module.modulemap -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c++ -fmodules-cache-path=%t -I %S/Inputs %s -verify
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -ast-print -x objective-c++ -fmodules-cache-path=%t -I %S/Inputs %s | FileCheck -check-prefix=CHECK-PRINT %s
// FIXME: When we have a syntax for modules in C++, use that.

// CHECK-PRINT: int *f0(int *);
// CHECK-PRINT: float *f0(float *);
// CHECK-PRINT: void test(int i, float f)

