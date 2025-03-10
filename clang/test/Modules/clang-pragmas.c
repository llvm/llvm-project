// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I%t %t/tu.c -fsyntax-only \
// RUN:   -verify 2>&1 | FileCheck %s

//--- module.modulemap

module A {
  header "A.h"
}

//--- A.h

//--- tu.c

#pragma clang __debug module_map A // expected-warning{{unknown module 'A'}}
#pragma clang __debug module_lookup B // expected-warning{{unable to find module 'B'}}
#pragma clang __debug module_lookup A // does header search for A
#pragma clang __debug module_map A // now finds module A

// CHECK: module A
// CHECK: module A
