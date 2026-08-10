// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/objc-category-3 %s -verify -fobjc-arc
// expected-no-diagnostics

// We have a definition of the base interface textually included from
// Category.h, the definition is also in the module that includes the base
// interface. We should be able to see both categories in the TU.
#include "Category.h" 
#import <Category_B.h>

void test(DVTSourceModel *m) {
  [m test:1];
  [m testB:1 matchingMask:2];
}
