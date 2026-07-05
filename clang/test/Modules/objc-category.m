// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/objc-category %s -verify -fobjc-arc
// expected-no-diagnostics

// We have a definition of the base interface textually included from
// Category.h, the definition is also in the module that includes the base
// interface. We should be able to see the category in the TU.
#include "Category.h" 
#import <H3.h>

void test(DVTSourceModel *m) {
  [m test:1];
}
