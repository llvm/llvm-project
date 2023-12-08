// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/objc-category-2 %s -verify -fobjc-arc

// We have a definition of category and the base interface imported from a
// module, definition for the base interface is also textually included.
// Currently we emit an error "duplicate interface definition".
#import <Category.h>
#include "H3.h"

void test(DVTSourceModel *m) {
  [m test:1];
}
