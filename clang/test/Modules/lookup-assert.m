// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/lookup-assert %s -verify
// expected-no-diagnostics

#include "Derive.h"
#import <H3.h>
@implementation DerivedInterface
- (void)test {
}
@end
