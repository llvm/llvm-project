// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/lookup-assert-protocol %s -verify
// expected-no-diagnostics

#include "Derive.h"
#import <H3.h>

__attribute__((objc_root_class))
@interface Thing<DerivedProtocol>
@end

@implementation Thing
- (void)test {
}
- (void)test2 {
}
@end
