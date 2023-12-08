// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang_cc1 -triple %itanium_abi_triple -S -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: AT_APPLE_property_name
@interface C {
  int _base;
}
@property int base;
@end

@implementation C
@synthesize base = _base;
@end

void foo(C *cptr) {}
