// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang_cc1 -triple %itanium_abi_triple -S -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: AT_APPLE_property_name
// CHECK-NOT: AT_APPLE_property_getter
// CHECK-NOT: AT_APPLE_property_setter
// CHECK: AT_APPLE_property_attribute
// CHECK: AT_APPLE_property


@interface I1
@property int p1;
@end

@implementation I1
@end

void foo(I1 *ptr) {}
