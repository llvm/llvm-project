// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang_cc1 -triple %itanium_abi_triple -S -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: AT_APPLE_objc_complete_type

@interface I1
@end

@implementation I1 {
int myi2;
}
int myi;
@end

void foo(I1 *iptr) {}

