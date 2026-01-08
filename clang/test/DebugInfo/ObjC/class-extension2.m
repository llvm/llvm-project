// FIXME: Check IR rather than asm, then triple is not needed.
// RUN: %clang_cc1 -triple %itanium_abi_triple -S -debug-info-kind=limited %s -o - | FileCheck %s
// CHECK: AT_APPLE_objc_complete_type

@interface Foo {} @end

@interface Foo () {
    int *bar;
}
@end

@implementation Foo
@end

void bar(Foo *fptr) {}
