// RUN: %clang_cc1  -masm-verbose -S -g %s -o - | FileCheck %s
// CHECK: AT_APPLE_objc_complete_type

@interface Foo {} @end

@interface Foo () {
    int *bar;
}
@end

@implementation Foo
@end

void bar(Foo *fptr) {}
