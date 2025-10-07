// RUN: %clang_cc1 -S -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK-NOT: AT_APPLE_objc_complete_type

@interface Foo {} @end

@interface Foo () {
    int *bar;
}
@end

void bar(Foo *fptr) {}
