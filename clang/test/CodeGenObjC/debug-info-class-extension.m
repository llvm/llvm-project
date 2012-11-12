// RUN: %clang_cc1 -masm-verbose -S -g %s -o - | FileCheck %s

// CHECK: AT_APPLE_objc_complete_type

@interface I1
@end

@implementation I1 {
int myi2;
}
int myi;
@end

void foo(I1 *iptr) {}

