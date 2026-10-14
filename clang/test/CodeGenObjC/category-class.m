// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5  -emit-llvm -o - %s | FileCheck %s
// PR7431

// CHECK: module asm
// CHECK-NEXT: "\09.lazy_reference .objc_class_name_A"
// CHECK-NEXT: "\09.objc_category_name_A_foo=0"
// CHECK-NEXT: "\09.globl .objc_category_name_A_foo"

@interface A
@end
@interface A(foo)
- (void)foo_myStuff;
@end
@implementation A(foo)
- (void)foo_myStuff {
}
@end

