// RUN: %clang_cc1 -triple i386-apple-darwin9 -O0 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// PR7431

// CHECK-NOT: @"OBJC_LABEL_CATEGORY_$" = private global [1 x i8*] [i8* bitcast (%struct._category_t* @"_OBJC_$_CATEGORY_A_$_foo"

@interface A
@end
__attribute__((objc_direct_members))
@interface A(foo)
- (void)foo_myStuff;
@end
@implementation A(foo)
- (void)foo_myStuff {
}
@end

