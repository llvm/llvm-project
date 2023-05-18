// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

__attribute__((objc_root_class))
@interface A
@property(direct, readonly) int i;
@end

__attribute__((objc_root_class))
@interface B
@property(direct, readonly) int i;
@property(readonly) int j;
@end

// CHECK-NOT: @"__OBJC_$_PROP_LIST_A"
@implementation A
@synthesize i = _i;
@end

// CHECK: @"_OBJC_$_PROP_LIST_B" = internal global { i32, i32, [1 x %struct._prop_t] } { i32 16, i32 1
@implementation B
@synthesize i = _i;
@synthesize j = _j;
@end
