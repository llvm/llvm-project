// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s


// Check that we have a method list that refers to the correct thing method:
// CHECK: internal global { ptr, i32, i64, [1 x { ptr, ptr, ptr }] } { ptr null, i32 1, i64 24, 
// CHECK-SAME: @_i_X_Cat_x 
// CHECK-SAME: @".objc_selector_x_i16\010:8"

// Check that we emit the correct encoding for the property (somewhere)
// CHECK: c"Ti,R\00"

// Check that we emit a single-element property list of the correct form.
// CHECK: internal global { i32, i32, ptr, [1 x { ptr, ptr, ptr, ptr, ptr }] }

// CHECK: @.objc_category_XCat = internal global { ptr, ptr, ptr, ptr, ptr, ptr, ptr }
// CHECK-SAME: section "__objc_cats", align 8

@interface X @end

@interface X (Cat)
@property (readonly) int x;
@end

@implementation X (Cat)
- (int)x { return 12; }
@end
