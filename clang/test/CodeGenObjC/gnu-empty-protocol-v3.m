// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fobjc-runtime=gnustep-1.9 -emit-llvm -o - %s | FileCheck %s

@protocol X;

__attribute__((objc_root_class))
@interface Z <X>
@end

@implementation Z
@end


// CHECK:      @.objc_protocol_list = internal global { ptr, i32, [0 x ptr] } zeroinitializer, align 4
// CHECK:      @.objc_method_list = internal global { i32, [0 x { ptr, ptr }] } zeroinitializer, align 4
// CHECK:      @.objc_protocol_name = private unnamed_addr constant [2 x i8] c"X\00", align 1
// CHECK:      @._OBJC_PROTOCOL_X = internal global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { 
// CHECK-SAME:     ptr inttoptr (i32 3 to ptr),
// CHECK-SAME:     ptr @.objc_protocol_name,
// CHECK-SAME:     ptr @.objc_protocol_list
// CHECK-SAME:     ptr @.objc_method_list
// CHECK-SAME:     ptr @.objc_method_list
// CHECK-SAME:     ptr @.objc_method_list
// CHECK-SAME:     ptr @.objc_method_list
// CHECK-SAME:     ptr null
// CHECK-SAME:     ptr null
// CHECK-SAME: }, align 4
