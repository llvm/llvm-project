// RUN: %clang_cc1 -no-opaque-pointers -triple i686-pc-linux-gnu -fobjc-runtime=gnustep-1.9 -emit-llvm -o - %s | FileCheck %s

@protocol X;

__attribute__((objc_root_class))
@interface Z <X>
@end

@implementation Z
@end


// CHECK:      @.objc_protocol_list = internal global { i8*, i32, [0 x i8*] } zeroinitializer, align 4
// CHECK:      @.objc_method_list = internal global { i32, [0 x { i8*, i8* }] } zeroinitializer, align 4
// CHECK:      @.objc_protocol_name = private unnamed_addr constant [2 x i8] c"X\00", align 1
// CHECK:      @._OBJC_PROTOCOL_X = internal global { i8*, i8*, { i8*, i32, [0 x i8*] }*, i8*, i8*, i8*, i8*, i8*, i8* } { 
// CHECK-SAME:     i8* inttoptr (i32 3 to i8*),
// CHECK-SAME:     i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.objc_protocol_name, i32 0, i32 0),
// CHECK-SAME:     { i8*, i32, [0 x i8*] }* @.objc_protocol_list
// CHECK-SAME:     { i32, [0 x { i8*, i8* }] }* @.objc_method_list
// CHECK-SAME:     { i32, [0 x { i8*, i8* }] }* @.objc_method_list
// CHECK-SAME:     { i32, [0 x { i8*, i8* }] }* @.objc_method_list
// CHECK-SAME:     { i32, [0 x { i8*, i8* }] }* @.objc_method_list
// CHECK-SAME:     i8* null
// CHECK-SAME:     i8* null
// CHECK-SAME: }, align 4
