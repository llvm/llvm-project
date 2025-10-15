// RUN: %clang_cc1 -Wno-objc-root-class -fptrauth-calls -triple arm64e -fptrauth-objc-class-ro %s -emit-llvm -o - | FileCheck %s
@interface X
@end

@implementation X
-(void)meth {}
@end

// CHECK: @"OBJC_CLASS_$_X" = global %struct._class_t { ptr @"OBJC_METACLASS_$_X", ptr null, ptr @_objc_empty_cache, ptr null, ptr ptrauth (ptr @"_OBJC_CLASS_RO_$_X", i32 2, i64 25080, ptr getelementptr inbounds (%struct._class_t, ptr @"OBJC_CLASS_$_X", i32 0, i32 4)) }
// CHECK: @"OBJC_METACLASS_$_X" = global %struct._class_t { ptr @"OBJC_METACLASS_$_X", ptr @"OBJC_CLASS_$_X", ptr @_objc_empty_cache, ptr null, ptr ptrauth (ptr @"_OBJC_METACLASS_RO_$_X", i32 2, i64 25080, ptr getelementptr inbounds (%struct._class_t, ptr @"OBJC_METACLASS_$_X", i32 0, i32 4)) }
// CHECK: @OBJC_CLASS_NAME_ = private unnamed_addr constant [2 x i8] c"X\00"
// CHECK: @"_OBJC_METACLASS_RO_$_X" = private global %struct._class_ro_t { i32 3, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr null, ptr null, ptr null }
// CHECK: @OBJC_METH_VAR_NAME_ = private unnamed_addr constant [5 x i8] c"meth\00"
// CHECK: @OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [8 x i8] c"v16@0:8\00"
// CHECK: @"_OBJC_$_INSTANCE_METHODS_X" = private global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr ptrauth (ptr @"\01-[X meth]", i32 0, i64 0, ptr getelementptr inbounds ({ i32, i32, [1 x %struct._objc_method] }, ptr @"_OBJC_$_INSTANCE_METHODS_X", i32 0, i32 2, i32 0, i32 2)) }] }
// CHECK: @"_OBJC_CLASS_RO_$_X" = private global %struct._class_ro_t { i32 2, i32 0, i32 0, ptr null, ptr @OBJC_CLASS_NAME_, ptr ptrauth (ptr @"_OBJC_$_INSTANCE_METHODS_X", i32 2, i64 49936, ptr getelementptr inbounds (%struct._class_ro_t, ptr @"_OBJC_CLASS_RO_$_X", i32 0, i32 5)), ptr null, ptr null, ptr null, ptr null }
// CHECK: @"OBJC_LABEL_CLASS_$" = private global [1 x ptr] [ptr @"OBJC_CLASS_$_X"]
