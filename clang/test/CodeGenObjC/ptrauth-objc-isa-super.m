// RUN: %clang_cc1 -I %S/Inputs -fptrauth-calls -fptrauth-objc-isa -triple arm64-apple-ios -emit-llvm -no-enable-noundef-analysis -fblocks -fobjc-arc -fobjc-runtime-has-weak -O2 -disable-llvm-passes -o - %s | FileCheck %s

#include "literal-support.h"

#if __has_feature(objc_bool)
#define YES __objc_yes
#define NO __objc_no
#else
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#endif

@class NSString;

// CHECK: @"OBJC_METACLASS_$_C" = global %struct._class_t { ptr ptrauth (ptr @"OBJC_METACLASS_$_Base", i32 2, i64 27361, ptr @"OBJC_METACLASS_$_C"), ptr ptrauth (ptr @"OBJC_METACLASS_$_Base", i32 2, i64 46507, ptr getelementptr inbounds (%struct._class_t, ptr @"OBJC_METACLASS_$_C", i32 0, i32 1)), ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_C" }
// CHECK: @"OBJC_CLASSLIST_SUP_REFS_$_" = private global ptr @"OBJC_METACLASS_$_C"
// CHECK: @OBJC_METH_VAR_NAME_ = private unnamed_addr constant [5 x i8] c"test\00"
// CHECK: @OBJC_SELECTOR_REFERENCES_ = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_
// CHECK: @"OBJC_METACLASS_$_Base" = external global %struct._class_t
// CHECK: @OBJC_CLASS_NAME_ = private unnamed_addr constant [2 x i8] c"C\00"
// CHECK: @OBJC_METH_VAR_NAME_.1 = private unnamed_addr constant [11 x i8] c"super_test\00"
// CHECK: @OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [8 x i8] c"v16@0:8\00"
// CHECK: @"_OBJC_$_CLASS_METHODS_C" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_.1, ptr @OBJC_METH_VAR_TYPE_, ptr ptrauth (ptr @"\01+[C super_test]", i32 0, i64 0, ptr getelementptr inbounds ({ i32, i32, [1 x %struct._objc_method] }, ptr @"_OBJC_$_CLASS_METHODS_C", i32 0, i32 2, i32 0, i32 2)) }] }
// CHECK: @"_OBJC_METACLASS_RO_$_C" = internal global %struct._class_ro_t { i32 129, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_, ptr ptrauth (ptr @"_OBJC_$_CLASS_METHODS_C", i32 2, i64 49936, ptr getelementptr inbounds (%struct._class_ro_t, ptr @"_OBJC_METACLASS_RO_$_C", i32 0, i32 5)), ptr null, ptr null, ptr null, ptr null }
// CHECK: @"OBJC_CLASS_$_Base" = external global %struct._class_t
// CHECK: @"_OBJC_CLASS_RO_$_C" = internal global %struct._class_ro_t { i32 128, i32 0, i32 0, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr null, ptr null, ptr null }
//        @"_OBJC_CLASS_RO_$_C" = internal global %struct._class_ro_t { i32 128, i32 0, i32 0, ptr null, ptr @OBJC_CLASS_NAME_, ptr ptrauth (ptr null, i32 2, i64 49936, ptr getelementptr inbounds (%struct._class_ro_t, ptr @"_OBJC_CLASS_RO_$_C", i32 0, i32 5)), ptr null, ptr null, ptr null, ptr null }
// CHECK: @"OBJC_CLASS_$_C" = global %struct._class_t { ptr ptrauth (ptr @"OBJC_METACLASS_$_C", i32 2, i64 27361, ptr @"OBJC_CLASS_$_C"), ptr ptrauth (ptr @"OBJC_CLASS_$_Base", i32 2, i64 46507, ptr getelementptr inbounds (%struct._class_t, ptr @"OBJC_CLASS_$_C", i32 0, i32 1)), ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_C" }
// CHECK: @"OBJC_LABEL_CLASS_$" = private global [1 x ptr] [ptr @"OBJC_CLASS_$_C"]

@interface Base
+ (void)test;
@end

@interface C : Base
@end

@implementation C
// CHECK-LABEL: define internal void @"\01+[C super_test]"(ptr %self, ptr %_cmd) #1 {
+ (void)super_test {
  return [super test];
  // CHECK: [[SELF_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: [[CMD_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: [[SUPER_STRUCT:%.*]] = alloca %struct._objc_super, align 8
  // CHECK: store ptr %self, ptr [[SELF_ADDR]], align 8, !tbaa !{{[0-9]+}}
  // CHECK: store ptr %_cmd, ptr [[CMD_ADDR]], align 8, !tbaa !{{[0-9]+}}
  // CHECK: [[TARGET:%.*]] = load ptr, ptr [[SELF_ADDR]], align 8, !tbaa !{{[0-9]+}}
  // CHECK: [[OBJC_SUPER_TARGET:%.*]] = getelementptr inbounds nuw %struct._objc_super, ptr [[SUPER_STRUCT]], i32 0, i32 0
  // CHECK: store ptr [[TARGET]], ptr [[OBJC_SUPER_TARGET]], align 8
  // CHECK: [[SUPER_REFERENCES:%.*]] = load ptr, ptr @"OBJC_CLASSLIST_SUP_REFS_$_"
  // CHECK: [[OBJC_SUPER_SUPER:%.*]] = getelementptr inbounds nuw %struct._objc_super, ptr [[SUPER_STRUCT]], i32 0, i32 1
  // CHECK: store ptr [[SUPER_REFERENCES]], ptr [[OBJC_SUPER_SUPER:%.*]], align 8
  // CHECK: call void @objc_msgSendSuper2(ptr %objc_super, ptr %4)
}
@end

id str = @"";
