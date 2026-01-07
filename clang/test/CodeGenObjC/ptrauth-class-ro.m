// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -Wno-objc-root-class -fptrauth-objc-class-ro -fobjc-arc -emit-llvm -o - %s | FileCheck %s

// CHECK: @"OBJC_CLASS_$_C" = global %struct._class_t { ptr @"OBJC_METACLASS_$_C", ptr null, ptr @_objc_empty_cache, ptr null, ptr ptrauth (ptr @"_OBJC_CLASS_RO_$_C", i32 2, i64 25080, ptr getelementptr inbounds (%struct._class_t, ptr @"OBJC_CLASS_$_C", i32 0, i32 4)) }, section "__DATA, __objc_data", align 8
// CHECK: @"OBJC_METACLASS_$_C" = global %struct._class_t { ptr @"OBJC_METACLASS_$_C", ptr @"OBJC_CLASS_$_C", ptr @_objc_empty_cache, ptr null, ptr ptrauth (ptr @"_OBJC_METACLASS_RO_$_C", i32 2, i64 25080, ptr getelementptr inbounds (%struct._class_t, ptr @"OBJC_METACLASS_$_C", i32 0, i32 4)) }, section "__DATA, __objc_data", align 8
// CHECK: @OBJC_CLASS_NAME_ = private unnamed_addr constant [2 x i8] c"C\00", section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK: @"_OBJC_METACLASS_RO_$_C" = internal global %struct._class_ro_t { i32 131, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
// CHECK: @OBJC_METH_VAR_NAME_ = private unnamed_addr constant [3 x i8] c"m0\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK: @OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [8 x i8] c"v16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
// CHECK: @"_OBJC_$_INSTANCE_METHODS_C" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr ptrauth (ptr @"\01-[C m0]", i32 0, i64 0, ptr getelementptr inbounds ({ i32, i32, [1 x %struct._objc_method] }, ptr @"_OBJC_$_INSTANCE_METHODS_C", i32 0, i32 2, i32 0, i32 2)) }] }, section "__DATA, __objc_const", align 8
// CHECK: @"_OBJC_CLASS_RO_$_C" = internal global %struct._class_ro_t { i32 130, i32 0, i32 0, ptr null, ptr @OBJC_CLASS_NAME_, ptr ptrauth (ptr @"_OBJC_$_INSTANCE_METHODS_C", i32 2, i64 49936, ptr getelementptr inbounds (%struct._class_ro_t, ptr @"_OBJC_CLASS_RO_$_C", i32 0, i32 5)), ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
// CHECK: @OBJC_SELECTOR_REFERENCES_ = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
// CHECK: @"OBJC_LABEL_CLASS_$" = private global [1 x ptr] [ptr @"OBJC_CLASS_$_C"], section "__DATA,__objc_classlist,regular,no_dead_strip"

@interface C
- (void) m0;
@end

@implementation C
- (void)m0 {}
@end

void test_sign_class_ro(C *c) {
  [c m0];
}
