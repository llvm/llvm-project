// RUN: %clang_cc1 -triple arm64-apple-macosx -fblocks -ffeature-availability=feature1:on -ffeature-availability=feature2:off -ffeature-availability=feature3:on -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -fblocks -emit-llvm -o - -DUSE_DOMAIN %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -fblocks -emit-llvm -o - -DUSE_DOMAIN -DALWAYS_ENABLED %s | FileCheck %s

#include <availability_domain.h>

#define AVAIL 0

#ifdef USE_DOMAIN
#ifdef ALWAYS_ENABLED
CLANG_ALWAYS_ENABLED_AVAILABILITY_DOMAIN(feature1);
CLANG_ALWAYS_ENABLED_AVAILABILITY_DOMAIN(feature3);
#else
CLANG_ENABLED_AVAILABILITY_DOMAIN(feature1);
CLANG_ENABLED_AVAILABILITY_DOMAIN(feature3);
#endif
CLANG_DISABLED_AVAILABILITY_DOMAIN(feature2);
#endif

// CHECK: @"OBJC_CLASS_$_C0" = global %struct._class_t { ptr @"OBJC_METACLASS_$_C0", ptr null, ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"_OBJC_CLASS_RO_$_C0" }, section "__DATA, __objc_data", align 8
// CHECK-NEXT: @"OBJC_METACLASS_$_C0" = global %struct._class_t { ptr @"OBJC_METACLASS_$_C0", ptr @"OBJC_CLASS_$_C0", ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"_OBJC_METACLASS_RO_$_C0" }, section "__DATA, __objc_data", align 8
// CHECK-NEXT: @OBJC_CLASS_NAME_ = private unnamed_addr constant [3 x i8] c"C0\00", section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_METACLASS_RO_$_C0" = internal global %struct._class_ro_t { i32 3, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @OBJC_METH_VAR_NAME_ = private unnamed_addr constant [3 x i8] c"m0\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [8 x i8] c"v16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.1 = private unnamed_addr constant [3 x i8] c"m1\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.2 = private unnamed_addr constant [6 x i8] c"prop0\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_.3 = private unnamed_addr constant [8 x i8] c"i16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.4 = private unnamed_addr constant [10 x i8] c"setProp0:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_.5 = private unnamed_addr constant [11 x i8] c"v20@0:8i16\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_INSTANCE_METHODS_C0" = internal global { i32, i32, [4 x %struct._objc_method] } { i32 24, i32 4, [4 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01-[C0 m0]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.1, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01-[C0 m1]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.2, ptr @OBJC_METH_VAR_TYPE_.3, ptr @"\01-[C0 prop0]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.4, ptr @OBJC_METH_VAR_TYPE_.5, ptr @"\01-[C0 setProp0:]" }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"OBJC_IVAR_$_C0.ivar0" = constant i32 0, section "__DATA, __objc_ivar", align 4
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.6 = private unnamed_addr constant [6 x i8] c"ivar0\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_.7 = private unnamed_addr constant [2 x i8] c"i\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
// CHECK-NEXT: @"OBJC_IVAR_$_C0._prop0" = hidden constant i32 4, section "__DATA, __objc_ivar", align 4
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.8 = private unnamed_addr constant [7 x i8] c"_prop0\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_INSTANCE_VARIABLES_C0" = internal global { i32, i32, [2 x %struct._ivar_t] } { i32 32, i32 2, [2 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_C0.ivar0", ptr @OBJC_METH_VAR_NAME_.6, ptr @OBJC_METH_VAR_TYPE_.7, i32 2, i32 4 }, %struct._ivar_t { ptr @"OBJC_IVAR_$_C0._prop0", ptr @OBJC_METH_VAR_NAME_.8, ptr @OBJC_METH_VAR_TYPE_.7, i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @OBJC_PROP_NAME_ATTR_ = private unnamed_addr constant [6 x i8] c"prop0\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_PROP_NAME_ATTR_.9 = private unnamed_addr constant [11 x i8] c"Ti,V_prop0\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_PROP_LIST_C0" = internal global { i32, i32, [1 x %struct._prop_t] } { i32 16, i32 1, [1 x %struct._prop_t] [%struct._prop_t { ptr @OBJC_PROP_NAME_ATTR_, ptr @OBJC_PROP_NAME_ATTR_.9 }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"_OBJC_CLASS_RO_$_C0" = internal global %struct._class_ro_t { i32 2, i32 0, i32 8, ptr null, ptr @OBJC_CLASS_NAME_, ptr @"_OBJC_$_INSTANCE_METHODS_C0", ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_C0", ptr null, ptr @"_OBJC_$_PROP_LIST_C0" }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @OBJC_CLASS_NAME_.10 = private unnamed_addr constant [5 x i8] c"Cat0\00", section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK-NEXT: @"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
// CHECK-NEXT: @OBJC_CLASS_NAME_.11 = private unnamed_addr constant [3 x i8] c"C2\00", section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_METACLASS_RO_$_C2" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_.11, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"OBJC_METACLASS_$_C2" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"_OBJC_METACLASS_RO_$_C2" }, section "__DATA, __objc_data", align 8
// CHECK-NEXT: @"OBJC_CLASS_$_NSObject" = external global %struct._class_t
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.12 = private unnamed_addr constant [6 x i8] c"ivar1\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_.13 = private unnamed_addr constant [8 x i8] c"@16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.14 = private unnamed_addr constant [10 x i8] c"setIvar1:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_.15 = private unnamed_addr constant [11 x i8] c"v24@0:8@16\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.16 = private unnamed_addr constant [6 x i8] c"ivar3\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.17 = private unnamed_addr constant [10 x i8] c"setIvar3:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.18 = private unnamed_addr constant [6 x i8] c"ivar4\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.19 = private unnamed_addr constant [10 x i8] c"setIvar4:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_INSTANCE_METHODS_C2" = internal global { i32, i32, [6 x %struct._objc_method] } { i32 24, i32 6, [6 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_.12, ptr @OBJC_METH_VAR_TYPE_.13, ptr @"\01-[C2 ivar1]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.14, ptr @OBJC_METH_VAR_TYPE_.15, ptr @"\01-[C2 setIvar1:]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.16, ptr @OBJC_METH_VAR_TYPE_.13, ptr @"\01-[C2 ivar3]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.17, ptr @OBJC_METH_VAR_TYPE_.15, ptr @"\01-[C2 setIvar3:]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.18, ptr @OBJC_METH_VAR_TYPE_.13, ptr @"\01-[C2 ivar4]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.19, ptr @OBJC_METH_VAR_TYPE_.15, ptr @"\01-[C2 setIvar4:]" }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"OBJC_IVAR_$_C2._ivar1" = hidden constant i32 8, section "__DATA, __objc_ivar", align 4
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.20 = private unnamed_addr constant [7 x i8] c"_ivar1\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_TYPE_.21 = private unnamed_addr constant [2 x i8] c"@\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
// CHECK-NEXT: @"OBJC_IVAR_$_C2._ivar3" = hidden constant i32 16, section "__DATA, __objc_ivar", align 4
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.22 = private unnamed_addr constant [7 x i8] c"_ivar3\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @"OBJC_IVAR_$_C2._ivar4" = hidden constant i32 24, section "__DATA, __objc_ivar", align 4
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.23 = private unnamed_addr constant [7 x i8] c"_ivar4\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_INSTANCE_VARIABLES_C2" = internal global { i32, i32, [3 x %struct._ivar_t] } { i32 32, i32 3, [3 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_C2._ivar1", ptr @OBJC_METH_VAR_NAME_.20, ptr @OBJC_METH_VAR_TYPE_.21, i32 3, i32 8 }, %struct._ivar_t { ptr @"OBJC_IVAR_$_C2._ivar3", ptr @OBJC_METH_VAR_NAME_.22, ptr @OBJC_METH_VAR_TYPE_.21, i32 3, i32 8 }, %struct._ivar_t { ptr @"OBJC_IVAR_$_C2._ivar4", ptr @OBJC_METH_VAR_NAME_.23, ptr @OBJC_METH_VAR_TYPE_.21, i32 3, i32 8 }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @OBJC_PROP_NAME_ATTR_.24 = private unnamed_addr constant [6 x i8] c"ivar1\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_PROP_NAME_ATTR_.25 = private unnamed_addr constant [11 x i8] c"T@,V_ivar1\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_PROP_NAME_ATTR_.26 = private unnamed_addr constant [6 x i8] c"ivar3\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_PROP_NAME_ATTR_.27 = private unnamed_addr constant [11 x i8] c"T@,V_ivar3\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_PROP_NAME_ATTR_.28 = private unnamed_addr constant [6 x i8] c"ivar4\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_PROP_NAME_ATTR_.29 = private unnamed_addr constant [11 x i8] c"T@,V_ivar4\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_PROP_LIST_C2" = internal global { i32, i32, [3 x %struct._prop_t] } { i32 16, i32 3, [3 x %struct._prop_t] [%struct._prop_t { ptr @OBJC_PROP_NAME_ATTR_.24, ptr @OBJC_PROP_NAME_ATTR_.25 }, %struct._prop_t { ptr @OBJC_PROP_NAME_ATTR_.26, ptr @OBJC_PROP_NAME_ATTR_.27 }, %struct._prop_t { ptr @OBJC_PROP_NAME_ATTR_.28, ptr @OBJC_PROP_NAME_ATTR_.29 }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"_OBJC_CLASS_RO_$_C2" = internal global %struct._class_ro_t { i32 0, i32 8, i32 32, ptr null, ptr @OBJC_CLASS_NAME_.11, ptr @"_OBJC_$_INSTANCE_METHODS_C2", ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_C2", ptr null, ptr @"_OBJC_$_PROP_LIST_C2" }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"OBJC_CLASS_$_C2" = global %struct._class_t { ptr @"OBJC_METACLASS_$_C2", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"_OBJC_CLASS_RO_$_C2" }, section "__DATA, __objc_data", align 8
// CHECK-NEXT: @OBJC_CLASS_NAME_.30 = private unnamed_addr constant [3 x i8] c"C3\00", section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.31 = private unnamed_addr constant [4 x i8] c"cm0\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_CLASS_METHODS_C3" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_.31, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01+[C3 cm0]" }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @OBJC_CLASS_NAME_.32 = private unnamed_addr constant [3 x i8] c"P1\00", section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK-NEXT: @OBJC_METH_VAR_NAME_.33 = private unnamed_addr constant [6 x i8] c"p1_m1\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_PROTOCOL_INSTANCE_METHODS_P1" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_.33, ptr @OBJC_METH_VAR_TYPE_, ptr null }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"_OBJC_$_PROTOCOL_METHOD_TYPES_P1" = internal global [1 x ptr] [ptr @OBJC_METH_VAR_TYPE_], section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"_OBJC_PROTOCOL_$_P1" = weak hidden global %struct._protocol_t { ptr null, ptr @OBJC_CLASS_NAME_.32, ptr null, ptr @"_OBJC_$_PROTOCOL_INSTANCE_METHODS_P1", ptr null, ptr null, ptr null, ptr null, i32 96, i32 0, ptr @"_OBJC_$_PROTOCOL_METHOD_TYPES_P1", ptr null, ptr null }, align 8
// CHECK-NEXT: @"_OBJC_LABEL_PROTOCOL_$_P1" = weak hidden global ptr @"_OBJC_PROTOCOL_$_P1", section "__DATA,__objc_protolist,coalesced,no_dead_strip", align 8
// CHECK-NEXT: @"_OBJC_CLASS_PROTOCOLS_$_C3" = internal global { i64, [2 x ptr] } { i64 1, [2 x ptr] [ptr @"_OBJC_PROTOCOL_$_P1", ptr null] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"_OBJC_METACLASS_RO_$_C3" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_.30, ptr @"_OBJC_$_CLASS_METHODS_C3", ptr @"_OBJC_CLASS_PROTOCOLS_$_C3", ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"OBJC_METACLASS_$_C3" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"_OBJC_METACLASS_RO_$_C3" }, section "__DATA, __objc_data", align 8
// CHECK-NEXT: @"_OBJC_$_INSTANCE_METHODS_C3" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_.33, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01-[C3 p1_m1]" }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"_OBJC_CLASS_RO_$_C3" = internal global %struct._class_ro_t { i32 0, i32 8, i32 8, ptr null, ptr @OBJC_CLASS_NAME_.30, ptr @"_OBJC_$_INSTANCE_METHODS_C3", ptr @"_OBJC_CLASS_PROTOCOLS_$_C3", ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"OBJC_CLASS_$_C3" = global %struct._class_t { ptr @"OBJC_METACLASS_$_C3", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"_OBJC_CLASS_RO_$_C3" }, section "__DATA, __objc_data", align 8
// CHECK-NEXT: @OBJC_CLASS_NAME_.34 = private unnamed_addr constant [5 x i8] c"cat3\00", section "__TEXT,__objc_classname,cstring_literals", align 1
// CHECK-NEXT: @"_OBJC_$_CATEGORY_INSTANCE_METHODS_C3_$_cat3" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_.1, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01-[C3(cat3) m1]" }] }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"_OBJC_$_CATEGORY_C3_$_cat3" = internal global %struct._category_t { ptr @OBJC_CLASS_NAME_.34, ptr @"OBJC_CLASS_$_C3", ptr @"_OBJC_$_CATEGORY_INSTANCE_METHODS_C3_$_cat3", ptr null, ptr null, ptr null, ptr null, i32 64 }, section "__DATA, __objc_const", align 8
// CHECK-NEXT: @"OBJC_LABEL_CLASS_$" = private global [3 x ptr] [ptr @"OBJC_CLASS_$_C0", ptr @"OBJC_CLASS_$_C2", ptr @"OBJC_CLASS_$_C3"], section "__DATA,__objc_classlist,regular,no_dead_strip", align 8
// CHECK-NEXT: @"OBJC_LABEL_CATEGORY_$" = private global [1 x ptr] [ptr @"_OBJC_$_CATEGORY_C3_$_cat3"], section "__DATA,__objc_catlist,regular,no_dead_strip", align 8

@interface NSObject {
  id a;
}
@end

__attribute__((availability(domain:feature2, AVAIL))) int unavailable_func1(void);
__attribute__((availability(domain:feature3, AVAIL))) int func3(void);

__attribute__((availability(domain:feature1, AVAIL)))
@interface C0 {
  int ivar0 __attribute__((availability(domain:feature3, AVAIL)));
  int unavailable_ivar1 __attribute__((availability(domain:feature2, AVAIL)));
}
@property int prop0 __attribute__((availability(domain:feature3, AVAIL)));
@property int unavailable_prop1 __attribute__((availability(domain:feature2, AVAIL)));
-(void)m0;
-(void)m1 __attribute__((availability(domain:feature3, AVAIL)));
-(void)unavailable_m2 __attribute__((availability(domain:feature2, AVAIL)));
@end

// CHECK: define internal void @"\01-[C0 m0]"(
// CHECK: define internal void @"\01-[C0 m1]"(
// CHECK: call i32 @func3()
// CHECK-NOT: [C0 m2]

@implementation C0
-(void)m0 {
}
-(void)m1 {
  func3();
}
-(void)unavailable_m2 {
  unavailable_func1();
}
@end

@interface C0(Cat0)
@end

@implementation C0(Cat0)
@end

__attribute__((availability(domain:feature2, AVAIL)))
@interface unavailable_C1
-(void)unavailable_m1;
@end

// CHECK-NOT: [unavailable_C1 m1]
@implementation unavailable_C1
-(void)unavailable_m1 {
}
@end

@interface unavailable_C1(Cat1)
@end

@implementation unavailable_C1(Cat1)
@end

@interface C2 : NSObject
@property id ivar1 __attribute__((availability(domain:feature1, AVAIL)));
@property id unavailable_ivar2 __attribute__((availability(domain:feature2, AVAIL)));
@property id ivar3 __attribute__((availability(domain:feature3, AVAIL)));
@property id ivar4;
@end

// CHECK: define internal ptr @"\01-[C2 ivar1]"(ptr noundef %[[SELF:.*]], ptr noundef %{{.*}})
// CHECK: %[[RETVAL:.*]] = alloca ptr, align 8
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[SELF:.*]], ptr %[[SELF_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: %[[LOAD:.*]] = load atomic i64, ptr %[[ADD_PTR]] unordered, align 8
// CHECK: store i64 %[[LOAD]], ptr %[[RETVAL]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[RETVAL]], align 8
// CHECK: ret ptr %[[V1]]

// CHECK: define internal ptr @"\01-[C2 ivar3]"(ptr noundef %[[SELF:.*]], ptr noundef %{{.*}})
// CHECK: %[[RETVAL:.*]] = alloca ptr, align 8
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[SELF:.*]], ptr %[[SELF_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 16
// CHECK: %[[LOAD:.*]] = load atomic i64, ptr %[[ADD_PTR]] unordered, align 8
// CHECK: store i64 %[[LOAD]], ptr %[[RETVAL]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[RETVAL]], align 8
// CHECK: ret ptr %[[V1]]

// CHECK: define internal ptr @"\01-[C2 ivar4]"(ptr noundef %[[SELF:.*]], ptr noundef %{{.*}})
// CHECK: %[[RETVAL:.*]] = alloca ptr, align 8
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[SELF:.*]], ptr %[[SELF_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 24
// CHECK: %[[LOAD:.*]] = load atomic i64, ptr %[[ADD_PTR]] unordered, align 8
// CHECK: store i64 %[[LOAD]], ptr %[[RETVAL]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[RETVAL]], align 8
// CHECK: ret ptr %[[V1]]

@implementation C2
@end

@protocol P1
-(void)p1_m1;
-(void)unavailable_p2_m2 __attribute__((availability(domain:feature2, AVAIL)));
@end

@interface C3 : NSObject <P1>
@property id prop0 __attribute__((availability(domain:feature2, AVAIL)));
+(void)cm0;
+(void)unavailable_cm1 __attribute__((availability(domain:feature2, AVAIL)));
@end

@implementation C3
-(void)p1_m1 {
}
-(void)unavailable_p2_m2 __attribute__((availability(domain:feature2, AVAIL))) {
  unavailable_func1();
}
+(void)cm0 {
}
+(void)unavailable_cm1 {
  unavailable_func1();
}
@end

@interface C3(cat3)
-(void)unavailable_m0 __attribute__((availability(domain:feature2, AVAIL)));
-(void)m1;
@end

@implementation C3(cat3)
-(void)unavailable_m0 {
}
-(void)m1 {
}
@end
