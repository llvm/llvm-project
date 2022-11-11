// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -emit-llvm -o - %s | FileCheck %s

typedef struct {
  id x;
} S0;

@interface C {
  S0 _p1;
}
@property(nonatomic) S0 nonatomic;
@property S0 atomic0;
@property S0 p1;
-(S0)p1;
-(void)setP1:(S0)s0;
@end

@implementation C
-(S0)p1 {
  return _p1;
}
-(void)setP1:(S0)s0 {
  _p1 = s0;
}
@end

// CHECK: %[[STRUCT_S0:.*]] = type { ptr }

// Check that parameters of user-defined setters are destructed.

// CHECK-LABEL: define internal void @"\01-[C setP1:]"(
// CHECK: %[[S0:.*]] = alloca %[[STRUCT_S0]], align 8
// CHECK: call void @__copy_assignment_8_8_s0(ptr %{{.*}}, ptr %[[S0]])
// CHECK: call void @__destructor_8_s0(ptr %[[S0]])

// CHECK: define internal i64 @"\01-[C nonatomic]"(ptr noundef %[[SELF:.*]], {{.*}})
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_S0]], align 8
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[SELF]], ptr %[[SELF_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[IVAR:.*]] = load i32, ptr @"OBJC_IVAR_$_C._nonatomic", align 8
// CHECK: %[[IVAR_CONV:.*]] = sext i32 %[[IVAR]] to i64
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 %[[IVAR_CONV]]
// CHECK: call void @__copy_constructor_8_8_s0(ptr %[[RETVAL]], ptr %[[ADD_PTR]])
// CHECK-NOT: call
// CHECK: ret i64

// CHECK: define internal void @"\01-[C setNonatomic:]"(ptr noundef %[[SELF:.*]], {{.*}}, i64 %[[NONATOMIC_COERCE:.*]])
// CHECK: %[[NONATOMIC:.*]] = alloca %[[STRUCT_S0]], align 8
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_S0]], ptr %[[NONATOMIC]], i32 0, i32 0
// CHECK: %[[COERCE_VAL_IP:.*]] = inttoptr i64 %[[NONATOMIC_COERCE]] to ptr
// CHECK: store ptr %[[COERCE_VAL_IP]], ptr %[[COERCE_DIVE]], align 8
// CHECK: store ptr %[[SELF]], ptr %[[SELF_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[IVAR:.*]] = load i32, ptr @"OBJC_IVAR_$_C._nonatomic", align 8
// CHECK: %[[IVAR_CONV:.*]] = sext i32 %[[IVAR]] to i64
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 %[[IVAR_CONV]]
// CHECK: call void @__move_assignment_8_8_s0(ptr %[[ADD_PTR]], ptr %[[NONATOMIC]])
// CHECK-NOT: call
// CHECK: ret void

// CHECK-LABEL: define internal i64 @"\01-[C atomic0]"(
// CHECK: call void @objc_copyCppObjectAtomic({{.*}}, {{.*}}, ptr noundef @__copy_constructor_8_8_s0)
// CHECK-NOT: call
// CHECK: ret i64

// CHECK-LABEL: define internal void @"\01-[C setAtomic0:]"(
// CHECK: call void @objc_copyCppObjectAtomic({{.*}}, {{.*}}, ptr noundef @__move_assignment_8_8_s0)
// CHECK-NOT: call
// CHECK: ret void
