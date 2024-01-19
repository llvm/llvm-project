// RUN: %clang_cc1 %s -triple arm64-apple-ios11.0 -fobjc-runtime=ios-11.0 -fptrauth-calls -emit-llvm -o - | FileCheck %s

extern int DEFAULT();

struct TCPPObject
{
 TCPPObject();
 ~TCPPObject();
 TCPPObject(const TCPPObject& inObj, int i = DEFAULT());
 TCPPObject& operator=(const TCPPObject& inObj);
 int filler[64];
};


@interface MyDocument
{
@private
 TCPPObject _cppObject;
 TCPPObject _cppObject1;
}
@property (assign, readwrite, atomic) const TCPPObject MyProperty;
@property (assign, readwrite, atomic) const TCPPObject MyProperty1;
@end

@implementation MyDocument
  @synthesize MyProperty = _cppObject;
  @synthesize MyProperty1 = _cppObject1;
@end

// CHECK-LABEL: @__copy_helper_atomic_property_.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @__copy_helper_atomic_property_, i32 0, i64 0, i64 0 }, section "llvm.ptrauth", align 8

// CHECK-LABEL: @__assign_helper_atomic_property_.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @__assign_helper_atomic_property_, i32 0, i64 0, i64 0 }, section "llvm.ptrauth", align 8

// CHECK-LABEL: define internal void @__copy_helper_atomic_property_(ptr noundef %0, ptr noundef %1) #
// CHECK: [[TWO:%.*]] = load ptr, ptr [[ADDR:%.*]], align 8
// CHECK: [[THREE:%.*]] = load ptr, ptr [[ADDR1:%.*]], align 8
// CHECK: [[CALL:%.*]] = call noundef i32 @_Z7DEFAULTv()
// CHECK:  call noundef ptr @_ZN10TCPPObjectC1ERKS_i(ptr noundef nonnull align {{[0-9]+}} dereferenceable(256) [[TWO]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[THREE]], i32 noundef [[CALL]])
// CHECK:  ret void

// CHECK: define internal void @"\01-[MyDocument MyProperty]"(ptr dead_on_unwind noalias writable sret(%{{.*}} align 4 %[[AGG_RESULT:.*]], ptr noundef %[[SELF:.*]],
// CHECK: %[[RESULT_PTR:.*]] = alloca ptr, align 8
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[AGG_RESULT]], ptr %[[RESULT_PTR]], align 8
// CHECK: store ptr %[[SELF]], ptr %[[SELF_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[IVAR:.*]] = load i32, ptr @"OBJC_IVAR_$_MyDocument._cppObject", align 8,
// CHECK: %[[IVAR_CONV:.*]] = sext i32 %[[IVAR]] to i64
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 %[[IVAR_CONV]]
// CHECK: call void @objc_copyCppObjectAtomic(ptr noundef %[[AGG_RESULT]], ptr noundef %[[ADD_PTR]], ptr noundef @__copy_helper_atomic_property_.ptrauth)

// CHECK-LABEL: define internal void @__assign_helper_atomic_property_(ptr noundef %0, ptr noundef %1) #
// CHECK: [[THREE:%.*]] = load ptr, ptr [[ADDR1:%.*]], align 8
// CHECK: [[TWO:%.*]] = load ptr, ptr [[ADDR:%.*]], align 8
// CHECK: [[CALL:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN10TCPPObjectaSERKS_(ptr noundef nonnull align {{[0-9]+}} dereferenceable(256) [[TWO]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[THREE]])
// CHECK:  ret void

// CHECK: define internal void @"\01-[MyDocument setMyProperty:]"(ptr noundef %[[SELF:.*]], ptr noundef %{{.*}}, ptr noundef %[[MYPROPERTY:.*]])
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[MYPROPERTY_INDIRECT:.*]]_addr = alloca ptr, align 8
// CHECK: store ptr %[[SELF]], ptr %[[SELF_ADDR]], align 8
// CHECK: store ptr %[[MYPROPERTY]], ptr %[[MYPROPERTY_INDIRECT]]_addr, align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[IVAR:.*]] = load i32, ptr @"OBJC_IVAR_$_MyDocument._cppObject", align 8,
// CHECK: %[[IVAR_CONV:.*]] = sext i32 %[[IVAR]] to i64
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 %[[IVAR_CONV]]
// CHECK: call void @objc_copyCppObjectAtomic(ptr noundef %[[ADD_PTR]], ptr noundef %[[MYPROPERTY]], ptr noundef @__assign_helper_atomic_property_.ptrauth)
// CHECK: ret void
