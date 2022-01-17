// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm -fobjc-arc -o - %s | FileCheck %s

id makeObject1() __attribute__((ns_returns_retained));
id makeObject2() __attribute__((ns_returns_retained));
void releaseObject(__attribute__((ns_consumed)) id);

// CHECK-LABEL: define{{.*}} void @_Z20basicCorrectnessTestv
void basicCorrectnessTest() {
  // CHECK: [[X:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[OBJ1:%.*]] = call noundef i8* @_Z11makeObject1v()
  // CHECK-NEXT: store i8* [[OBJ1]], i8** [[X]], align 8
  id x = makeObject1();

  // CHECK-NEXT: [[OBJ2:%.*]] = call noundef i8* @_Z11makeObject2v()
  // CHECK-NEXT: call void @_Z13releaseObjectP11objc_object(i8* noundef [[OBJ2]])
  releaseObject(makeObject2());

  // CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[X]], i8* null)
  // CHECK-NEXT: ret void
}


template <typename T>
T makeObjectT1() __attribute__((ns_returns_retained));
template <typename T>
T makeObjectT2() __attribute__((ns_returns_retained));

template <typename T>
void releaseObjectT(__attribute__((ns_consumed)) T);

// CHECK-LABEL: define{{.*}} void @_Z12templateTestv
void templateTest() {
  // CHECK: [[X:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[OBJ1:%.*]] = call noundef i8* @_Z12makeObjectT1IU8__strongP11objc_objectET_v()
  // CHECK-NEXT: store i8* [[OBJ1]], i8** [[X]], align 8
  id x = makeObjectT1<id>();

  // CHECK-NEXT: [[OBJ2:%.*]] = call noundef i8* @_Z12makeObjectT2IU8__strongP11objc_objectET_v()
  // CHECK-NEXT: call void @_Z13releaseObjectP11objc_object(i8* noundef [[OBJ2]])
  releaseObject(makeObjectT2<id>());

  // CHECK-NEXT: [[OBJ3:%.*]] = call noundef i8* @_Z11makeObject1v()
  // CHECK-NEXT: call void @_Z14releaseObjectTIU8__strongP11objc_objectEvT_(i8* noundef [[OBJ3]])
  releaseObjectT(makeObject1());

  // CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[X]], i8* null)
  // CHECK-NEXT: ret void
}

// PR27887
struct ForwardConsumed {
  ForwardConsumed(__attribute__((ns_consumed)) id x);
};

ForwardConsumed::ForwardConsumed(__attribute__((ns_consumed)) id x) {}

// CHECK: define{{.*}} void @_ZN15ForwardConsumedC2EP11objc_object(
// CHECK-NOT:  objc_retain
// CHECK:      store i8* {{.*}}, i8** [[X:%.*]],
// CHECK-NOT:  [[X]]
// CHECK:      call void @llvm.objc.storeStrong(i8** [[X]], i8* null)

// CHECK: define{{.*}} void @_ZN15ForwardConsumedC1EP11objc_object(
// CHECK-NOT:  objc_retain
// CHECK:      store i8* {{.*}}, i8** [[X:%.*]],
// CHECK:      [[T0:%.*]] = load i8*, i8** [[X]],
// CHECK-NEXT: store i8* null, i8** [[X]],
// CHECK-NEXT: call void @_ZN15ForwardConsumedC2EP11objc_object({{.*}}, i8* noundef [[T0]])
// CHECK:      call void @llvm.objc.storeStrong(i8** [[X]], i8* null)
