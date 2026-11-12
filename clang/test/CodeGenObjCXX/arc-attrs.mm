// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm -fobjc-arc -o - %s | FileCheck %s

id makeObject1() __attribute__((ns_returns_retained));
id makeObject2() __attribute__((ns_returns_retained));
void releaseObject(__attribute__((ns_consumed)) id);

// CHECK-LABEL: define{{.*}} void @_Z20basicCorrectnessTestv
void basicCorrectnessTest() {
  // CHECK: [[X:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[OBJ1:%.*]] = call noundef ptr @_Z11makeObject1v()
  // CHECK-NEXT: store ptr [[OBJ1]], ptr [[X]], align 8
  id x = makeObject1();

  // CHECK-NEXT: [[OBJ2:%.*]] = call noundef ptr @_Z11makeObject2v()
  // CHECK-NEXT: call void @_Z13releaseObjectP11objc_object(ptr noundef [[OBJ2]])
  releaseObject(makeObject2());

  // CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr null)
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
  // CHECK: [[X:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[OBJ1:%.*]] = call noundef ptr @_Z12makeObjectT1IU8__strongP11objc_objectET_v()
  // CHECK-NEXT: store ptr [[OBJ1]], ptr [[X]], align 8
  id x = makeObjectT1<id>();

  // CHECK-NEXT: [[OBJ2:%.*]] = call noundef ptr @_Z12makeObjectT2IU8__strongP11objc_objectET_v()
  // CHECK-NEXT: call void @_Z13releaseObjectP11objc_object(ptr noundef [[OBJ2]])
  releaseObject(makeObjectT2<id>());

  // CHECK-NEXT: [[OBJ3:%.*]] = call noundef ptr @_Z11makeObject1v()
  // CHECK-NEXT: call void @_Z14releaseObjectTIU8__strongP11objc_objectEvT_(ptr noundef [[OBJ3]])
  releaseObjectT(makeObject1());

  // CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr null)
  // CHECK-NEXT: ret void
}

// PR27887
struct ForwardConsumed {
  ForwardConsumed(__attribute__((ns_consumed)) id x);
};

ForwardConsumed::ForwardConsumed(__attribute__((ns_consumed)) id x) {}

// CHECK: define{{.*}} void @_ZN15ForwardConsumedC2EP11objc_object(
// CHECK-NOT:  objc_retain
// CHECK:      store ptr %x, ptr [[X:%.*]],
// CHECK-NOT:  [[X]]
// CHECK:      call void @llvm.objc.storeStrong(ptr [[X]], ptr null)

// CHECK: define{{.*}} void @_ZN15ForwardConsumedC1EP11objc_object(
// CHECK-NOT:  objc_retain
// CHECK:      store ptr %x, ptr [[X:%.*]],
// CHECK:      [[T0:%.*]] = load ptr, ptr [[X]],
// CHECK-NEXT: store ptr null, ptr [[X]],
// CHECK-NEXT: call void @_ZN15ForwardConsumedC2EP11objc_object({{.*}}, ptr noundef [[T0]])
// CHECK:      call void @llvm.objc.storeStrong(ptr [[X]], ptr null)
