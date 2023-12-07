// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -O2 -disable-llvm-passes -o - %s | FileCheck %s

@interface A
@end

id getObject();
void callee();

// Lifetime extension for binding a reference to an rvalue
// CHECK-LABEL: define{{.*}} void @_Z5test0v()
void test0() {
  // CHECK: call noundef ptr @_Z9getObjectv{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(
  const __strong id &ref1 = getObject();
  // CHECK: call void @_Z6calleev
  callee();
  // CHECK: call noundef ptr @_Z9getObjectv{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(
  // CHECK-NEXT: call ptr @llvm.objc.autorelease
  const __autoreleasing id &ref2 = getObject();
  // CHECK: call void @_Z6calleev
  callee();
  // CHECK: call void @llvm.objc.release
  // CHECK: ret
}

// No lifetime extension when we're binding a reference to an lvalue.
// CHECK-LABEL: define{{.*}} void @_Z5test1RU8__strongP11objc_objectRU6__weakS0_
void test1(__strong id &x, __weak id &y) {
  // CHECK-NOT: release
  const __strong id &ref1 = x;
  const __autoreleasing id &ref2 = x;
  const __weak id &ref3 = y;
  // CHECK: ret void
}

typedef __strong id strong_id;

//CHECK: define{{.*}} void @_Z5test3v
void test3() {
  // CHECK: [[REF:%.*]] = alloca ptr, align 8
  // CHECK: call ptr @llvm.objc.initWeak
  // CHECK-NEXT: store ptr
  const __weak id &ref = strong_id();
  // CHECK-NEXT: call void @_Z6calleev()
  callee();
  // CHECK-NEXT: call void @llvm.objc.destroyWeak
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[REF]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z5test4RU8__strongP11objc_object
void test4(__strong id &x) {
  // CHECK: call ptr @llvm.objc.retain
  __strong A* const &ar = x;
  // CHECK: store i32 17, ptr
  int i = 17;
  // CHECK: call void @llvm.objc.release(
  // CHECK: ret void
}

void sink(__strong A* &&);

// CHECK-LABEL: define{{.*}} void @_Z5test5RU8__strongP11objc_object
void test5(__strong id &x) {
  // CHECK:      [[REFTMP:%.*]] = alloca ptr, align 8
  // CHECK:      [[I:%.*]] = alloca i32, align 4
  // CHECK:      [[OBJ_ID:%.*]] = call ptr @llvm.objc.retain(
  // CHECK-NEXT: store ptr [[OBJ_ID]], ptr [[REFTMP:%[a-zA-Z0-9]+]]
  // CHECK-NEXT: call void @_Z4sinkOU8__strongP1A
  sink(x);  
  // CHECK-NEXT: [[OBJ_A:%[a-zA-Z0-9]+]] = load ptr, ptr [[REFTMP]]
  // CHECK-NEXT: call void @llvm.objc.release
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 4, ptr [[I]])
  // CHECK-NEXT: store i32 17, ptr
  int i = 17;
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4, ptr [[I]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define internal void @__cxx_global_var_init(
// CHECK: call noundef ptr @_Z9getObjectv{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT: call void (...) @llvm.objc.clang.arc.noop.use(
const __strong id &global_ref = getObject();

// Note: we intentionally don't release the object.

