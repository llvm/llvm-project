// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -O2 -disable-llvm-passes -o - %s | FileCheck %s

@interface A
@end

id getObject();
void callee();

// Lifetime extension for binding a reference to an rvalue
// CHECK-LABEL: define void @_Z5test0v()
void test0() {
  // CHECK: call i8* @_Z9getObjectv
  // CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
  const __strong id &ref1 = getObject();
  // CHECK: call void @_Z6calleev
  callee();
  // CHECK: call i8* @_Z9getObjectv
  // CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
  // CHECK-NEXT: call i8* @objc_autorelease
  const __autoreleasing id &ref2 = getObject();
  // CHECK: call void @_Z6calleev
  callee();
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret
}

// No lifetime extension when we're binding a reference to an lvalue.
// CHECK-LABEL: define void @_Z5test1RU8__strongP11objc_objectRU6__weakS0_
void test1(__strong id &x, __weak id &y) {
  // CHECK-NOT: release
  const __strong id &ref1 = x;
  const __autoreleasing id &ref2 = x;
  const __weak id &ref3 = y;
  // CHECK: ret void
}

typedef __strong id strong_id;

//CHECK: define void @_Z5test3v
void test3() {
  // CHECK: [[REF:%.*]] = alloca i8**, align 8
  // CHECK: call i8* @objc_initWeak
  // CHECK-NEXT: store i8**
  const __weak id &ref = strong_id();
  // CHECK-NEXT: call void @_Z6calleev()
  callee();
  // CHECK-NEXT: [[PTR:%.*]] = bitcast i8*** [[REF]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[PTR]])
  // CHECK-NEXT: call void @objc_destroyWeak
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define void @_Z5test4RU8__strongP11objc_object
void test4(__strong id &x) {
  // CHECK: call i8* @objc_retain
  __strong A* const &ar = x;
  // CHECK: store i32 17, i32*
  int i = 17;
  // CHECK: call void @objc_release(
  // CHECK: ret void
}

void sink(__strong A* &&);

// CHECK-LABEL: define void @_Z5test5RU8__strongP11objc_object
void test5(__strong id &x) {
  // CHECK:      [[REFTMP:%.*]] = alloca {{%.*}}*, align 8
  // CHECK:      [[I:%.*]] = alloca i32, align 4
  // CHECK:      [[OBJ_ID:%.*]] = call i8* @objc_retain(
  // CHECK-NEXT: [[OBJ_A:%.*]] = bitcast i8* [[OBJ_ID]] to [[A:%[a-zA-Z0-9]+]]*
  // CHECK-NEXT: store [[A]]* [[OBJ_A]], [[A]]** [[REFTMP:%[a-zA-Z0-9]+]]
  // CHECK-NEXT: call void @_Z4sinkOU8__strongP1A
  sink(x);  
  // CHECK-NEXT: [[OBJ_A:%[a-zA-Z0-9]+]] = load [[A]]*, [[A]]** [[REFTMP]]
  // CHECK-NEXT: [[OBJ_ID:%[a-zA-Z0-9]+]] = bitcast [[A]]* [[OBJ_A]] to i8*
  // CHECK-NEXT: call void @objc_release
  // CHECK-NEXT: [[IPTR1:%.*]] = bitcast i32* [[I]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 4, i8* [[IPTR1]])
  // CHECK-NEXT: store i32 17, i32
  int i = 17;
  // CHECK-NEXT: [[IPTR2:%.*]] = bitcast i32* [[I]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 4, i8* [[IPTR2]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define internal void @__cxx_global_var_init(
// CHECK: call i8* @_Z9getObjectv
// CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
const __strong id &global_ref = getObject();

// Note: we intentionally don't release the object.

