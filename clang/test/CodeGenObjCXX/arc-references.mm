// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -O2 -disable-llvm-optzns -o - %s | FileCheck %s

@interface A
@end

id getObject();
void callee();

// Lifetime extension for binding a reference to an rvalue
// CHECK: define void @_Z5test0v()
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
// CHECK: define void @_Z5test1RU8__strongP11objc_objectRU6__weakS0_
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
  // CHECK: call i8* @objc_initWeak
  // CHECK-NEXT: store i8**
  const __weak id &ref = strong_id();
  // CHECK-NEXT: call void @_Z6calleev()
  callee();
  // CHECK-NEXT: call void @objc_destroyWeak
  // CHECK-NEXT: ret void
}

// CHECK: define void @_Z5test4RU8__strongP11objc_object
void test4(__strong id &x) {
  // CHECK: call i8* @objc_retain
  __strong A* const &ar = x;
  // CHECK: store i32 17, i32*
  int i = 17;
  // CHECK: call void @objc_release(
  // CHECK: ret void
}

void sink(__strong A* &&);

// CHECK: define void @_Z5test5RU8__strongP11objc_object
void test5(__strong id &x) {
  // CHECK:      [[REFTMP:%.*]] = alloca {{%.*}}*, align 8
  // CHECK:      [[OBJ_ID:%.*]] = call i8* @objc_retain(
  // CHECK-NEXT: [[OBJ_A:%.*]] = bitcast i8* [[OBJ_ID]] to [[A:%[a-zA-Z0-9]+]]*
  // CHECK-NEXT: store [[A]]* [[OBJ_A]], [[A]]** [[REFTMP:%[a-zA-Z0-9]+]]
  // CHECK-NEXT: call void @_Z4sinkOU8__strongP1A
  sink(x);  
  // CHECK-NEXT: [[OBJ_A:%[a-zA-Z0-9]+]] = load [[A]]** [[REFTMP]]
  // CHECK-NEXT: [[OBJ_ID:%[a-zA-Z0-9]+]] = bitcast [[A]]* [[OBJ_A]] to i8*
  // CHECK-NEXT: call void @objc_release
  // CHECK-NEXT: store i32 17, i32
  int i = 17;
  // CHECK-NEXT: ret void
}

// CHECK: define internal void @__cxx_global_var_init(
// CHECK: call i8* @_Z9getObjectv
// CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
const __strong id &global_ref = getObject();

// Note: we intentionally don't release the object.

