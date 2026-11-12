// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-runtime-has-weak -o - -fobjc-arc-exceptions %s | FileCheck %s

@class Ety;

// These first four tests are all PR11732.

void test0_helper(void);
void test0(void) {
  @try {
    test0_helper();
  } @catch (Ety *e) {
  }
}
// CHECK-LABEL: define{{.*}} void @_Z5test0v()
// CHECK:      [[E:%e]] = alloca ptr, align 8
// CHECK-NEXT: invoke void @_Z12test0_helperv()
// CHECK:      [[T0:%.*]] = call ptr @objc_begin_catch(
// CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]]) [[NUW:#[0-9]+]]
// CHECK-NEXT: store ptr [[T3]], ptr [[E]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[E]], ptr null) [[NUW]]
// CHECK-NEXT: call void @objc_end_catch() [[NUW]]

void test1_helper(void);
void test1(void) {
  @try {
    test1_helper();
  } @catch (__weak Ety *e) {
  }
}
// CHECK-LABEL: define{{.*}} void @_Z5test1v()
// CHECK:      [[E:%e]] = alloca ptr, align 8
// CHECK-NEXT: invoke void @_Z12test1_helperv()
// CHECK:      [[T0:%.*]] = call ptr @objc_begin_catch(
// CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[E]], ptr [[T0]]) [[NUW]]
// CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[E]]) [[NUW]]
// CHECK-NEXT: call void @objc_end_catch() [[NUW]]

void test2_helper(void);
void test2(void) {
  try {
    test2_helper();
  } catch (Ety *e) {
  }
}
// CHECK-LABEL: define{{.*}} void @_Z5test2v()
// CHECK:      [[E:%e]] = alloca ptr, align 8
// CHECK-NEXT: invoke void @_Z12test2_helperv()
// CHECK:      [[T0:%.*]] = call ptr @__cxa_begin_catch(
// CHECK-NEXT: [[T3:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]]) [[NUW]]
// CHECK-NEXT: store ptr [[T3]], ptr [[E]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[E]], ptr null) [[NUW]]
// CHECK-NEXT: call void @__cxa_end_catch() [[NUW]]

void test3_helper(void);
void test3(void) {
  try {
    test3_helper();
  } catch (Ety * __weak e) {
  }
}
// CHECK-LABEL: define{{.*}} void @_Z5test3v()
// CHECK:      [[E:%e]] = alloca ptr, align 8
// CHECK-NEXT: invoke void @_Z12test3_helperv()
// CHECK:      [[T0:%.*]] = call ptr @__cxa_begin_catch(
// CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[E]], ptr [[T0]]) [[NUW]]
// CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[E]]) [[NUW]]
// CHECK-NEXT: call void @__cxa_end_catch() [[NUW]]

namespace test4 {
  struct A {
    id single;
    id array[2][3];

    A();
  };

  A::A() {
    throw 0;
  }
  // CHECK-LABEL:    define{{.*}} void @_ZN5test41AC2Ev(
  // CHECK:      [[THIS:%.*]] = load ptr, ptr {{%.*}}
  //   Construct single.
  // CHECK-NEXT: [[SINGLE:%.*]] = getelementptr inbounds nuw [[A:%.*]], ptr [[THIS]], i32 0, i32 0
  // CHECK-NEXT: store ptr null, ptr [[SINGLE]], align 8
  //   Construct array.
  // CHECK-NEXT: [[ARRAY:%.*]] = getelementptr inbounds nuw [[A:%.*]], ptr [[THIS]], i32 0, i32 1
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 8 [[ARRAY]], i8 0, i64 48, i1 false)
  //   throw 0;
  // CHECK:      invoke void @__cxa_throw(
  //   Landing pad from throw site:
  // CHECK:      landingpad
  //     - First, destroy all of array.
  // CHECK:      [[ARRAYBEGIN:%.*]] = getelementptr inbounds [2 x [3 x ptr]], ptr [[ARRAY]], i32 0, i32 0, i32 0
  // CHECK-NEXT: [[ARRAYEND:%.*]] = getelementptr inbounds ptr, ptr [[ARRAYBEGIN]], i64 6
  // CHECK-NEXT: br label
  // CHECK:      [[AFTER:%.*]] = phi ptr [ [[ARRAYEND]], {{%.*}} ], [ [[ELT:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[ELT]] = getelementptr inbounds ptr, ptr [[AFTER]], i64 -1
  // CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[ELT]], ptr null) [[NUW]]
  // CHECK-NEXT: [[DONE:%.*]] = icmp eq ptr [[ELT]], [[ARRAYBEGIN]]
  // CHECK-NEXT: br i1 [[DONE]],
  //     - Next, destroy single.
  // CHECK:      call void @llvm.objc.storeStrong(ptr [[SINGLE]], ptr null) [[NUW]]
  // CHECK:      br label
  // CHECK:      resume
}

__attribute__((ns_returns_retained)) id test5_helper(unsigned);
void test5(void) {
  id array[][2] = {
    test5_helper(0),
    test5_helper(1),
    test5_helper(2),
    test5_helper(3)
  };
}
// CHECK-LABEL: define{{.*}} void @_Z5test5v()
// CHECK:       [[ARRAY:%.*]] = alloca [2 x [2 x ptr]], align
// CHECK:       store ptr [[ARRAY]],
// CHECK-NEXT:  store ptr [[ARRAY]],
// CHECK-NEXT:  [[T0:%.*]] = invoke noundef ptr @_Z12test5_helperj(i32 noundef 0)
// CHECK:       store ptr [[T0]], ptr [[ARRAY]], align
// CHECK-NEXT:  [[A01:%.*]] = getelementptr inbounds ptr, ptr [[ARRAY]], i64 1
// CHECK-NEXT:  store ptr [[A01]],
// CHECK-NEXT:  [[T0:%.*]] = invoke noundef ptr @_Z12test5_helperj(i32 noundef 1)
// CHECK:       store ptr [[T0]], ptr [[A01]], align
// CHECK-NEXT:  [[A1:%.*]] = getelementptr inbounds [2 x ptr], ptr [[ARRAY]], i64 1
// CHECK-NEXT:  store ptr [[A1]],
// CHECK-NEXT:  store ptr [[A1]],
// CHECK-NEXT:  [[T0:%.*]] = invoke noundef ptr @_Z12test5_helperj(i32 noundef 2)
// CHECK:       store ptr [[T0]], ptr [[A1]], align
// CHECK-NEXT:  [[A11:%.*]] = getelementptr inbounds ptr, ptr [[A1]], i64 1
// CHECK-NEXT:  store ptr [[A11]],
// CHECK-NEXT:  [[T0:%.*]] = invoke noundef ptr @_Z12test5_helperj(i32 noundef 3)
// CHECK:       store ptr [[T0]], ptr [[A11]], align

// CHECK: attributes [[NUW]] = { nounwind }
