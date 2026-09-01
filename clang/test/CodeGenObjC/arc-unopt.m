// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -o - %s | FileCheck %s

// A test to ensure that we generate fused calls at -O0.

@class Test0;
Test0 *test0(void) {
  extern Test0 *test0_helper;
  return test0_helper;

  // CHECK:      [[LD:%.*]] = load ptr, ptr @test0_helper
  // CHECK-NEXT: [[T1:%.*]] = tail call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr [[LD]])
  // CHECK-NEXT: ret ptr [[T1]]
}

id test1(void) {
  extern id test1_helper;
  return test1_helper;

  // CHECK:      [[LD:%.*]] = load ptr, ptr @test1_helper
  // CHECK-NEXT: [[T0:%.*]] = tail call ptr @llvm.objc.retainAutoreleaseReturnValue(ptr [[LD]])
  // CHECK-NEXT: ret ptr [[T0]]
}

void test2(void) {
  // CHECK:      [[X:%.*]] = alloca ptr
  // CHECK-NEXT: store ptr null, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[X]])
  // CHECK-NEXT: ret void
  __weak id x;
}

id test3(void) {
  extern id test3_helper(void);
  // CHECK:      [[T0:%.*]] = call ptr @test3_helper()
  // CHECK-NEXT: ret ptr [[T0]]
  return test3_helper();
}

@interface Test4 { id x; } @end
@interface Test4_sub : Test4 { id y; } @end
Test4 *test4(void) {
  extern Test4_sub *test4_helper(void);
  // CHECK:      [[T0:%.*]] = call ptr @test4_helper()
  // CHECK-NEXT: ret ptr [[T0]]
  return test4_helper();
}

@class Test5;
void test5(void) {
  Test5 *x, *y;
  if ((x = y))
    y = 0;

// CHECK-LABEL:    define{{.*}} void @test5()
// CHECK:      [[X:%.*]] = alloca ptr,
// CHECK-NEXT: [[Y:%.*]] = alloca ptr,
// CHECK-NEXT: store ptr null, ptr [[X]],
// CHECK-NEXT: store ptr null, ptr [[Y]],
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]],
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr [[T0]])
// CHECK-NEXT: [[T3:%.*]] = icmp ne ptr [[T0]], null
// CHECK-NEXT: br i1 [[T3]],
}
