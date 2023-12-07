// RUN: %clang_cc1 -triple i386-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-iossimulator6.0 -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

// implement objc_retainAutoreleasedReturnValue on i386

// CHECK-LABEL: define{{.*}} ptr @test0()
id test0(void) {
  extern id test0_helper(void);
  // CHECK:      [[T0:%.*]] = call ptr @test0_helper()
  // CHECK-NEXT: ret ptr [[T0]]
  return test0_helper();
}

// CHECK-LABEL: define{{.*}} void @test1()
void test1(void) {
  extern id test1_helper(void);
  // CHECK:      [[T0:%.*]] = call ptr @test1_helper()
  // CHECK-NEXT: call void asm sideeffect "mov
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T1]],
  // CHECK-NEXT: call void @llvm.objc.storeStrong(
  // CHECK-NEXT: ret void
  id x = test1_helper();
}

// CHECK-LABEL: define {{.*}} @test2()
@class A;
A *test2(void) {
  extern A *test2_helper(void);
  // CHECK:      [[T0:%.*]] = call ptr @test2_helper()
  // CHECK-NEXT: ret ptr [[T0]]
  return test2_helper();
}

// CHECK-LABEL: define{{.*}} ptr @test3()
id test3(void) {
  extern A *test3_helper(void);
  // CHECK:      [[T0:%.*]] = call ptr @test3_helper()
  // CHECK-NEXT: ret ptr [[T0]]
  return test3_helper();
}
