// RUN: %clang_cc1 -triple armv7-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -emit-llvm -fblocks -fobjc-arc -o - %s | FileCheck %s

// use an autorelease marker on ARM64.

id test0(void) {
  extern id test0_helper(void);
  // CHECK:      [[T0:%.*]] = call [[CC:(arm_aapcscc )?]]ptr @test0_helper()
  // CHECK-NEXT: ret ptr [[T0]]
  return test0_helper();
}

void test1(void) {
  extern id test1_helper(void);
  // CHECK:      [[T0:%.*]] = call [[CC]]ptr @test1_helper()
  // CHECK-NEXT: call void asm sideeffect "mov\09{{fp, fp|r7, r7}}\09\09// marker for objc_retainAutoreleaseReturnValue"
  // CHECK-NEXT: [[T1:%.*]] = call [[CC]]ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T1]],
  // CHECK-NEXT: call [[CC]]void @llvm.objc.storeStrong(
  // CHECK-NEXT: ret void
  id x = test1_helper();
}

@class A;
A *test2(void) {
  extern A *test2_helper(void);
  // CHECK:      [[T0:%.*]] = call [[CC]]ptr @test2_helper()
  // CHECK-NEXT: ret ptr [[T0]]
  return test2_helper();
}

id test3(void) {
  extern A *test3_helper(void);
  // CHECK:      [[T0:%.*]] = call [[CC]]ptr @test3_helper()
  // CHECK-NEXT: ret ptr [[T0]]
  return test3_helper();
}
