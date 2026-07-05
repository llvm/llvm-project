// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-gc -emit-llvm -o - %s | FileCheck %s

void test0(void) {
  extern id test0_helper(void);
  __attribute__((objc_precise_lifetime)) id x = test0_helper();
  test0_helper();
  // CHECK-LABEL: define{{.*}} void @test0()
  // CHECK:      [[T0:%.*]] = call ptr @test0_helper()
  // CHECK-NEXT: store ptr [[T0]], ptr [[X:%.*]], align 8
  // CHECK-NEXT: call ptr @test0_helper()
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]], align 8
  // CHECK-NEXT: call void asm sideeffect "", "r"(ptr [[T0]]) [[NUW:#[0-9]+]]
  // CHECK-NEXT: ret void
}

// CHECK: attributes [[NUW]] = { nounwind }
