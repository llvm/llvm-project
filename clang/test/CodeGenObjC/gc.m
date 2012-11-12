// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-gc -emit-llvm -o - %s | FileCheck %s

void test0(void) {
  extern id test0_helper(void);
  __attribute__((objc_precise_lifetime)) id x = test0_helper();
  test0_helper();
  // CHECK: define void @test0()
  // CHECK:      [[T0:%.*]] = call i8* @test0_helper()
  // CHECK-NEXT: store i8* [[T0]], i8** [[X:%.*]], align 8
  // CHECK-NEXT: call i8* @test0_helper()
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[X]], align 8
  // CHECK-NEXT: call void asm sideeffect "", "r"(i8* [[T0]]) nounwind
  // CHECK-NEXT: ret void
}
