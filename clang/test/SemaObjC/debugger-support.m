// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fdebugger-support %s -emit-llvm -o - | FileCheck %s

// rdar://problem/9416370
void test0(id x) {
  struct A { int w, x, y, z; };
  struct A result = (struct A) [x makeStruct];
  // CHECK:     define{{.*}} void @test0(
  // CHECK:      [[X:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[RESULT:%.*]] = alloca [[A:%.*]], align 4
  // CHECK-NEXT: store ptr {{%.*}}, ptr [[X]],
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]],
  // CHECK-NEXT: [[T1:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[T2:%.*]] = call { i64, i64 } @objc_msgSend(ptr noundef [[T0]], ptr noundef [[T1]])
}
