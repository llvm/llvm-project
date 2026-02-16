// RUN: %clang_cc1 -flto -flto-unit -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-vcall,signed-integer-overflow  -fsanitize-trap=cfi-vcall,signed-integer-overflow -fsanitize-trap-loop -emit-llvm -o - %s | FileCheck %s

struct A {
  virtual void f();
};

void vcall(A *a) {
  // CHECK: [[TEST:%.*]] = call i1 @llvm.type.test
  // CHECK-NEXT: [[NOT:%.*]] = xor i1 [[TEST]], true
  // CHECK-NEXT: call void @llvm.cond.loop(i1 [[NOT]])
  a->f();
}

int overflow(int a, int b) {
  // CHECK: [[OVERFLOW:%.*]] = extractvalue { i32, i1 } %2, 1, !nosanitize
  // CHECK-NEXT: [[NOTOVERFLOW:%.*]] = xor i1 [[OVERFLOW]], true, !nosanitize
  // CHECK-NEXT: [[NOTNOTOVERFLOW:%.*]] = xor i1 [[NOTOVERFLOW]], true, !nosanitize
  // CHECK-NEXT: call void @llvm.cond.loop(i1 [[NOTNOTOVERFLOW]])
  return a + b;
}
