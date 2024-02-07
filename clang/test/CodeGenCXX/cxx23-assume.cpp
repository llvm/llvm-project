// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 %s -emit-llvm -o - | FileCheck %s

bool f();

// CHECK: @_Z1gii(i32 noundef [[X:%.*]], i32 noundef [[Y:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT: [[X_ADDR:%.*]] = alloca i32
// CHECK-NEXT: [[Y_ADDR:%.*]] = alloca i32
// CHECK-NEXT: store i32 [[X]], ptr [[X_ADDR]]
// CHECK-NEXT: store i32 [[Y]], ptr [[Y_ADDR]]
void g(int x, int y) {
  // Not emitted because it has side-effects.
  [[assume(f())]];

  // CHECK-NEXT: call void @llvm.assume(i1 true)
  [[assume((1, 2))]];

  // [[X1:%.*]] = load i32, ptr [[X_ADDR]]
  // [[CMP1:%.*]] = icmp ne i32 [[X1]], 27
  // call void @llvm.assume(i1 [[CMP1]])
  [[assume(x != 27)]];

  // [[X2:%.*]] = load i32, ptr [[X_ADDR]]
  // [[Y2:%.*]] = load i32, ptr [[Y_ADDR]]
  // [[CMP2:%.*]] = icmp eq i32 [[X2]], [[Y2]]
  // call void @llvm.assume(i1 [[CMP2]])
  [[assume(x == y)]];
}
