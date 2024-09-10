// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -fno-assumptions %s -emit-llvm -o - | FileCheck %s --check-prefix=DISABLED

// DISABLED-NOT: @llvm.assume

bool f();

template <typename T>
void f2() {
  [[assume(sizeof(T) == sizeof(int))]];
}

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

  // CHECK-NEXT: [[X1:%.*]] = load i32, ptr [[X_ADDR]]
  // CHECK-NEXT: [[CMP1:%.*]] = icmp ne i32 [[X1]], 27
  // CHECK-NEXT: call void @llvm.assume(i1 [[CMP1]])
  [[assume(x != 27)]];

  // CHECK-NEXT: [[X2:%.*]] = load i32, ptr [[X_ADDR]]
  // CHECK-NEXT: [[Y2:%.*]] = load i32, ptr [[Y_ADDR]]
  // CHECK-NEXT: [[CMP2:%.*]] = icmp eq i32 [[X2]], [[Y2]]
  // CHECK-NEXT: call void @llvm.assume(i1 [[CMP2]])
  [[assume(x == y)]];

  // CHECK-NEXT: call void @_Z2f2IiEvv()
  f2<int>();

  // CHECK-NEXT: call void @_Z2f2IdEvv()
  f2<double>();
}

// CHECK: void @_Z2f2IiEvv()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @llvm.assume(i1 true)

// CHECK: void @_Z2f2IdEvv()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @llvm.assume(i1 false)
