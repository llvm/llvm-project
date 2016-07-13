// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// Check that no empty blocks are generated for nested ifs.

extern void func();

int f0(int val) {
  if (val == 0) {
    func();
  } else if (val == 1) {
    func();
  }
  return 0;
}

// CHECK-LABEL: define {{.*}}i32 @f0
// CHECK: call void {{.*}} @func
// CHECK: call void {{.*}} @func
// CHECK: br label %[[RETBLOCK1:[^ ]*]]
// CHECK: [[RETBLOCK1]]:
// CHECK-NOT: br label
// CHECK: ret i32

int f1(int val, int g) {
  if (val == 0)
    if (g == 1) {
      func();
    }
  return 0;
}

// CHECK-LABEL: define {{.*}}i32 @f1
// CHECK: call void {{.*}} @func
// CHECK: br label %[[RETBLOCK2:[^ ]*]]
// CHECK: [[RETBLOCK2]]:
// CHECK-NOT: br label
// CHECK: ret i32
