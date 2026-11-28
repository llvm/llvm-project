// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -O1 -relaxed-aliasing -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-apple-darwin %s -emit-llvm -O1 -relaxed-aliasing -o - | FileCheck %s %}

enum e1 {e1_a = -1 };
enum e1 g1(enum e1 *x) {
  return *x;
}

// CHECK-LABEL: define{{.*}} i32 @g1
// CHECK-SAME: (ptr {{.*}}%[[X:[^)]*]])
// CHECK: load i32, ptr %[[X]], align 4
// CHECK-NOT: range
// CHECK: ret
