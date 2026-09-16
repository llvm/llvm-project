// RUN: %clang_cc1 -emit-llvm -x c %s -triple x86_64-unknown-linux-gnu -O1 -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -x c++ %s -triple x86_64-unknown-linux-gnu -O1 -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} @{{.*}}with_noipa{{.*}}(
// CHECK-SAME: ) {{.*}}#0 {
__attribute__((noipa)) int with_noipa(int x) {
  return x + 1;
}

// CHECK-LABEL: define {{.*}} @{{.*}}without_noipa{{.*}}(
// CHECK-SAME: ) {{.*}}#1 {
int without_noipa(int x) {
  return x - 1;
}


static inline __attribute__((noipa, always_inline)) int always_inline_callee(int x) {
  return x * 3 + 1;
}

// CHECK-LABEL: define {{.*}} @{{.*}}always_inline_caller{{.*}}(
// CHECK-NOT: call {{.*}}always_inline_callee{{.*}}
// CHECK: mul nsw i32
// CHECK: add nsw i32
// CHECK: ret i32
int always_inline_caller(int x) {
  return always_inline_callee(x);
}

// CHECK: attributes #0 = { {{.*}}noipa noinline
