// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -emit-llvm -o - %s | FileCheck %s
struct S {
  int s;
  int e;
};

static constexpr struct S V = {
    .s = 9,
    .e = (1ULL << V.s),
};

extern void f(int *);

// CHECK-LABEL: define{{.*}} void @t1()
// CHECK: %arr = alloca [10 x i32], align 16
void t1(void) {
    int arr[10];
    f(arr);
}

// CHECK-LABEL: define{{.*}} void @t2()
// CHECK: %arr = alloca [512 x i32], align 16
void t2(void) {
    int arr[V.e];
    f(arr);
}

// CHECK-LABEL: define{{.*}} void @t3(
// CHECK: %saved_stack = alloca ptr
// CHECK: %vla = alloca i32, i64
void t3(int n) {
  int arr[n];
  f(arr);
}
