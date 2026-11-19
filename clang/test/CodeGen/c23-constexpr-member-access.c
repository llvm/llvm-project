// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-unknown-linux-gnu -std=c23 -emit-llvm -o - %s | FileCheck %s %}
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
// CHECK: [[ARR:%.*]] = alloca [10 x i32], align 16
// CHECK: [[DECAY:%.*]] = getelementptr {{.*}}ptr [[ARR]],
// CHECK: call void @f(ptr noundef [[DECAY]])
void t1(void) {
    int arr[10];
    f(arr);
}

// CHECK-LABEL: define{{.*}} void @t2()
// CHECK: [[ARR:%.*]] = alloca [512 x i32], align 16
// CHECK: [[DECAY:%.*]] = getelementptr {{.*}}ptr [[ARR]],
// CHECK: call void @f(ptr noundef [[DECAY]])
void t2(void) {
    int arr[V.e];
    f(arr);
}

// CHECK-LABEL: define{{.*}} void @t3(
// CHECK: [[SAVED:%.*]] = alloca ptr
// CHECK: [[STACK:%.*]] = call ptr @llvm.stacksave.p0()
// CHECK: store ptr [[STACK]], ptr [[SAVED]]
// CHECK: [[VLA:%.*]] = alloca i32, i64
// CHECK: call void @f(ptr noundef [[VLA]])
// CHECK: [[RESTORE:%.*]] = load ptr, ptr [[SAVED]]
// CHECK: call void @llvm.stackrestore.p0(ptr [[RESTORE]])
void t3(int n) {
  int arr[n];
  f(arr);
}
