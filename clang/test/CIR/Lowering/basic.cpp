// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s

int f1() {
  int i;
  return i;
}

// CHECK: define{{.*}} i32 @_Z2f1v(){{.*}} {
// CHECK:    %[[RV:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[I_PTR:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// CHECK:    store i32 %[[I]], ptr %[[RV]], align 4
// CHECK:    %[[R:.*]] = load i32, ptr %[[RV]], align 4
// CHECK:    ret i32 %[[R]]

int f2() {
  const int i = 2;
  return i;
}

// CHECK: define{{.*}} i32 @_Z2f2v(){{.*}} {
// CHECK:    %[[RV:.*]] = alloca i32, i64 1, align 4
// CHECK:    %[[I_PTR:.*]] = alloca i32, i64 1, align 4
// CHECK:    store i32 2, ptr %[[I_PTR]], align 4
// CHECK:    %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// CHECK:    store i32 %[[I]], ptr %[[RV]], align 4
// CHECK:    %[[R:.*]] = load i32, ptr %[[RV]], align 4
// CHECK:    ret i32 %[[R]]

int f3(int i) {
    return i;
  }

// CHECK: define{{.*}} i32 @_Z2f3i(i32 %[[ARG:.*]])
// CHECK:   %[[ARG_ALLOCA:.*]] = alloca i32, i64 1, align 4
// CHECK:   %[[RV:.*]] = alloca i32, i64 1, align 4
// CHECK:   store i32 %[[ARG]], ptr %[[ARG_ALLOCA]], align 4
// CHECK:   %[[ARG_VAL:.*]] = load i32, ptr %[[ARG_ALLOCA]], align 4
// CHECK:   store i32 %[[ARG_VAL]], ptr %[[RV]], align 4
// CHECK:   %[[R:.*]] = load i32, ptr %[[RV]], align 4
// CHECK:   ret i32 %[[R]]

int f4(const int i) {
  return i;
}

// CHECK: define{{.*}} i32 @_Z2f4i(i32 %[[ARG:.*]])
// CHECK:   %[[ARG_ALLOCA:.*]] = alloca i32, i64 1, align 4
// CHECK:   %[[RV:.*]] = alloca i32, i64 1, align 4
// CHECK:   store i32 %[[ARG]], ptr %[[ARG_ALLOCA]], align 4
// CHECK:   %[[ARG_VAL:.*]] = load i32, ptr %[[ARG_ALLOCA]], align 4
// CHECK:   store i32 %[[ARG_VAL]], ptr %[[RV]], align 4
// CHECK:   %[[R:.*]] = load i32, ptr %[[RV]], align 4
// CHECK:   ret i32 %[[R]]
