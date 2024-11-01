// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm --std=c++17 -fcxx-exceptions -fexceptions -discard-value-names %s -o - | FileCheck %s

struct Q { Q(); };
struct R { R(Q); ~R(); };
struct S { S(Q); ~S(); };
struct T : R, S {};

Q q;
T t { R{q}, S{q} };

// CHECK-LABEL: define internal void @__cxx_global_var_init.1() {{.*}} {
// CHECK-NEXT: [[TMP_R:%[a-z0-9.]+]] = alloca %struct.R, align 1
// CHECK-NEXT: [[TMP_Q1:%[a-z0-9.]+]] = alloca %struct.Q, align 1
// CHECK-NEXT: [[TMP_S:%[a-z0-9.]+]] = alloca %struct.S, align 1
// CHECK-NEXT: [[TMP_Q2:%[a-z0-9.]+]] = alloca %struct.Q, align 1
// CHECK-NEXT: [[XPT:%[a-z0-9.]+]] = alloca ptr
// CHECK-NEXT: [[SLOT:%[a-z0-9.]+]] = alloca i32
// CHECK-NEXT: [[ACTIVE:%[a-z0-9.]+]] = alloca i1, align 1
// CHECK-NEXT: call void @_ZN1RC1E1Q(ptr {{[^,]*}} [[TMP_R]])
// CHECK-NEXT: store i1 true, ptr [[ACTIVE]], align 1
// CHECK-NEXT: invoke void @_ZN1SC1E1Q(ptr {{[^,]*}} [[TMP_S]])
// CHECK-NEXT:   to label %[[L1:[a-z0-9.]+]] unwind label %[[L2:[a-z0-9.]+]]
// CHECK-EMPTY:
// CHECK-NEXT: [[L1]]:
// CHECK-NEXT: store i1 false, ptr [[ACTIVE]], align 1
// CHECK-NEXT: call void @_ZN1SD1Ev(ptr
// CHECK-NEXT: call void @_ZN1RD1Ev(ptr
// CHECK-NEXT: [[EXIT:%[a-z0-9.]+]] = call i32 @__cxa_atexit(
// CHECK-NEXT: ret void
// CHECK-EMPTY:
// CHECK-NEXT: [[L2]]:
// CHECK-NEXT: [[LP:%[a-z0-9.]+]] = landingpad { ptr, i32 }
// CHECK-NEXT:                      cleanup
// CHECK-NEXT: [[X1:%[a-z0-9.]+]] = extractvalue { ptr, i32 } [[LP]], 0
// CHECK-NEXT: store ptr [[X1]], ptr [[XPT]], align 8
// CHECK-NEXT: [[X2:%[a-z0-9.]+]] = extractvalue { ptr, i32 } [[LP]], 1
// CHECK-NEXT: store i32 [[X2]], ptr [[SLOT]], align 4
// CHECK-NEXT: [[IS_ACT:%[a-z0-9.]+]] = load i1, ptr [[ACTIVE]], align 1
// CHECK-NEXT: br i1 [[IS_ACT]], label %[[L3:[a-z0-9.]+]], label %[[L4:[a-z0-9.]+]]
// CHECK-EMPTY:
// CHECK-NEXT: [[L3]]:
// CHECK-NEXT: call void @_ZN1RD1Ev(ptr
// CHECK-NEXT: br label %[[L4]]
// CHECK-EMPTY:
// CHECK-NEXT: [[L4]]:
// CHECK-NEXT: call void @_ZN1RD1Ev(ptr {{[^,]*}} [[TMP_R]])
// CHECK-NEXT: br label %[[L5:[a-z0-9.]+]]
// CHECK-EMPTY:
// CHECK-NEXT: [[L5]]:
// CHECK-NEXT: [[EXN:%[a-z0-9.]+]] = load ptr, ptr [[XPT]], align 8
// CHECK-NEXT: [[SEL:%[a-z0-9.]+]] = load i32, ptr [[SLOT]], align 4
// CHECK-NEXT: [[LV1:%[a-z0-9.]+]] = insertvalue { ptr, i32 } poison, ptr [[EXN]], 0
// CHECK-NEXT: [[LV2:%[a-z0-9.]+]] = insertvalue { ptr, i32 } [[LV1]], i32 [[SEL]], 1
// CHECK-NEXT: resume { ptr, i32 } [[LV2]]
// CHECK-NEXT: }
