// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

int f1() {
  int i;
  return i;
}

// CHECK: module
// CHECK: cir.func @f1() -> !s32i
// CHECK:    %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:    %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"] {alignment = 4 : i64}
// CHECK:    %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.store %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.return %[[R]] : !s32i

int f2() {
  const int i = 2;
  return i;
}

// CHECK: cir.func @f2() -> !s32i
// CHECK:    %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:    %[[I_PTR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init, const] {alignment = 4 : i64}
// CHECK:    %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CHECK:    cir.store %[[TWO]], %[[I_PTR]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.store %[[I]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:    %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.return %[[R]] : !s32i

int f3(int i) {
  return i;
}

// CHECK: cir.func @f3(%[[ARG:.*]]: !s32i loc({{.*}})) -> !s32i
// CHECK:   %[[ARG_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CHECK:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   cir.store %[[ARG]], %[[ARG_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[ARG_VAL:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store %[[ARG_VAL]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.return %[[R]] : !s32i

int f4(const int i) {
  return i;
}

// CHECK: cir.func @f4(%[[ARG:.*]]: !s32i loc({{.*}})) -> !s32i
// CHECK:   %[[ARG_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init, const] {alignment = 4 : i64}
// CHECK:   %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:   cir.store %[[ARG]], %[[ARG_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[ARG_VAL:.*]] = cir.load %[[ARG_ALLOCA]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.store %[[ARG_VAL]], %[[RV]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[R:.*]] = cir.load %[[RV]] : !cir.ptr<!s32i>, !s32i
// CHECK:   cir.return %[[R]] : !s32i
